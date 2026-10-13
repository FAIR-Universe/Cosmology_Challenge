"""Ingestion-ready Phase 2 weak-lensing autoencoder baseline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
IMAGE_SHAPE = (1424, 176)
TRAIN_MEAN = -0.00012428639456629753
TRAIN_STD = 0.020479273051023483


class _MaskedMaps(Dataset):
    def __init__(self, data: np.ndarray, mask: np.ndarray):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = np.asarray(self.data[index], dtype=np.float32)
        if sample.shape == IMAGE_SHAPE:
            image = sample
        else:
            image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
            image[self.mask] = sample
        image = (image - TRAIN_MEAN) / TRAIN_STD
        return torch.from_numpy(image[None, ...])


class _ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            encoded = self.encoder_conv(torch.zeros(1, 1, *IMAGE_SHAPE))
        self.encoded_shape = tuple(encoded.shape[1:])
        flat_dim = int(np.prod(self.encoded_shape))
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def decode_from_mean(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(x).flatten(1)
        latent_mean = self.fc_mu(encoded)
        decoded = self.fc_decode(latent_mean).view(-1, *self.encoded_shape)
        return self.decoder_conv(decoded)


def _load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def _negative_two_sided_p_value(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference))
    # The notebook uses mean(score < reference), i.e. the empirical survival function.
    survival = (len(reference) - np.searchsorted(reference, scores, side="right")) / len(reference)
    return -2.0 * np.minimum(survival, 1.0 - survival)


class Model:
    """Return autoencoder reconstruction-based OoD scores."""

    def __init__(self):
        requested_device = os.environ.get("WL_DEVICE")
        self.device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(os.environ.get("WL_AE_BATCH_SIZE", "64"))
        self.mask = np.load(HERE / "WIDE12H_bin2_2arcmin_mask.npy", allow_pickle=False)
        self.reference = np.load(HERE / "AE_TS_InD.npy", allow_pickle=False)
        self.network = _ConvAE().to(self.device)
        state = _load_state_dict(HERE / "Phase2_starting_kit_AE_baseline.pth", self.device)
        self.network.load_state_dict(state)
        self.network.eval()

    def predict(self, test_data) -> np.ndarray:
        loader = DataLoader(
            _MaskedMaps(test_data, self.mask),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        chunks = []
        with torch.inference_mode():
            for images in loader:
                images = images.to(self.device, non_blocking=True)
                reconstructed = self.network.decode_from_mean(images)
                loss = F.mse_loss(reconstructed, images, reduction="none").flatten(1).sum(1)
                chunks.append(loss.cpu().numpy())
        scores = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        return _negative_two_sided_p_value(scores, self.reference)
