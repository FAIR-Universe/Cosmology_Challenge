"""Ingestion-ready Phase 2 weak-lensing CNN + MCMC baseline."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
IMAGE_SHAPE = (1424, 176)
TRAIN_MEAN = -0.00012428639456629753
TRAIN_STD = 0.020479273051023483


class _MaskedMaps(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = np.asarray(self.data[index], dtype=np.float32)
        if sample.shape == IMAGE_SHAPE:
            image = sample
        else:
            image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
            image[self.mask] = sample
        image = (image - TRAIN_MEAN) / TRAIN_STD
        return torch.from_numpy(image[None, ...])


class _SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            feature_size = int(np.prod(self.conv_stack(torch.zeros(1, 1, *IMAGE_SHAPE)).shape))
        self.fc_stack = nn.Sequential(
            nn.Flatten(), nn.Linear(feature_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.fc_stack(self.conv_stack(x))


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_pickle(name):
    # These are trusted artifacts created with SciPy by the baseline notebook.
    with open(HERE / name, "rb") as stream:
        return pickle.load(stream)


class Model:
    """Infer cosmology with the CNN, then score an MCMC posterior fit."""

    def __init__(self):
        requested_device = os.environ.get("WL_DEVICE")
        self.device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(os.environ.get("WL_CNN_BATCH_SIZE", "64"))
        self.n_steps = int(os.environ.get("WL_MCMC_STEPS", "10000"))
        self.burn_in = int(0.2 * self.n_steps)
        if self.n_steps <= 0 or self.burn_in >= self.n_steps:
            raise ValueError("WL_MCMC_STEPS must be a positive integer")
        self.mask = np.load(HERE / "WIDE12H_bin2_2arcmin_mask.npy", allow_pickle=False)
        labels = np.load(HERE / "label_train.npy", allow_pickle=False)
        self.label_mean = labels.mean(axis=0)
        self.label_scale = labels.std(axis=0)
        full_labels = np.load(HERE / "label_newrealization.npy", mmap_mode="r", allow_pickle=False)
        self.cosmology = np.asarray(full_labels[:, 0, :2])
        self.reference = np.load(HERE / "CNN_TS_InD.npy", allow_pickle=False)
        self.mean_interp = _load_pickle("CNN_mean_d_vector_interp.pkl")
        self.cov_interp = _load_pickle("CNN_cov_d_vector_interp.pkl")
        self.prior_interp = _load_pickle("CNN_logprior_interp.pkl")
        self.network = _SimpleCNN().to(self.device)
        state = _torch_load(HERE / "Phase2_starting_kit_CNN_MCMC_baseline.pth", self.device)
        self.network.load_state_dict(state)
        self.network.eval()

    def _cnn_summary(self, test_data):
        loader = DataLoader(_MaskedMaps(test_data, self.mask), batch_size=self.batch_size,
                            shuffle=False, num_workers=0, pin_memory=self.device.type == "cuda")
        chunks = []
        with torch.inference_mode():
            for images in loader:
                prediction = self.network(images.to(self.device, non_blocking=True)).cpu().numpy()
                chunks.append(prediction * self.label_scale + self.label_mean)
        return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.float64)

    def _log_posterior(self, x, summary):
        result = np.asarray(self.prior_interp(x)).reshape(-1)
        valid = np.isfinite(result)
        if np.any(valid):
            mean = self.mean_interp(x[valid])
            cov = self.cov_interp(x[valid])
            delta = summary[valid] - mean
            sign, logdet = np.linalg.slogdet(cov)
            quadratic = np.einsum("ni,nij,nj->n", delta, np.linalg.inv(cov), delta)
            likelihood = -0.5 * logdet - 0.5 * quadratic
            likelihood[sign <= 0] = -np.inf
            result[valid] += likelihood
        return result

    def _posterior_mean(self, summary):
        rng = np.random.default_rng(31415)
        current = self.cosmology[rng.integers(0, len(self.cosmology), size=len(summary))].copy()
        current_logp = self._log_posterior(current, summary)
        total = np.zeros_like(current)
        kept = 0
        for step in range(self.n_steps):
            proposal = current + rng.normal(scale=0.06, size=current.shape)
            proposal_logp = self._log_posterior(proposal, summary)
            log_ratio = np.minimum(proposal_logp - current_logp, 0.0)
            accept = np.log(rng.random(len(current))) < log_ratio
            current[accept] = proposal[accept]
            current_logp[accept] = proposal_logp[accept]
            if step >= self.burn_in:
                total += current
                kept += 1
        return total / kept

    def predict(self, test_data):
        summary = self._cnn_summary(test_data)
        if len(summary) == 0:
            return np.empty(0, dtype=np.float64)
        parameters = self._posterior_mean(summary)
        mean = self.mean_interp(parameters)
        cov = self.cov_interp(parameters)
        delta = summary - mean
        chi2 = np.einsum("ni,nij,nj->n", delta, np.linalg.inv(cov), delta)
        reference = np.sort(self.reference)
        survival = (len(reference) - np.searchsorted(reference, chi2, side="right")) / len(reference)
        return -survival
