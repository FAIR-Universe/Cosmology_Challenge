"""Ingestion-ready Phase 2 weak-lensing power-spectrum baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
IMAGE_SHAPE = (1424, 176)
N_BINS = 10
ELL_EDGES = np.logspace(2, 4, N_BINS + 1)
PIXEL_SIZE_RADIANS = 2.0 / 60.0 / 180.0 * np.pi


def _power_spectrum(image, pixel_size, edges):
    transformed = np.fft.rfft2(image)
    squared = (transformed * transformed.conj()).real
    height, width = image.shape
    frequency = np.fft.fftfreq(height, d=pixel_size)[:, None] ** 2
    frequency = frequency + np.fft.rfftfreq(width, d=pixel_size)[None, :] ** 2
    wave_number = np.sqrt(frequency) * 2.0 * np.pi
    index = np.searchsorted(edges, wave_number)
    power = np.bincount(index.ravel(), weights=squared.ravel())
    modes = np.bincount(index.ravel())
    if width % 2 == 0:
        mirrored_index, mirrored_power = index[..., 1:-1], squared[..., 1:-1]
    else:
        mirrored_index, mirrored_power = index[..., 1:], squared[..., 1:]
    power += np.bincount(mirrored_index.ravel(), weights=mirrored_power.ravel(), minlength=len(power))
    modes += np.bincount(mirrored_index.ravel(), minlength=len(modes))
    power = power[1:len(edges)]
    modes = modes[1:len(edges)]
    nonempty = modes > 0
    power[nonempty] /= modes[nonempty]
    power *= pixel_size ** 2 / height / width
    return power


class Model:
    """Fit cosmology to each map's power spectrum and return fit OoD scores."""

    def __init__(self):
        self.mask = np.load(HERE / "WIDE12H_bin2_2arcmin_mask.npy", allow_pickle=False)
        labels = np.load(HERE / "label_newrealization.npy", mmap_mode="r", allow_pickle=False)
        cosmology = np.asarray(labels[:, 0, :2])
        mean = np.load(HERE / "meanlogPS.npy", allow_pickle=False)
        covariance = np.load(HERE / "covlogPS.npy", allow_pickle=False)
        self.mean_interp = LinearNDInterpolator(cosmology, mean, fill_value=np.nan)
        self.cov_interp = LinearNDInterpolator(cosmology, covariance, fill_value=np.nan)
        self.prior_interp = LinearNDInterpolator(cosmology, np.zeros(len(cosmology)), fill_value=-np.inf)
        self.reference = np.load(HERE / "PSA_TS_InD.npy", allow_pickle=False)

    def _image(self, sample):
        sample = np.asarray(sample, dtype=np.float64)
        if sample.shape == IMAGE_SHAPE:
            return sample
        image = np.zeros(IMAGE_SHAPE, dtype=np.float64)
        image[self.mask] = sample
        return image

    def _log_posterior(self, parameters, summary):
        prior = np.asarray(self.prior_interp(parameters)).item()
        if not np.isfinite(prior):
            return -np.inf
        mean = np.asarray(self.mean_interp(parameters)).reshape(N_BINS)
        covariance = np.asarray(self.cov_interp(parameters)).reshape(N_BINS, N_BINS)
        delta = summary - mean
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        return prior - 0.5 * logdet - 0.5 * float(delta @ np.linalg.solve(covariance, delta))

    def predict(self, test_data):
        chi2_values = np.empty(len(test_data), dtype=np.float64)
        for index in range(len(test_data)):
            spectrum = _power_spectrum(self._image(test_data[index]), PIXEL_SIZE_RADIANS, ELL_EDGES)
            summary = np.log10(spectrum)
            fit = minimize(lambda x: -self._log_posterior(x, summary),
                           x0=np.array([0.3, 0.8]), method="Nelder-Mead")
            parameters = fit.x
            mean = np.asarray(self.mean_interp(parameters)).reshape(N_BINS)
            covariance = np.asarray(self.cov_interp(parameters)).reshape(N_BINS, N_BINS)
            delta = summary - mean
            chi2_values[index] = float(delta @ np.linalg.solve(covariance, delta))
        reference = np.sort(self.reference)
        survival = (len(reference) - np.searchsorted(reference, chi2_values, side="right")) / len(reference)
        return -survival
