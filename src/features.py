"""Hand-crafted per-lead ECG features (pure numpy).

Design choice: with only ~150-200 records on CPU, a small 1D-CNN would be
data-starved and slow to train 5x (one model per LOSO fold). Per-lead
morphological/statistical features + a small MLP is fast (<1 min per fold on
CPU), interpretable, and a fair test of whether simple embeddings generalize
across superclasses. Documented in BUILD-NOTES.md.
"""

from __future__ import annotations

import numpy as np

N_LEADS = 12
EXPECTED_LEN = 5000


def _lead_features(x: np.ndarray) -> np.ndarray:
    """20 features from one 1D lead (5000 samples, mV)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mean = x.mean()
    std = x.std()
    xc = x - mean
    feats = [
        mean,
        std,
        x.min(),
        x.max(),
        x.max() - x.min(),                    # peak-to-peak
        np.sqrt(np.mean(x**2)),                # RMS
        np.mean(x**2),                          # energy
        np.median(x),
        np.percentile(x, 5),
        np.percentile(x, 95),
        np.percentile(x, 95) - np.percentile(x, 5),  # robust range
        np.percentile(x, 75) - np.percentile(x, 25), # IQR
        np.mean(np.diff(np.sign(xc)) != 0),     # zero-crossing rate
        float(np.sum(np.abs(np.diff(x)))) / n,  # total variation per sample
    ]
    # Skewness / kurtosis (guard against zero variance)
    if std > 1e-9:
        z = xc / std
        feats += [np.mean(z**3), np.mean(z**4) - 3.0]
    else:
        feats += [0.0, 0.0]
    # Hjorth parameters
    d1 = np.diff(x)
    d2 = np.diff(d1)
    v0, v1, v2 = np.var(x), np.var(d1), np.var(d2)
    mobility = np.sqrt(v1 / v0) if v0 > 1e-12 else 0.0
    complexity = (np.sqrt(v2 / v1) / mobility) if (v1 > 1e-12 and mobility > 1e-12) else 0.0
    feats += [mobility, complexity]
    # Spectral: dominant frequency + spectral centroid (numpy FFT only)
    spec = np.abs(np.fft.rfft(xc)) ** 2
    freqs = np.fft.rfftfreq(n, d=1 / 500.0)
    spec_sum = spec.sum()
    if spec_sum > 1e-12:
        dom = freqs[int(np.argmax(spec[1:])) + 1]
        centroid = float(np.sum(freqs * spec) / spec_sum)
    else:
        dom, centroid = 0.0, 0.0
    feats += [dom, centroid]
    return np.asarray(feats, dtype=np.float64)


def extract_features(sig: np.ndarray) -> np.ndarray:
    """Signal (5000, 12) -> 1D feature vector (N_LEADS * 20 = 240)."""
    sig = np.asarray(sig)
    assert sig.shape == (EXPECTED_LEN, N_LEADS), f"bad shape {sig.shape}"
    return np.concatenate([_lead_features(sig[:, lead]) for lead in range(N_LEADS)])


def feature_matrix(signals: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    """Stack feature vectors for ecg_ids in `order` -> (n, 228)."""
    return np.stack([extract_features(signals[eid]) for eid in order])


def raw_downsampled_features(sig: np.ndarray, target_len: int = 125) -> np.ndarray:
    """Baseline features: each lead downsampled (mean-pool) to target_len.

    (5000, 12) -> (12 * target_len,) raw amplitude vector, no feature
    engineering. Used as the 'trivial' baseline for nearest-centroid.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.shape[0]
    k = n // target_len
    trimmed = sig[: k * target_len]
    pooled = trimmed.reshape(target_len, k, 12).mean(axis=1)
    return pooled.reshape(-1)
