"""Loudness / peak metering (ITU-R BS.1770-4 + true peak).

All metering is streamed in chunks so that multi-minute 24-bit masters never
need more than a few tens of MB of RAM.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12
CHUNK = 1 << 18  # 262 144 samples


def as_2d(x: np.ndarray) -> np.ndarray:
    """(samples,) / (samples, ch) -> float64 (samples, ch) view or copy as needed."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.dtype != np.float64:
        x = x.astype(np.float64, copy=False)
    return x


def to_mono(x: np.ndarray) -> np.ndarray:
    return as_2d(x).mean(axis=1)


# --------------------------------------------------------------------------- #
#  BS.1770 K-weighting (analytic coefficients, any sample rate)
# --------------------------------------------------------------------------- #
def k_weight_coeffs(sr: int):
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    A = 10 ** (G / 40.0)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    cw, sw = np.cos(w0), np.sin(w0)
    b1 = np.array([A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
                   -2 * A * ((A - 1) + (A + 1) * cw),
                   A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)])
    a1 = np.array([(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
                   2 * ((A - 1) - (A + 1) * cw),
                   (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha])
    b1, a1 = b1 / a1[0], a1 / a1[0]

    fc2, Q2 = 38.13547087602444, 0.5003270373238773
    w0 = 2 * np.pi * fc2 / sr
    alpha = np.sin(w0) / (2 * Q2)
    cw = np.cos(w0)
    b2 = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a2 = np.array([1 + alpha, -2 * cw, 1 - alpha])
    b2, a2 = b2 / a2[0], a2 / a2[0]
    return (b1, a1), (b2, a2)


# --------------------------------------------------------------------------- #
#  Integrated loudness
# --------------------------------------------------------------------------- #
def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness (LUFS), BS.1770-4, streamed."""
    n_ch = 1 if np.asarray(x).ndim == 1 else np.asarray(x).shape[1]
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    total = x.shape[0]
    if total < int(0.4 * sr):
        return -70.0

    (b1, a1), (b2, a2) = k_weight_coeffs(sr)
    block = int(0.400 * sr)
    step = max(1, block // 4)
    chunk = step * 60          # keep the 0.1 s block grid aligned across chunks

    ms_blocks: list[float] = []
    z1 = np.zeros((max(len(a1), len(a2)) - 1, n_ch))
    z2 = np.zeros((max(len(a1), len(a2)) - 1, n_ch))
    keep = np.zeros((block, n_ch))

    pos = 0
    while pos < total:
        seg = x[pos:pos + chunk].astype(np.float64, copy=False)
        y1, z1 = signal.lfilter(b1, a1, seg, axis=0, zi=z1)
        y2, z2 = signal.lfilter(b2, a2, y1, axis=0, zi=z2)
        buf = np.concatenate([keep, y2], axis=0)
        m = y2.shape[0]
        starts = range(0, m, step)
        for s in starts:
            seg_b = buf[s:s + block]
            if seg_b.shape[0] < block:
                break
            ms_blocks.append(float(np.sum(np.mean(seg_b ** 2, axis=0))))
        keep = buf[-block:]
        pos += chunk

    ms = np.asarray(ms_blocks, dtype=np.float64)
    if ms.size == 0:
        return -70.0
    l = -0.691 + 10 * np.log10(ms + EPS)
    abs_gate = l > -70.0
    if not np.any(abs_gate):
        return -70.0
    rel = -0.691 + 10 * np.log10(np.mean(ms[abs_gate]) + EPS)
    keep_mask = abs_gate & (l > (rel - 10.0))
    if not np.any(keep_mask):
        return float(rel)
    return float(-0.691 + 10 * np.log10(np.mean(ms[keep_mask]) + EPS))


# --------------------------------------------------------------------------- #
#  Peaks
# --------------------------------------------------------------------------- #
def sample_peak_db(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return -120.0
    if x.ndim == 1:
        p = float(np.max(np.abs(x)))
    else:
        p = max(float(np.max(np.abs(x[:, ch]))) for ch in range(x.shape[1]))
    return 20 * np.log10(p + EPS)


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """True peak in dBTP: 4x band-limited oversampling, chunked."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    peak = 0.0
    block = 1 << 18
    pad = 64
    for ch in range(x.shape[1]):
        i = 0
        while i < x.shape[0]:
            seg = x[max(0, i - pad): i + block + pad, ch]
            up = signal.resample_poly(seg.astype(np.float64, copy=False), oversample, 1)
            if up.size:
                peak = max(peak, float(np.max(np.abs(up))))
            i += block
    return 20 * np.log10(peak + EPS)


def peak_envelope(x: np.ndarray, points: int = 420) -> list[float]:
    """Normalised peak envelope for the UI waveform."""
    x = np.asarray(x)
    mono = np.max(np.abs(x), axis=1) if x.ndim > 1 else np.abs(x)
    n = mono.shape[0]
    if n == 0:
        return [0.0] * points
    idx = np.linspace(0, n, points + 1).astype(int)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = mono[a:max(b, a + 1)]
        out.append(float(np.max(seg)) if seg.size else 0.0)
    m = max(out) or 1.0
    return [round(v / m, 4) for v in out]
