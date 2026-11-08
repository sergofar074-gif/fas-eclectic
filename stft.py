"""Memory-bounded STFT / iSTFT in float32 (complex64 spectra).

Frames are transformed in chunks, so a 10-minute 24-bit track never needs more
than a few tens of megabytes of scratch memory.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

EPS = 1e-12


def _window(n_fft: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def stft(x: np.ndarray, n_fft: int = 4096, hop: int | None = None,
         chunk_frames: int = 256) -> tuple[np.ndarray, int, int]:
    """Return (spectrum[frames, freq, ch] complex64, n_fft, hop)."""
    hop = hop or n_fft // 4
    x = np.asarray(x, dtype=np.float32)
    mono = x.ndim == 1
    if mono:
        x = x[:, None]
    n, ch = x.shape

    pad = n_fft
    total = n + 2 * pad
    n_pad = total + (-total) % hop
    xp = np.zeros((n_pad, ch), dtype=np.float32)
    xp[pad:pad + n, :] = x

    n_frames = 1 + (n_pad - n_fft) // hop
    win = _window(n_fft)
    out = np.empty((n_frames, n_fft // 2 + 1, ch), dtype=np.complex64)

    base = np.arange(n_fft)[None, :]
    for i in range(0, n_frames, chunk_frames):
        j = min(n_frames, i + chunk_frames)
        idx = base + hop * np.arange(i, j)[:, None]
        fr = xp[idx] * win[None, :, None]
        out[i:j] = sfft.rfft(fr, axis=1)
    return out, n_fft, hop


def istft(Z: np.ndarray, hop: int, n: int | None = None,
          chunk_frames: int = 256) -> np.ndarray:
    """Overlap-add inverse of :func:`stft` (weighted, COLA-safe)."""
    Z = np.asarray(Z)
    n_frames, n_bins, ch = Z.shape
    n_fft = 2 * (n_bins - 1)
    win = _window(n_fft)

    n_out = (n_frames - 1) * hop + n_fft
    y = np.zeros((n_out, ch), dtype=np.float32)
    wsum = np.zeros(n_out, dtype=np.float32)
    w2 = (win ** 2).astype(np.float32)

    for i in range(0, n_frames, chunk_frames):
        j = min(n_frames, i + chunk_frames)
        fr = sfft.irfft(Z[i:j], n=n_fft, axis=1).astype(np.float32)
        for k in range(j - i):
            s = (i + k) * hop
            y[s:s + n_fft] += fr[k] * win[:, None]
            wsum[s:s + n_fft] += w2

    wsum[wsum < 1e-8] = 1e-8
    y /= wsum[:, None]

    pad = n_fft
    y = y[pad:pad + (n if n is not None else n_out - 2 * pad)]
    return y[:, 0] if ch == 1 else y
