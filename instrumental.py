"""Instrumental mastering chain (LANDR / BandLab-style).

Band splitting is done with complementary tanh crossfades in the FFT domain
(sum of weights == 1, zero phase), so the split itself is perfectly transparent
and costs one spectrum at a time instead of four time-domain copies.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from . import dsp
from .measure import integrated_lufs, true_peak_db

EPS = 1e-12


def _nfft(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1024))))


def band_weights(freqs: np.ndarray, edges: list[float],
                 softness: float = 0.18) -> list[np.ndarray]:
    """Complementary band weights (they sum to exactly 1)."""
    bounds = [0.0] + list(edges) + [freqs[-1] * 2.0]
    weights = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        w = np.ones_like(freqs, dtype=np.float32)
        if i > 0:
            w *= 0.5 * (1.0 + np.tanh((freqs - lo) / max(lo, 1.0) * softness * 5.0))
        if i < len(bounds) - 2:
            w *= 0.5 * (1.0 + np.tanh((hi - freqs) / max(hi, 1.0) * softness * 5.0))
        weights.append(w)
    norm = np.sum(weights, axis=0)
    norm[norm < 1e-9] = 1.0
    return [w / norm for w in weights]


# --------------------------------------------------------------------------- #
#  Multiband glue compression
# --------------------------------------------------------------------------- #
def multiband_compress(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    n = x.shape[0]
    nfft = _nfft(n)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr).astype(np.float32)

    settings = [
        (-22.0, 2.4, 30.0, 220.0),   # sub / kick
        (-20.0, 1.9, 18.0, 180.0),   # bass / body
        (-22.0, 1.7, 12.0, 140.0),   # mid
        (-26.0, 2.2, 6.0, 120.0),    # air / cymbals
    ]
    out = np.zeros_like(x)
    # one channel at a time: keeps the FFT scratch memory at ~1/4
    for w, (thr, ratio, att, rel) in zip(band_weights(freqs, [120.0, 800.0, 5000.0]),
                                         settings):
        band = np.empty_like(x)
        for ch in range(x.shape[1]):
            spec = sfft.rfft(x[:, ch], nfft)
            band[:, ch] = sfft.irfft(spec * w, nfft)[:n]
            del spec
        out += dsp.compressor(band, sr, threshold_db=thr, ratio=ratio,
                              attack_ms=att, release_ms=rel, knee_db=9.0,
                              makeup=0.0)
        del band
    return out


# --------------------------------------------------------------------------- #
#  Stereo widening + mono bass
# --------------------------------------------------------------------------- #
def stereo_widen(x: np.ndarray, sr: int, mono_below: float = 300.0,
                 wide_range: tuple[float, float] = (300.0, 8000.0),
                 width: float = 1.38, air_width: float = 1.18):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)

    M = 0.5 * (x[:, 0] + x[:, 1])
    S = 0.5 * (x[:, 0] - x[:, 1])

    n = x.shape[0]
    nfft = _nfft(n)
    spec = sfft.rfft(S, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr).astype(np.float32)
    w_low, w_mid, w_high = band_weights(freqs, [mono_below, wide_range[1]])
    total_w = (0.0 * w_low + width * w_mid + air_width * w_high).astype(np.float32)
    S2 = sfft.irfft(spec * total_w, nfft)[:n].astype(np.float32)
    del spec

    out = np.stack([M + S2, M - S2], axis=1)

    corr = float(np.corrcoef(out[:, 0], out[:, 1])[0, 1]) if out.shape[0] > 2 else 1.0
    if corr < 0.12:                       # mono-compatibility guard
        out = 0.5 * out + 0.5 * x
        corr = float(np.corrcoef(out[:, 0], out[:, 1])[0, 1])

    info = {
        "mono_below_hz": mono_below,
        "widened_range_hz": list(wide_range),
        "width_factor": width,
        "air_width_factor": air_width,
        "stereo_correlation": round(corr, 3),
    }
    return out, info


# --------------------------------------------------------------------------- #
#  Full instrumental chain
# --------------------------------------------------------------------------- #
def master(instrumental: np.ndarray, sr: int, report: dict | None = None) -> np.ndarray:
    report = report if report is not None else {}
    x = np.asarray(instrumental, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    lufs_in = integrated_lufs(x, sr)

    # 1. rumble: hard 25 Hz high-pass (24 dB/oct) + a little sub tidying
    x = dsp.highpass(x, 25.0, sr, stages=2)
    x = dsp.highpass(x, 28.0, sr, stages=1)

    # 2. multiband glue compression
    x = multiband_compress(x, sr)

    # 3. stereo image: bass & kick strictly mono, 300 Hz-8 kHz spread out
    x, stereo_info = stereo_widen(x, sr)

    # 4. tonal balance (LANDR-ish "Universal" curve)
    x = dsp.shelf(x, 90.0, sr, 0.8, high=False)
    x = dsp.peq(x, 320.0, sr, -1.0, q=1.0)
    x = dsp.peq(x, 2600.0, sr, -0.6, q=1.4)
    x = dsp.shelf(x, 10000.0, sr, 1.4, high=True)

    # 5. final glue + peak trim
    x = dsp.compressor(x, sr, threshold_db=-12.0, ratio=1.5,
                       attack_ms=30.0, release_ms=250.0, knee_db=10.0, makeup=0.0)
    x = dsp.normalize_peak(x, -3.0)

    report["instrumental"] = {
        "lufs_before": round(lufs_in, 2),
        "lufs_after": round(integrated_lufs(x, sr), 2),
        "true_peak_dbtp": round(true_peak_db(x, sr), 2),
        "stereo": stereo_info,
        "chain": ["HPF 25 Hz (24 dB/oct)", "4-band glue compression",
                  "M/S widening 300 Hz-8 kHz, mono bass", "tonal balance EQ",
                  "bus glue 1.5:1"],
    }
    return x
