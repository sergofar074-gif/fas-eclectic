"""Vocal enhancement chain (Adobe-Podcast-style, tuned for Suno artefacts)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from . import dsp
from .stft import stft, istft
from .measure import as_2d, integrated_lufs, true_peak_db

EPS = 1e-12


# --------------------------------------------------------------------------- #
#  Spectral de-noise / de-artefact
# --------------------------------------------------------------------------- #
def spectral_denoise(x: np.ndarray, sr: int, strength: float = 1.6,
                     floor: float = 0.10, n_fft: int = 2048) -> np.ndarray:
    """STFT Wiener-ish gate with a per-bin noise profile.

    Removes the 'digital sand' / codec-ish hiss Suno leaves under the voice
    while leaving sustained notes untouched (over-subtraction is floored).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    hop = n_fft // 4
    Z, _, _ = stft(x, n_fft, hop)                 # (frames, bins, 1)
    mag = np.abs(Z)

    if mag.shape[0] > 4:
        noise = np.percentile(mag, 15, axis=0, keepdims=True)
    else:
        noise = np.zeros((1, mag.shape[1], 1), dtype=np.float32)
    noise = median_filter(noise, size=(1, 3, 1), mode="nearest")
    gain = (mag - strength * noise) / (mag + EPS)
    np.clip(gain, floor, 1.0, out=gain)
    gain = median_filter(gain, size=(3, 1, 1), mode="nearest")
    y = istft(Z * gain, hop, n=x.shape[0])
    del Z, mag, gain
    return np.asarray(y, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
#  De-esser / harshness control (dynamic EQ on the two Suno problem bands)
# --------------------------------------------------------------------------- #
def dynamic_band_control(x: np.ndarray, sr: int, lo: float, hi: float,
                         threshold_db: float = -26.0, ratio: float = 4.0,
                         max_reduction_db: float = 7.0) -> np.ndarray:
    """Split a band out, compress only that band, sum back."""
    band = dsp.bandpass(x, lo, hi, sr, stages=2)
    if band.ndim > 1:
        band = band[:, 0]
    rest = x - band
    env = dsp._envelope(band, 1.0, 70.0, sr) + EPS
    env_db = 20 * np.log10(env)
    over = np.maximum(0.0, env_db - threshold_db)
    gr = np.minimum(over * (1 - 1 / ratio), max_reduction_db)
    g = 10 ** (-gr / 20.0)
    return rest + band * g


def de_ess(x: np.ndarray, sr: int, amount: float = 1.0) -> np.ndarray:
    y = dynamic_band_control(x, sr, 3200, 4600, threshold_db=-28.0,
                             ratio=3.5, max_reduction_db=6.0 * amount)
    y = dynamic_band_control(y, sr, 6800, 9200, threshold_db=-30.0,
                             ratio=4.0, max_reduction_db=7.0 * amount)
    return y


# --------------------------------------------------------------------------- #
#  Full vocal chain
# --------------------------------------------------------------------------- #
def enhance(vocal: np.ndarray, sr: int, report: dict | None = None) -> np.ndarray:
    report = report if report is not None else {}
    x = np.asarray(vocal, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x - float(np.mean(x))                      # kill DC
    lufs_in = integrated_lufs(x[:, None], sr)

    # 1. clean-up: rumble out, spectral de-noise (digital sand), gentle gate
    x = dsp.highpass(x, 85.0, sr, stages=2)
    x = spectral_denoise(x, sr, strength=1.7, floor=0.10)
    x = dsp.gate_expander(x, sr, threshold_db=-58.0, ratio=1.8)

    # 2. de-artefact: 3-4 kHz "robot whistle" + 7-9 kHz laser sibilance
    x = dsp.peq(x, 3400, sr, -1.6, q=2.2)          # static surgical dip
    x = dsp.peq(x, 7600, sr, -1.4, q=2.0)
    x = de_ess(x, sr, amount=1.0)

    # 3. presence / body shaping before the compressor
    x = dsp.peq(x, 190, sr, -1.5, q=1.0)            # de-mud
    x = dsp.peq(x, 5200, sr, 1.6, q=1.1)            # intelligibility

    # 4. dynamics: exactly the spec'd 3:1 / 20 ms / 150 ms
    x = dsp.compressor(x, sr, threshold_db=-20.0, ratio=3.0,
                       attack_ms=20.0, release_ms=150.0, knee_db=8.0, makeup="auto")
    # second, slower stage -> density without pumping
    x = dsp.compressor(x, sr, threshold_db=-14.0, ratio=1.8,
                       attack_ms=35.0, release_ms=220.0, knee_db=10.0, makeup=0.0)

    # 5. air / gloss: saturation + high shelf above 12 kHz
    x = dsp.saturate(x, drive_db=3.5, mix=0.22, even=0.12)
    x = dsp.shelf(x, 12000, sr, 3.2, high=True)
    x = dsp.highpass(x, 70.0, sr, stages=1)

    # 6. ride the level into the mix bus
    x = dsp.normalize_peak(x, -3.0)

    report["vocal"] = {
        "lufs_before": round(lufs_in, 2),
        "lufs_after": round(integrated_lufs(x[:, None], sr), 2),
        "true_peak_dbtp": round(true_peak_db(x[:, None], sr), 2),
        "chain": ["HPF 85 Hz", "spectral denoise", "expander",
                  "notch 3.4 kHz / 7.6 kHz", "de-esser 3.2-4.6 / 6.8-9.2 kHz",
                  "compressor 3:1 20/150", "compressor 1.8:1 35/220",
                  "saturation", "high-shelf +3.2 dB @12 kHz"],
    }
    return x
