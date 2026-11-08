"""Core DSP building blocks: biquad EQ, dynamics, saturation, limiter."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

EPS = 1e-12


# --------------------------------------------------------------------------- #
#  Biquad (RBJ cookbook) filters
# --------------------------------------------------------------------------- #
def _rbj(kind: str, f0: float, sr: int, gain_db: float = 0.0, q: float = 0.707):
    f0 = float(np.clip(f0, 10.0, sr * 0.49))
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40.0)

    if kind == "peak":
        b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = np.array([A * ((A + 1) + (A - 1) * cw + sq),
                      -2 * A * ((A - 1) + (A + 1) * cw),
                      A * ((A + 1) + (A - 1) * cw - sq)])
        a = np.array([(A + 1) - (A - 1) * cw + sq,
                      2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - sq])
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = np.array([A * ((A + 1) - (A - 1) * cw + sq),
                      2 * A * ((A - 1) - (A + 1) * cw),
                      A * ((A + 1) - (A - 1) * cw - sq)])
        a = np.array([(A + 1) + (A - 1) * cw + sq,
                      -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - sq])
    elif kind == "highpass":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    elif kind == "lowpass":
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    else:
        raise ValueError(kind)
    b = b / a[0]
    a = a / a[0]
    return b, a


def apply_biquad(x: np.ndarray, kind: str, f0: float, sr: int,
                 gain_db: float = 0.0, q: float = 0.707) -> np.ndarray:
    b, a = _rbj(kind, f0, sr, gain_db, q)
    # float32 fast path halves the RAM; very low cut-offs keep float64
    # precision because their poles sit right next to DC.
    if f0 < 100.0:
        y = signal.lfilter(b, a, np.asarray(x, dtype=np.float64), axis=0)
        return y.astype(np.float32, copy=False)
    return signal.lfilter(b.astype(np.float32), a.astype(np.float32),
                          np.asarray(x, dtype=np.float32), axis=0)


def highpass(x, f0, sr, stages: int = 2) -> np.ndarray:
    """Cascaded Butterworth HPF (each stage = 12 dB/oct)."""
    y = np.asarray(x, dtype=np.float32)
    for _ in range(max(1, stages)):
        y = apply_biquad(y, "highpass", f0, sr, q=0.707)
    return y


def lowpass(x, f0, sr, stages: int = 2) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    for _ in range(max(1, stages)):
        y = apply_biquad(y, "lowpass", f0, sr, q=0.707)
    return y


def peq(x, f0, sr, gain_db, q=1.0) -> np.ndarray:
    return apply_biquad(x, "peak", f0, sr, gain_db, q)


def shelf(x, f0, sr, gain_db, high=True, q=0.707) -> np.ndarray:
    return apply_biquad(x, "highshelf" if high else "lowshelf", f0, sr, gain_db, q)


def bandpass(x, lo, hi, sr, stages: int = 2) -> np.ndarray:
    """Band-split via cascaded Butterworth HPF + LPF."""
    return lowpass(highpass(x, lo, sr, stages), hi, sr, stages)


# --------------------------------------------------------------------------- #
#  Utility
# --------------------------------------------------------------------------- #
def db2lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin2db(v: float) -> float:
    return 20.0 * np.log10(max(float(v), EPS))


def normalize_peak(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= EPS:
        return x
    return x * (db2lin(target_db) / peak)


# --------------------------------------------------------------------------- #
#  Dynamics: compressor with soft knee, attack / release, auto-makeup
# --------------------------------------------------------------------------- #
def _envelope(x: np.ndarray, attack_ms: float, release_ms: float, sr: int) -> np.ndarray:
    """Peak envelope follower (vectorised, sample-accurate attack/release)."""
    a_att = np.exp(-1.0 / (max(attack_ms, 0.01) * 0.001 * sr))
    a_rel = np.exp(-1.0 / (max(release_ms, 0.01) * 0.001 * sr))
    xa = np.asarray(x, dtype=np.float32)
    mag = np.max(np.abs(xa), axis=1) if xa.ndim > 1 else np.abs(xa)
    # Two passes: release-only smoothing gives a smooth floor, then a
    # downward-only attack clamp guarantees the classic behaviour.
    rel = signal.lfilter(np.float32([1 - a_rel]), np.float32([1, -a_rel]), mag)
    att = signal.lfilter(np.float32([1 - a_att]), np.float32([1, -a_att]), mag)
    return np.maximum(rel, att)


def compressor(x: np.ndarray, sr: int, threshold_db: float = -18.0,
               ratio: float = 3.0, attack_ms: float = 20.0,
               release_ms: float = 150.0, knee_db: float = 8.0,
               makeup: str | float = "auto") -> np.ndarray:
    """Feed-forward compressor (stereo-linked), soft knee, optional auto makeup."""
    x = np.asarray(x, dtype=np.float32)
    env = _envelope(x, attack_ms, release_ms, sr) + EPS
    env_db = (20 * np.log10(env)).astype(np.float32)
    over = (env_db - threshold_db).astype(np.float32)
    slope = np.float32(1.0 - 1.0 / ratio)
    gr = np.zeros_like(over)
    if knee_db > 0:
        # soft knee interpolation between 1:1 and ratio:1 (float32, in-place)
        k = np.float32(knee_db / 2.0)
        hard = over >= k
        gr[hard] = over[hard] * slope
        mid = (~hard) & (over > -k)
        gr[mid] = slope * (over[mid] + k) ** 2 / np.float32(2.0 * knee_db)
    else:
        np.maximum(over, 0.0, out=gr)
        gr *= slope
    g = (10 ** (-gr / 20.0)).astype(np.float32)
    y = x * g[:, None] if x.ndim > 1 else x * g

    if makeup == "auto":
        mk = float(np.mean(gr)) * 0.6
    else:
        mk = float(makeup)
    return y * db2lin(mk)


def gate_expander(x: np.ndarray, sr: int, threshold_db: float = -50.0,
                  ratio: float = 2.0, attack_ms: float = 2.0,
                  release_ms: float = 120.0) -> np.ndarray:
    """Gentle downward expander used as a residual-noise gate."""
    x = np.asarray(x, dtype=np.float32)
    env = _envelope(x, attack_ms, release_ms, sr) + EPS
    env_db = (20 * np.log10(env)).astype(np.float32)
    under = (threshold_db - env_db).astype(np.float32)
    gr = np.maximum(under, 0.0)
    gr *= np.float32(1.0 - 1.0 / ratio)
    g = (10 ** (-gr / 20.0)).astype(np.float32)
    return x * g[:, None] if x.ndim > 1 else x * g


# --------------------------------------------------------------------------- #
#  Saturation
# --------------------------------------------------------------------------- #
def saturate(x: np.ndarray, drive_db: float = 3.0, mix: float = 0.25,
             even: float = 0.15) -> np.ndarray:
    """Soft tanh saturation with a touch of 2nd harmonic; parallel mixed."""
    x = np.asarray(x, dtype=np.float32)
    d = db2lin(drive_db)
    w = np.tanh(x * d)
    if even > 0:
        w = w + even * (np.tanh(x * d) ** 2 - np.mean(np.tanh(x * d) ** 2))
    w = w / (db2lin(drive_db) ** 0.5 + EPS)
    return (1 - mix) * x + mix * w


# --------------------------------------------------------------------------- #
#  Brickwall limiter with look-ahead, driven by a true-peak envelope
# --------------------------------------------------------------------------- #
def tp_envelope(x: np.ndarray, sr: int, oversample: int = 4) -> np.ndarray:
    """Sample peak envelope merged with inter-sample (4x) peaks, chunked."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    env = np.max(np.abs(x), axis=1).copy()
    block, pad = 1 << 18, 64
    for ch in range(x.shape[1]):
        i = 0
        while i < n:
            a, b = max(0, i - pad), min(n, i + block + pad)
            seg = x[a:b, ch].astype(np.float64)
            up = signal.resample_poly(seg, oversample, 1)
            m = b - a
            up = up[:m * oversample].reshape(m, oversample).max(axis=1)
            np.maximum(env[a:b], up.astype(np.float32), out=env[a:b])
            i += block
    return env


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0,
            lookahead_ms: float = 8.0, attack_ms: float = 3.0,
            release_ms: float = 90.0, true_peak: bool = True,
            mag: np.ndarray | None = None,
            pre_gain: float = 1.0) -> np.ndarray:
    """Look-ahead brickwall limiter with a smoothed gain envelope.

    `mag` may carry a pre-computed peak envelope of the un-scaled input (so the
    4x oversampled envelope is only ever computed once per master).
    """
    x = np.asarray(x, dtype=np.float32)
    ceil = np.float32(db2lin(ceiling_db))
    la = max(1, int(sr * lookahead_ms / 1000.0))

    if mag is None:
        mag = tp_envelope(x, sr) if true_peak else (
            np.max(np.abs(x), axis=1) if x.ndim > 1 else np.abs(x))
    env = maximum_filter1d(mag * np.float32(pre_gain), size=2 * la + 1, mode="nearest")
    need = np.minimum(np.float32(1.0), ceil / (env + np.float32(EPS)))

    a_att = np.float32(np.exp(-1.0 / (max(attack_ms, 0.05) * 0.001 * sr)))
    a_rel = np.float32(np.exp(-1.0 / (max(release_ms, 0.1) * 0.001 * sr)))
    att = signal.lfilter(np.float32([1 - a_att]), np.float32([1, -a_att]), need)
    rel = signal.lfilter(np.float32([1 - a_rel]), np.float32([1, -a_rel]), need)
    g = np.minimum(np.minimum(att, rel), need)

    y = x * np.float32(pre_gain)
    return y * (g[:, None] if x.ndim > 1 else g)
