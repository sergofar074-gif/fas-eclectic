"""Voice Activity Detection — the gate that protects against "bubbling".

What this module guarantees
---------------------------
Separating a **pure instrumental** is not merely useless, it is harmful: the
vocal mask latches onto a centred lead synth / melody, that artificial "vocal"
receives its own compressor + de-esser + saturation + level lift, and is summed
back into the mix.  The audible result is a pumping / bubbling artefact.
So the pipeline asks this module *before* committing to a split.

Two lines of defence
--------------------
1. **Probe.** A short excerpt of the track (the loudest 25 s, plus the middle)
   is run through the cheap, phase-coherent DSP separator.  The extracted vocal
   stem is scored by a real, trained VAD.
2. **Final check.** If the probe says "voice", the *full* vocal stem is scored
   again after the real separation.  If it does not look like a voice, the
   stems are thrown away and the untouched mix is mastered instead.

Why score the stem and not the mix
----------------------------------
Measured on the reference battery (``samples/vad``): on the raw mix, a voice
buried 18 dB under the bed scores 0.003 — indistinguishable from an
instrumental.  On the extracted stem the same case scores 0.645 while *every*
instrumental case stays below 0.11.  The separator, not the detector, is what
makes VAD reliable inside music.

Backends
--------
* ``silero``   — Silero VAD v5 (ONNX, 2.2 MB, bundled in ``pipeline/models``).
                 Trained on real speech; runs on CPU at ~0.2 ms per 32 ms frame.
* ``heuristic``— built-in fallback (modulation / pauses / harmonicity / pitch).
                 Used only if onnxruntime or the model file is missing.
"""
from __future__ import annotations

import os
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from .stft import stft

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "silero_vad.onnx")

# Calibrated on samples/vad (positives 0.65-0.90, negatives <= 0.11)
THRESHOLD_VOCAL = 0.30
THRESHOLD_INSTRUMENTAL = 0.15
MIN_HIT_RATIO = 0.10

EPS = 1e-12
_SESSION = None
_SESSION_TRIED = False


# --------------------------------------------------------------------------- #
#  Silero VAD v5 (ONNX)
# --------------------------------------------------------------------------- #
def _session():
    global _SESSION, _SESSION_TRIED
    if _SESSION_TRIED:
        return _SESSION
    _SESSION_TRIED = True
    try:
        import onnxruntime as ort
        if not os.path.exists(MODEL_PATH):
            _SESSION = None
            return None
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _SESSION = ort.InferenceSession(MODEL_PATH,
                                        providers=["CPUExecutionProvider"],
                                        sess_options=opts)
    except Exception:  # noqa: BLE001
        _SESSION = None
    return _SESSION


def backend_available() -> str:
    return "silero" if _session() is not None else "heuristic"


def _to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        x = signal.resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > EPS:
        x = x / peak * 0.9                     # level-independent
    return x


def _silero_probabilities(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame voice probability (32 ms hops)."""
    sess = _session()
    if sess is None:
        return np.array([])
    audio = _to_16k(x, sr)
    n, ctx_size = 512, 64
    if audio.size < n:
        return np.array([])
    ctx = np.zeros(ctx_size, np.float32)
    state = np.zeros((2, 1, 128), np.float32)
    sr_i = np.array(16000, np.int64)
    out = []
    for i in range(0, audio.size - n + 1, n):
        inp = np.concatenate([ctx, audio[i:i + n]])[None, :].astype(np.float32)
        p, state = sess.run(None, {"input": inp, "state": state, "sr": sr_i})
        ctx = inp[0, -ctx_size:]
        out.append(float(p[0, 0]))
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Heuristic fallback (no onnxruntime / model missing)
# --------------------------------------------------------------------------- #
def _heuristic(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    n = x.shape[0]
    mid = 0.5 * (x[:, 0] + x[:, 1])

    Z, _, hop = stft(mid, 2048, 512)
    nfr = max(1, int(np.ceil(n / hop)))
    mag = np.abs(Z)[:nfr, :, 0]
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    bi = (freqs >= 180) & (freqs <= 4200)
    band = mag[:, bi]
    del Z, mag

    energy = (band ** 2).mean(axis=1) + EPS
    e_db = 10 * np.log10(energy)
    act = e_db > (np.percentile(e_db, 95) - 30.0)

    harmonic = median_filter(band, size=(17, 1), mode="nearest")
    perc = median_filter(band, size=(1, 25), mode="nearest")
    h_ratio = np.sum(harmonic ** 2, axis=1) / (np.sum(band ** 2, axis=1) + EPS)
    flat = np.exp(np.mean(np.log(band + 1e-9), axis=1)) / (np.mean(band, axis=1) + 1e-9)

    env = np.sqrt((harmonic ** 2).mean(axis=1) + EPS)
    if env.size > 43:
        env = env - np.convolve(env, np.ones(43) / 43, mode="same")
    e_h = 10 * np.log10((harmonic ** 2).mean(axis=1) + EPS)

    return {
        "energy": energy, "e_db": e_db, "active": act,
        "harmonic_ratio": float(np.median(h_ratio[act])) if np.any(act) else 0.0,
        "flatness": float(np.median(flat[act])) if np.any(act) else 1.0,
        "gap_fraction": float(np.mean(e_h < np.percentile(e_h, 95) - 20.0)),
        "dyn_range": float(np.percentile(e_h, 95) - np.percentile(e_h, 50)),
        "mod": float(_mod_ratio(env[act] if np.any(act) else env, sr, hop)),
    }


def _mod_ratio(env: np.ndarray, sr: int, hop: int) -> float:
    if env.size < 64:
        return 0.0
    fr = sr / hop
    e = env.astype(np.float64)
    e = e - e.mean()
    if np.allclose(e, 0):
        return 0.0
    nper = min(512, max(64, int(fr * 4)))
    f, p = signal.welch(e, fs=fr, nperseg=nper, noverlap=nper // 2)
    m1 = (f >= 0.5) & (f <= 20.0)
    m2 = (f >= 2.5) & (f <= 8.0)
    tot = np.trapezoid(p[m1], f[m1])
    voi = np.trapezoid(p[m2], f[m2])
    return float(np.clip(voi / (tot + EPS), 0.0, 1.0))


def _ramp(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.0


def _heuristic_score(x: np.ndarray, sr: int) -> dict:
    f = _heuristic(x, sr)
    s_mod = _ramp(f["mod"], 0.18, 0.48)
    s_gap = _ramp(f["gap_fraction"], 0.03, 0.25)
    s_dyn = _ramp(f["dyn_range"], 7.0, 20.0)
    s_harm = _ramp(f["harmonic_ratio"], 0.30, 0.75)
    s_flat = 1.0 - _ramp(f["flatness"], 0.25, 0.70)
    core = 0.40 * s_mod + 0.22 * s_gap + 0.13 * s_dyn
    qualify = 0.5 * (s_harm + s_flat)
    return {"score": float(np.clip((core * (0.55 + 0.45 * qualify)) / 0.75, 0, 1)),
            "features": {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in f.items() if k not in ("energy", "e_db", "active")}}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def score_signal(x: np.ndarray, sr: int) -> dict:
    """Voice-likeness of a signal in [0, 1] + diagnostics."""
    backend = backend_available()
    if backend == "silero":
        p = _silero_probabilities(x, sr)
        if p.size == 0:
            return {"score": 0.0, "hit_ratio": 0.0, "backend": "silero",
                    "frames": 0, "note": "signal too short"}
        mean = float(np.mean(p))
        hit = float(np.mean(p > 0.5))
        return {
            "score": round(mean, 3),
            "hit_ratio": round(hit, 3),
            "peak": round(float(np.max(p)), 3),
            "backend": "silero",
            "frames": int(p.size),
        }
    h = _heuristic_score(x, sr)
    return {"score": round(h["score"], 3), "hit_ratio": round(h["score"], 3),
            "peak": round(h["score"], 3), "backend": "heuristic",
            "features": h["features"]}


def decide(scores: list[dict]) -> dict:
    """Combine per-window scores into a routing decision."""
    if not scores:
        return {"mode": "unknown", "score": 0.0, "verdict": "no data"}
    best = max(scores, key=lambda s: s["score"])
    score = float(best["score"])
    hit = float(best.get("hit_ratio", 0.0))

    if score >= THRESHOLD_VOCAL and hit >= MIN_HIT_RATIO:
        verdict = "vocal"
    elif score < THRESHOLD_INSTRUMENTAL:
        verdict = "instrumental"
    else:
        verdict = "uncertain"
    return {
        "verdict": verdict,
        "score": round(score, 3),
        "hit_ratio": round(hit, 3),
        "backend": best.get("backend"),
        "thresholds": {"vocal": THRESHOLD_VOCAL,
                       "instrumental": THRESHOLD_INSTRUMENTAL,
                       "min_hit_ratio": MIN_HIT_RATIO},
    }


def probe_windows(x: np.ndarray, sr: int, seconds: float = 25.0,
                  max_windows: int = 2) -> list[tuple[int, int]]:
    """Pick the excerpts to probe: the loudest window and the middle one."""
    n = x.shape[0]
    w = int(seconds * sr)
    if n <= w * 1.4:
        return [(0, n)]

    mono = np.max(np.abs(x), axis=1) if x.ndim > 1 else np.abs(x)
    hop = max(1, w // 4)
    energies = [(float(np.sum(mono[i:i + w] ** 2)), i)
                for i in range(0, n - w + 1, hop)]
    energies.sort(reverse=True)
    picks = [energies[0][1], max(0, (n - w) // 2)]
    out = []
    for p in picks[:max_windows]:
        if all(abs(p - q) > w // 2 for q, _ in out):
            out.append((p, p + w))
    return out or [(0, min(w, n))]
