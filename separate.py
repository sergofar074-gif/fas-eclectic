"""Stem separation.

Two backends:
  * ``dsp``    – built-in, always available. Centre / harmonic / modulation
                 spectral masking in the mid/side domain with phase-coherent
                 reconstruction:  M_instr = M * (1 - mask),  S_instr = S, so the
                 two stems add back to the input without comb filtering.
                 Processed in overlapping blocks -> bounded RAM on long tracks.
  * ``demucs`` – optional neural backend (htdemucs / demucs v4). Activated with
                 FAS_SEPARATOR=demucs when torch+demucs are installed; falls back
                 to ``dsp`` automatically when unavailable.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .stft import stft, istft

EPS = 1e-12
N_FFT = 4096
BLOCK_SEC = 20.0
OVERLAP_SEC = 2.0


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def to_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        x = x[:, :2]
    return x


def _band_weight(freqs: np.ndarray) -> np.ndarray:
    """Where a lead vocal can physically live (tuned on reference material).

    Double sigmoid -> ~ -40 dB at 110 Hz (no kick / bass leakage into the vocal)
    plus a gentle taper above 13 kHz.
    """
    w = (1.0 / (1.0 + np.exp(-(freqs - 170.0) / 32.0))) ** 2
    w = w * (1.0 - 0.45 / (1.0 + np.exp(-(freqs - 13000.0) / 1400.0)))
    w = w * (1.0 - 0.30 / (1.0 + np.exp(-(freqs - 17000.0) / 900.0)))
    return w.astype(np.float32)


def _sep_core(mid: np.ndarray, side: np.ndarray, sr: int):
    """One block: returns (vocal_mono, inst_mid, inst_side) float32."""
    Zm, _, hop = stft(mid, N_FFT, N_FFT // 4)      # (frames, bins, 1) complex64
    Zs, _, _ = stft(side, N_FFT, N_FFT // 4)
    n = mid.shape[0]

    magM = np.abs(Zm)
    magS = np.abs(Zs)
    centre = np.clip((magM - magS) / (magM + magS + 1e-9), 0.0, 1.0)
    del magS

    # harmonic vs percussive: keeps kick / snare out of the vocal stem
    harm = median_filter(magM, size=(17, 1, 1), mode="nearest")
    perc = median_filter(magM, size=(1, 31, 1), mode="nearest")
    h_mask = (harm ** 2) / (harm ** 2 + perc ** 2 + 1e-9)
    del harm, perc

    # temporal modulation: a lead vocal fluctuates at syllable rate, pads do not
    smooth = uniform_filter1d(magM, size=41, axis=0, mode="nearest") + 1e-9
    mod = np.clip((magM - smooth) / smooth / 3.0, 0.0, 1.0)
    del magM, smooth

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr).astype(np.float32)
    w = _band_weight(freqs)[None, :, None]
    mask = (centre ** 1.8) * (0.15 + 0.85 * h_mask) * (0.60 + 0.40 * mod) * w
    del centre, h_mask, mod
    np.clip(mask, 0.0, 1.0, out=mask)
    mask = median_filter(mask, size=(3, 3, 1), mode="nearest")

    vox = istft(Zm * mask, hop, n=n)
    inst_mid = istft(Zm * (1.0 - mask), hop, n=n)
    del Zm, mask
    inst_side = istft(Zs, hop, n=n)
    return (vox.astype(np.float32, copy=False),
            inst_mid.astype(np.float32, copy=False),
            inst_side.astype(np.float32, copy=False))


def _fade(m: int, fade: int, first: bool, last: bool) -> np.ndarray:
    w = np.ones(m, dtype=np.float32)
    if fade > 0:
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)).astype(np.float32)
        if not first:
            w[:fade] = ramp
        if not last:
            w[m - fade:] = ramp[::-1]
    return w


# --------------------------------------------------------------------------- #
#  DSP separation (block-wise)
# --------------------------------------------------------------------------- #
def separate_dsp(x: np.ndarray, sr: int, progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    x = to_stereo(x)
    n = x.shape[0]
    mid = (0.5 * (x[:, 0] + x[:, 1])).astype(np.float32)
    side = (0.5 * (x[:, 0] - x[:, 1])).astype(np.float32)

    block = int(BLOCK_SEC * sr)
    over = int(OVERLAP_SEC * sr)
    if n <= block + over:
        starts = [(0, n)]
    else:
        stride = block - over
        starts = []
        a = 0
        while a + block < n:
            starts.append((a, a + block))
            a += stride
        starts.append((n - block, n))

    if len(starts) == 1:
        if progress:
            progress(0.15, "STFT + маска")
        vox, im, isd = _sep_core(mid, side, sr)
        inst = np.stack([im + isd, im - isd], axis=1)
    else:
        vox_out = np.zeros(n, dtype=np.float32)
        inst_out = np.zeros((n, 2), dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        n_blocks = len(starts)
        for k, (a, b) in enumerate(starts):
            if progress:
                progress(0.1 + 0.85 * k / n_blocks,
                         f"блок {k + 1}/{n_blocks}")
            m = b - a
            v, im, isd = _sep_core(mid[a:b].copy(), side[a:b].copy(), sr)
            w = _fade(m, min(over, m // 3), a == 0, b == n)
            vox_out[a:b] += v * w
            inst_out[a:b, 0] += (im + isd) * w
            inst_out[a:b, 1] += (im - isd) * w
            wsum[a:b] += w
        wsum[wsum < 1e-6] = 1.0
        vox = vox_out / wsum
        inst = inst_out / wsum[:, None]

    stereo_vox = np.stack([vox, vox], axis=1)
    err = float(np.max(np.abs(stereo_vox + inst - x))) if n else 0.0
    info = {
        "backend": "dsp-centre-hpss",
        "note": "Mid/Side + HPSS + modulation spectral mask, phase-coherent, block-wise",
        "block_sec": None if len(starts) == 1 else BLOCK_SEC,
        "blocks": len(starts),
        "reconstruction_error_db": round(20 * np.log10(err + EPS), 1),
        "vocal_energy_share": round(float(np.sum(vox ** 2) /
                                         (np.sum(x ** 2) + EPS)), 3),
    }
    if progress:
        progress(1.0, "готово")
    return vox.astype(np.float32), inst.astype(np.float32), info


# --------------------------------------------------------------------------- #
#  Optional Demucs backend
# --------------------------------------------------------------------------- #
def separate_demucs(x: np.ndarray, sr: int, progress=None):
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"demucs backend unavailable: {exc}") from exc

    if progress:
        progress(0.05, "загружаю htdemucs")
    model = get_model(os.environ.get("FAS_DEMUCS_MODEL", "htdemucs"))
    model.eval()

    x = to_stereo(x)
    wav = torch.from_numpy(np.ascontiguousarray(x.T.astype(np.float32)))
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    if progress:
        progress(0.2, "нейросеть работает")
    with torch.no_grad():
        out = apply_model(model, wav[None], device="cpu", shifts=1,
                          split=True, overlap=0.25, progress=False)[0]
    out = out * ref.std() + ref.mean()
    stems = dict(zip(model.sources, out.cpu().numpy()))

    vox = np.zeros(x.shape[0], dtype=np.float32)
    inst = np.zeros((x.shape[0], 2), dtype=np.float32)
    for name, data in stems.items():
        data = data.T.astype(np.float32)
        if name in ("vocals", "vocal"):
            vox += data.mean(axis=1)
        else:
            inst += data
    info = {"backend": "demucs-htdemucs", "note": "neural stem separation"}
    if progress:
        progress(1.0, "готово")
    return vox, inst, info


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def separate(x: np.ndarray, sr: int, backend: str = "auto", progress=None):
    """Separate into (vocal_mono, instrumental_stereo, info)."""
    backend = (backend or "auto").lower()
    if backend in ("auto", "demucs"):
        try:
            return separate_demucs(x, sr, progress=progress)
        except Exception as exc:  # noqa: BLE001
            if backend == "demucs":
                raise
            fallback_reason = str(exc)
        vox, inst, info = separate_dsp(x, sr, progress=progress)
        info["demucs_fallback"] = fallback_reason
        return vox, inst, info
    return separate_dsp(x, sr, progress=progress)
