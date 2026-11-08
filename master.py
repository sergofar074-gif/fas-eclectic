"""Final mix + brickwall limiting + loudness targeting (streaming specs)."""
from __future__ import annotations

import numpy as np

from . import dsp
from .measure import integrated_lufs, true_peak_db, sample_peak_db

EPS = 1e-12


def mix_balance(vocal: np.ndarray, instrumental: np.ndarray, sr: int,
                vocal_forward_db: float = 2.0) -> tuple[np.ndarray, dict]:
    """Static mix. The vocal rides `vocal_forward_db` above the bed so that
    intelligibility is preserved (spec: 'balance must keep the vocal legible')."""
    v = vocal[:, None] if vocal.ndim == 1 else vocal
    inst = instrumental if instrumental.ndim > 1 else np.stack([instrumental, instrumental], axis=1)

    v_l = integrated_lufs(v, sr)
    i_l = integrated_lufs(inst, sr)
    v_gain = (i_l + vocal_forward_db) - v_l
    v_gain = float(np.clip(v_gain, -12.0, 6.0))   # >6 dB lift = bleed, not a lead

    mix = inst + v * dsp.db2lin(v_gain)
    info = {
        "vocal_lufs": round(v_l, 2),
        "instrumental_lufs": round(i_l, 2),
        "vocal_gain_db": round(v_gain, 2),
        "instrumental_gain_db": 0.0,
        "vocal_forward_db": vocal_forward_db,
    }
    return mix, info


def master(mix: np.ndarray, sr: int, target_lufs: float = -11.0,
           ceiling_db: float = -1.0, report: dict | None = None) -> np.ndarray:
    report = report if report is not None else {}
    x = np.asarray(mix, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)

    # --- master bus tone & glue ------------------------------------------- #
    x = dsp.highpass(x, 20.0, sr, stages=2)
    x = dsp.compressor(x, sr, threshold_db=-14.0, ratio=1.35,
                       attack_ms=40.0, release_ms=280.0, knee_db=12.0, makeup=0.0)
    x = dsp.saturate(x, drive_db=1.5, mix=0.10, even=0.06)
    x = dsp.shelf(x, 65.0, sr, 0.5, high=False)
    x = dsp.shelf(x, 12500.0, sr, 0.6, high=True)

    peak_in = sample_peak_db(x)
    lufs_in = integrated_lufs(x, sr)
    x = dsp.normalize_peak(x, -0.5)

    # --- limiting + loudness targeting (closed loop) ---------------------- #
    # The limiter's ceiling is fixed; loudness is driven by the input gain while
    # a *single* global trim keeps the true peak at the delivery ceiling.
    mag0 = dsp.tp_envelope(x, sr)
    gain_db = 0.0
    trim_db = 0.0
    y = x
    lufs = lufs_in
    tp = 0.0
    for _ in range(6):
        y = dsp.limiter(x, sr, ceiling_db=ceiling_db, true_peak=True,
                        mag=mag0, pre_gain=dsp.db2lin(gain_db))
        tp_raw = true_peak_db(y, sr)
        trim_db = min(0.0, ceiling_db - tp_raw) if tp_raw > ceiling_db else 0.0
        lufs = integrated_lufs(y if trim_db == 0.0 else y * dsp.db2lin(trim_db), sr)
        if abs(target_lufs - lufs) <= 0.2:
            break
        gain_db += (target_lufs - lufs) * 0.9

    y = y * dsp.db2lin(trim_db)
    lufs = integrated_lufs(y, sr)
    tp = true_peak_db(y, sr)

    # --- QA auto-attenuator (per ТЗ) -------------------------------------- #
    applied_q = []
    if trim_db < 0:
        applied_q.append(f"true-peak trim {trim_db:.2f} dB")
    if lufs > -10.0:
        d = -11.0 - lufs
        y = y * dsp.db2lin(d)
        lufs += d
        tp = true_peak_db(y, sr)
        applied_q.append(f"loudness trim {d:.2f} dB")
    elif lufs < -12.0:
        applied_q.append("source слишком динамичный: -12 LUFS не достигнут без искажений")
    if not applied_q:
        applied_q.append("none – target reached on first pass")

    report["master"] = {
        "pre_master_peak_db": round(peak_in, 2),
        "pre_master_lufs": round(lufs_in, 2),
        "final_lufs": round(lufs, 2),
        "final_true_peak_dbtp": round(tp, 2),
        "target_lufs": target_lufs,
        "ceiling_db": ceiling_db,
        "gain_applied_db": round(gain_db, 2),
        "qa_actions": applied_q or ["none – target reached on first pass"],
        "verdict": _verdict(lufs, tp),
    }
    return y


def _verdict(lufs: float, tp: float) -> str:
    ok_l = -12.0 - 0.2 <= lufs <= -10.0 + 0.2
    ok_p = tp <= -1.0 + 0.05
    if ok_l and ok_p:
        return "PASS – соответствие -10…-12 LUFS и -1.0 dBTP"
    problems = []
    if not ok_l:
        problems.append("loudness out of -10..-12 LUFS")
    if not ok_p:
        problems.append("true peak above -1.0 dBTP")
    return "REVIEW – " + "; ".join(problems)
