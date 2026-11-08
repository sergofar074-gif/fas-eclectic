"""Pipeline orchestration.

    WAV ──▶ analyse ──▶ VAD gate ──┬─ instrumental ─▶ master bus  ─────────┐
                                   │  (separation SKIPPED)                 │
                                   └─ voice ─▶ separate ─▶ per-stem chain ─┤
                                                    │                     ▼
                                                    └─ VAD re-check ─▶ limiter ─▶ WAV
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy import signal

from . import instrumental as inst_mod
from . import master as master_mod
from . import separate as sep_mod
from . import vocal as vocal_mod
from . import vad
from . import dsp
from .measure import (integrated_lufs, peak_envelope, sample_peak_db,
                      true_peak_db)

TARGET_SR = 44100
TARGET_LUFS = -11.0
CEILING_DB = -1.0

# --------------------------------------------------------------------------- #
#  Two alternative step ladders (the UI renders whichever one is active)
# --------------------------------------------------------------------------- #
STEP_ANALYSE = "Анализирую исходник и нормализую пик до -1.0 dBFS…"
STEP_VAD = "Проверяю наличие вокала (VAD)…"
STEP_SPLIT = "Разделяю трек на вокал и музыку…"
STEP_VOCAL = "Удаляю цифровой песок и чищу вокал…"
STEP_INSTR = "Прокачиваю басы и расширяю стереобазу…"
STEP_LIMIT = "Финальный мастеринг под стандарты VK и Яндекс…"
STEP_EXPORT = "Экспортирую мастер 24-bit / 44.1 kHz…"

STEP_SKIP = "Вокал не обнаружен — разделение пропущено, трек сразу на мастер-шину…"
STEP_BUS = "Срез гула ниже 25 Гц, расширение стереобазы, склейка полос…"

STEPS_SPLIT = [STEP_ANALYSE, STEP_VAD, STEP_SPLIT, STEP_VOCAL, STEP_INSTR,
               STEP_LIMIT, STEP_EXPORT]
STEPS_INSTRUMENTAL = [STEP_ANALYSE, STEP_VAD, STEP_SKIP, STEP_BUS,
                      STEP_LIMIT, STEP_EXPORT]


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def sanitize(name: str, fallback: str = "Track") -> str:
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:80] or fallback


def master_filename(title: str) -> str:
    if not title:
        return "[FAS]__Eclectic__Master.wav"
    return f"[FAS] Eclectic — {sanitize(title)} — Master.wav"


def load_wav(path: str) -> tuple[np.ndarray, int, dict]:
    info = sf.info(path)
    if info.format != "WAV":
        raise PipelineError("UNSUPPORTED_FORMAT")
    if info.frames == 0:
        raise PipelineError("Файл пустой.")

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    duration = x.shape[0] / sr

    if duration > 12 * 60:
        raise PipelineError("Трек длиннее 12 минут — превышен лимит обработки.")
    if duration < 3.0:
        raise PipelineError("Слишком короткий файл (менее 3 секунд).")

    meta = {
        "original_sr": sr,
        "original_subtype": info.subtype,
        "channels": x.shape[1],
        "duration_sec": round(duration, 2),
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "resampled": False,
        "warnings": [],
    }
    if info.subtype and "16" in info.subtype:
        meta["warnings"].append("16-bit исходник: рекомендован 24-bit WAV из Suno v6 Pro.")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
        meta["warnings"].append("Моно-исходник развёрнут в стерео.")

    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        x = signal.resample_poly(x, TARGET_SR // g, int(sr) // g, axis=0)
        meta["resampled"] = True
        meta["warnings"].append(f"Ресемплинг {sr} Hz → {TARGET_SR} Hz (polyphase).")
        sr = TARGET_SR

    x = x - np.mean(x, axis=0, keepdims=True)
    return x.astype(np.float32), sr, meta


# --------------------------------------------------------------------------- #
#  Job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    job_id: str
    input_path: str
    out_dir: str
    title: str = ""
    backend: str = "auto"
    target_lufs: float = TARGET_LUFS
    state: dict = field(default_factory=dict)


def _write(path: str, data: np.ndarray, sr: int, subtype: str = "PCM_24"):
    sf.write(path, np.asarray(data, dtype=np.float32), sr, subtype=subtype)
    return path


# --------------------------------------------------------------------------- #
#  VAD gate
# --------------------------------------------------------------------------- #
def run_vad_gate(x: np.ndarray, sr: int, report_progress, state: dict) -> dict:
    """Stage 1: probe a couple of excerpts and decide whether a split is needed."""
    windows = vad.probe_windows(x, sr, seconds=25.0, max_windows=2)
    probe_scores = []
    for k, (a, b) in enumerate(windows):
        report_progress(8 + 5 * k, f"{STEP_VAD}  ·  фрагмент {k + 1}/{len(windows)}")
        stem, _inst, _info = sep_mod.separate_dsp(x[a:b], sr)
        del _inst
        s = vad.score_signal(stem, sr)
        s["window_sec"] = [round(a / sr, 1), round(b / sr, 1)]
        probe_scores.append(s)
        del stem
    decision = vad.decide(probe_scores)
    decision["probe_windows"] = [[round(a / sr, 1), round(b / sr, 1)] for a, b in windows]
    decision["probe_scores"] = probe_scores
    return decision


# --------------------------------------------------------------------------- #
#  Path A: pure instrumental — mastering bus only
# --------------------------------------------------------------------------- #
def path_instrumental(x: np.ndarray, sr: int, job: Job, report: dict,
                      report_progress) -> tuple[np.ndarray, dict]:
    report_progress(22, STEP_SKIP)
    i_report: dict = {}
    processed = inst_mod.master(x, sr, i_report)      # HPF 25 Hz, MS width, 4-band glue
    report_progress(62, STEP_BUS)
    gc.collect()
    return processed, i_report


# --------------------------------------------------------------------------- #
#  Path B: voice present — split, process each stem, sum back
# --------------------------------------------------------------------------- #
def path_vocal(x: np.ndarray, sr: int, job: Job, report: dict,
               report_progress) -> tuple[np.ndarray | None, dict]:
    report_progress(22, STEP_SPLIT)

    def sep_progress(frac: float, note: str):
        report_progress(22 + 28 * frac, f"{STEP_SPLIT}  ·  {note}")

    vocal_raw, inst_raw, sep_info = sep_mod.separate(
        x, sr, backend=job.backend, progress=sep_progress)

    # ---- second line of defence: is the extracted stem really a voice? ---- #
    final = vad.score_signal(vocal_raw, sr)
    share = float(np.sum(vocal_raw ** 2) / (np.sum(x ** 2) + EPS_))
    # The score decides; the energy share is only an extra guard for the
    # heuristic backend (a real but very quiet vocal must still pass).
    passed = bool(final["score"] >= vad.THRESHOLD_VOCAL and
                  final.get("hit_ratio", 0) >= vad.MIN_HIT_RATIO * 0.5)
    if final.get("backend") == "heuristic":
        passed = passed and share >= 0.008
    final.update({"energy_share": round(share, 4), "passed": passed})
    if not passed:
        del vocal_raw, inst_raw
        gc.collect()
        return None, {"final_check": final, "separation": sep_info}

    _write(os.path.join(job.out_dir, "vocal.wav"), vocal_raw, sr)
    _write(os.path.join(job.out_dir, "instrumental.wav"), inst_raw, sr)
    report_progress(55, STEP_VOCAL)

    v_report: dict = {}
    vocal_enh = vocal_mod.enhance(vocal_raw, sr, v_report)
    del vocal_raw
    gc.collect()
    _write(os.path.join(job.out_dir, "vocal_enhanced.wav"), vocal_enh, sr)

    report_progress(76, STEP_INSTR)
    i_report: dict = {}
    inst_mastered = inst_mod.master(inst_raw, sr, i_report)
    del inst_raw
    gc.collect()
    _write(os.path.join(job.out_dir, "instrumental_mastered.wav"), inst_mastered, sr)

    mix, mix_info = master_mod.mix_balance(vocal_enh, inst_mastered, sr)
    del vocal_enh, inst_mastered
    gc.collect()
    return mix, {"final_check": final, "separation": sep_info,
                 "vocal": v_report, "instrumental": i_report, "mix": mix_info}


EPS_ = 1e-12


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run_job(job: Job, progress=None) -> dict:
    t0 = time.time()
    state = job.state

    def report_progress(pct: float, msg: str | None = None):
        if progress:
            progress(min(99.0, max(0.0, float(pct))), msg)

    os.makedirs(job.out_dir, exist_ok=True)

    # ---------------- STEP 1: load & analyse ----------------------------- #
    state["steps"] = STEPS_SPLIT        # provisional; replaced after the gate
    state["step"] = 0
    report_progress(1, STEP_ANALYSE)

    x, sr, meta = load_wav(job.input_path)
    meta.update({
        "peak_db_before": round(sample_peak_db(x), 2),
        "lufs_before": round(integrated_lufs(x, sr), 2),
        "true_peak_dbtp_before": round(true_peak_db(x, sr), 2),
    })
    x = dsp.normalize_peak(x, -1.0)
    meta["peak_db_after_norm"] = round(sample_peak_db(x), 2)
    _write(os.path.join(job.out_dir, "_input_normalized.wav"), x, sr)

    report = {
        "job_id": job.job_id,
        "title": job.title or "Untitled",
        "source": meta,
    }

    # ---------------- STEP 2: VAD gate ----------------------------------- #
    report_progress(6, STEP_VAD)
    decision = run_vad_gate(x, sr, report_progress, state)
    verdict = decision["verdict"]

    # route
    if verdict == "instrumental":
        mode = "instrumental"
        state["steps"] = STEPS_INSTRUMENTAL
        state["step"] = 2
        report["vad"] = dict(decision, mode=mode,
                             reason="VAD: голоса нет — разделение пропущено")
        processed, extra = path_instrumental(x, sr, job, report, report_progress)
        report.update({k: v for k, v in extra.items()})
        report["separation"] = {"skipped": True, "reason": "instrumental track"}
    else:
        mode = "split"
        state["steps"] = STEPS_SPLIT
        state["step"] = 2
        mix, extra = path_vocal(x, sr, job, report, report_progress)
        if mix is None:
            # final check failed -> never process a fake vocal, master the mix
            mode = "instrumental"
            state["steps"] = STEPS_INSTRUMENTAL
            state["step"] = 2
            report["vad"] = dict(decision, mode=mode,
                                 reason="VAD: стем не подтвердил голос — "
                                        "разделение отменено, мастер из оригинала")
            report["separation"] = {"skipped": True,
                                    "reason": "vocal stem failed final VAD check",
                                    "discarded": True}
            report["vocal_final_check"] = extra.get("final_check")
            processed, extra2 = path_instrumental(x, sr, job, report, report_progress)
            report.update(extra2)
        else:
            report["vad"] = dict(decision, mode=mode,
                                 reason="VAD: голос найден — раздельный мастеринг")
            report["separation"] = extra.pop("separation", {})
            report["vocal"] = extra.pop("vocal", {}).get("vocal", {})
            report["instrumental"] = extra.pop("instrumental", {}).get("instrumental", {})
            report["mix"] = extra.pop("mix", {})
            report["vocal_final_check"] = extra.pop("final_check", {})
            processed = mix

    # ---------------- STEP 5/6: limiter + loudness ----------------------- #
    if mode == "instrumental":
        report_progress(78, STEP_LIMIT)
    else:
        report_progress(88, STEP_LIMIT)
    m_report: dict = {}
    final_audio = master_mod.master(processed, sr, target_lufs=job.target_lufs,
                                    ceiling_db=CEILING_DB, report=m_report)
    del processed
    gc.collect()

    report_progress(97, STEP_EXPORT)
    fname = master_filename(job.title)
    master_path = os.path.join(job.out_dir, fname)
    _write(master_path, final_audio, sr, subtype="PCM_24")

    # ---------------- report --------------------------------------------- #
    report.update({
        "file_name": fname,
        "mode": mode,
        "master": m_report.get("master", {}),
        "output": {
            "sample_rate": sr,
            "bit_depth": 24,
            "channels": 2,
            "duration_sec": round(final_audio.shape[0] / sr, 2),
            "lufs": round(integrated_lufs(final_audio, sr), 2),
            "true_peak_dbtp": round(true_peak_db(final_audio, sr), 2),
            "sample_peak_db": round(sample_peak_db(final_audio), 2),
        },
        "processing_time_sec": round(time.time() - t0, 1),
    })

    with open(os.path.join(job.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    state.update({
        "input_peaks": peak_envelope(x),
        "output_peaks": peak_envelope(final_audio),
        "report": report,
        "step": len(state.get("steps", [])) - 1,
    })
    return report


def run_job_safe(job: Job, on_progress=None, on_done=None, on_error=None):
    try:
        run_job(job, progress=on_progress)
        if on_done:
            on_done(job)
    except Exception as exc:  # noqa: BLE001
        job.state["error"] = str(exc)
        job.state["traceback"] = traceback.format_exc()
        if on_error:
            on_error(job, exc)
