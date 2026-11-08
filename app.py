"""FAS ECLECTIC AUDIO ENHANCER — web application (FastAPI).

Upload a Suno WAV -> separation -> per-stem AI processing -> final master.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.runner import Job, PipelineError, run_job, master_filename, sanitize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.environ.get("FAS_JOBS_DIR", os.path.join(BASE_DIR, "jobs"))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_BYTES = int(float(os.environ.get("FAS_MAX_MB", "100")) * 1024 * 1024)

os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="FAS Eclectic Audio Enhancer", version="1.0")

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_QUEUE = threading.Semaphore(1)   # one heavy job at a time (CPU/RAM bound)

MP3_WARNING = ("Внимание! Для сохранения коммерческого качества и роялти "
               "используйте только WAV исходники из Suno v6 Pro.")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _new_job(title: str) -> str:
    jid = uuid.uuid4().hex[:12]
    out = os.path.join(JOBS_DIR, jid)
    os.makedirs(out, exist_ok=True)
    with _LOCK:
        JOBS[jid] = {
            "id": jid,
            "title": title,
            "status": "queued",
            "progress": 0.0,
            "message": "В очереди…",
            "steps": [],
            "created": time.time(),
            "dir": out,
            "result": None,
            "error": None,
        }
    return jid


def _sniff_is_wav(path: str) -> tuple[bool, str]:
    with open(path, "rb") as fh:
        head = fh.read(12)
    if len(head) < 12:
        return False, "empty"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True, "wav"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return False, "mp3"
    if head[:4] == b"fLaC":
        return False, "flac"
    if head[4:8] == b"ftyp":
        return False, "m4a"
    return False, "unknown"


def _job_paths(jid: str) -> tuple[dict, str]:
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return job, job["dir"]


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.post("/api/process")
async def process(file: Optional[UploadFile] = File(None),
                  title: str = Form(""),
                  backend: str = Form("auto"),
                  demo: str = Form(""),
                  target_lufs: float = Form(-11.0)):
    """Accept a WAV, queue the pipeline, return the job id."""
    title = sanitize(title, fallback="")

    jid = _new_job(title)
    job = JOBS[jid]
    src = os.path.join(job["dir"], "source.wav")

    if demo:
        demo_path = os.path.join(SAMPLES_DIR, "test_mix.wav")
        if not os.path.exists(demo_path):
            raise HTTPException(410, "Демо-трек недоступен")
        shutil.copyfile(demo_path, src)
        if not title:
            job["title"] = "Demo — Eclectic Test"
    else:
        if file is None:
            raise HTTPException(400, "Файл не передан")

        name = (file.filename or "").lower()
        total = 0
        with open(src, "wb") as out:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    out.close()
                    os.remove(src)
                    raise HTTPException(413, "Файл больше 100 МБ. Загрузите WAV до 100 МБ.")
                out.write(chunk)

        ok, kind = _sniff_is_wav(src)
        if not ok:
            os.remove(src)
            if kind == "mp3" or name.endswith(".mp3"):
                raise HTTPException(415, detail={"code": "MP3_NOT_ALLOWED",
                                                 "message": MP3_WARNING})
            raise HTTPException(415, detail={
                "code": "NOT_WAV",
                "message": ("Формат не поддерживается. Принимается только .wav "
                            "(PCM 16/24/32-bit, 44.1 kHz и выше). " + MP3_WARNING)})

    job["status"] = "running"
    job["message"] = "Анализирую исходник…"

    def on_progress(pct: float, msg: str | None):
        job["progress"] = round(pct, 1)
        if msg:
            job["message"] = msg

    def on_done(j: Job):
        report = j.state.get("report", {})
        job["status"] = "done"
        job["progress"] = 100.0
        job["message"] = "Готово"
        job["steps"] = j.state.get("steps", [])
        # only advertise artefacts that really exist on disk
        wanted = [("master", None), ("vocal", "vocal.wav"),
                  ("instrumental", "instrumental.wav"),
                  ("vocal_enhanced", "vocal_enhanced.wav"),
                  ("instrumental_mastered", "instrumental_mastered.wav"),
                  ("report", "report.json")]
        stems = {}
        for key, fname in wanted:
            if key == "master":
                ok = bool(report.get("file_name")) and os.path.exists(
                    os.path.join(job["dir"], report["file_name"]))
            else:
                ok = os.path.exists(os.path.join(job["dir"], fname))
            if ok:
                stems[key] = f"/api/download/{jid}/{key}"
        job["result"] = {
            "file_name": report.get("file_name"),
            "mode": report.get("mode", "split"),
            "report": report,
            "input_peaks": j.state.get("input_peaks", []),
            "output_peaks": j.state.get("output_peaks", []),
            "stems": stems,
        }

    def on_error(j: Job, exc: Exception):
        job["status"] = "error"
        job["error"] = str(exc) or exc.__class__.__name__
        job["message"] = "Ошибка обработки"
        if os.environ.get("FAS_DEBUG"):
            job["traceback"] = j.state.get("traceback")

    def worker():
        with _QUEUE:
            j = Job(job_id=jid, input_path=src, out_dir=job["dir"],
                    title=job["title"], backend=backend,
                    target_lufs=float(target_lufs), state=job)
            from pipeline.runner import run_job_safe
            run_job_safe(j, on_progress=on_progress, on_done=on_done, on_error=on_error)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": jid, "title": job["title"]}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    job, _ = _job_paths(jid)
    return {
        "id": job["id"],
        "title": job["title"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "steps": job.get("steps", []),
        "result": job.get("result"),
        "error": job.get("error"),
    }


ARTIFACTS = {
    "master": None,           # resolved dynamically (name contains the title)
    "vocal": "vocal.wav",
    "instrumental": "instrumental.wav",
    "vocal_enhanced": "vocal_enhanced.wav",
    "instrumental_mastered": "instrumental_mastered.wav",
    "report": "report.json",
}


@app.get("/api/download/{jid}/{artifact}")
def download(jid: str, artifact: str, inline: int = 0):
    job, d = _job_paths(jid)
    if artifact == "master":
        rep = (job.get("result") or {}).get("report") or {}
        fname = rep.get("file_name") or master_filename(job["title"])
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            raise HTTPException(404, "Мастер-файл ещё не готов")
        media = "audio/wav"
    else:
        fname = ARTIFACTS.get(artifact)
        if not fname:
            raise HTTPException(404, "Неизвестный артефакт")
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            raise HTTPException(404, "Файл не найден")
        media = "application/json" if artifact == "report" else "audio/wav"

    return FileResponse(path, media_type=media, filename=fname,
                        content_disposition_type="inline" if inline else "attachment")


@app.delete("/api/job/{jid}")
def delete_job(jid: str):
    job, d = _job_paths(jid)
    if job["status"] == "running":
        raise HTTPException(409, "Задача ещё выполняется")
    with _LOCK:
        JOBS.pop(jid, None)
    shutil.rmtree(d, ignore_errors=True)
    return {"deleted": jid}


PROJECT_ZIP = os.environ.get("FAS_ZIP",
                             os.path.join(os.path.dirname(BASE_DIR),
                                          "fas-eclectic.zip"))


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(os.path.join(STATIC_DIR, "manifest.webmanifest"),
                        media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return FileResponse(os.path.join(STATIC_DIR, "sw.js"),
                        media_type="application/javascript",
                        headers={"Cache-Control": "no-cache",
                                 "Service-Worker-Allowed": "/"})


@app.get("/download/project.zip")
def download_project():
    """Source archive of the whole project (built by make_zip.sh)."""
    if not os.path.exists(PROJECT_ZIP):
        raise HTTPException(404, "Архив проекта ещё не собран")
    return FileResponse(PROJECT_ZIP, media_type="application/zip",
                        filename="fas-eclectic.zip")


@app.get("/api/health")
def health():
    try:
        import numpy, scipy, soundfile  # noqa: F401
        deps = "ok"
    except Exception as exc:  # noqa: BLE001
        deps = str(exc)
    from pipeline import vad
    return {"status": "ok", "deps": deps,
            "separator": os.environ.get("FAS_SEPARATOR", "auto"),
            "vad_backend": vad.backend_available(),
            "project_zip": os.path.exists(PROJECT_ZIP)}


# --------------------------------------------------------------------------- #
#  Static
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)),
                log_level="info")
