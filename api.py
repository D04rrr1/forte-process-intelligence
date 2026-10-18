"""Forte Process Intelligence — FastAPI video analysis jobs.

Compatible with the existing Streamlit app and pipeline.analyze_video().
Run from the project root:
  python -m uvicorn api:app --host 127.0.0.1 --port 8000

Hackathon note: image evidence is served at public (unguarded) /evidence/ URLs.
Do not use with real customer recordings without evidence authorization.
"""

import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from src.pipeline import analyze_video


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly publish ONLY generated evidence folders — never data/uploads.
EVIDENCE_DIRS = {
    "handover_auto_results": DATA_ROOT / "handover_auto_results",
    "cash_auto_results": DATA_ROOT / "cash_auto_results",
    "presence_results": DATA_ROOT / "presence_results",
}
for directory in EVIDENCE_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ["FORTE_API_KEY"]
MAX_UPLOAD_MB = int(os.getenv("FORTE_MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

# One inference at a time to avoid exhausting CPU/GPU on a MacBook.
executor = ThreadPoolExecutor(max_workers=1)
jobs = {}
jobs_lock = threading.Lock()

app = FastAPI(
    title="Forte Process Intelligence API",
    version="2.0.0",
    description="Local video-based cash-discipline screening with explainable evidence.",
)

for folder_name, directory in EVIDENCE_DIRS.items():
    app.mount(
        f"/evidence/{folder_name}",
        StaticFiles(directory=str(directory)),
        name=f"evidence_{folder_name}",
    )

# Backwards compatibility with older /evidence/X-1/event_1_peak.jpg links.
app.mount(
    "/evidence",
    StaticFiles(directory=str(EVIDENCE_DIRS["handover_auto_results"])),
    name="evidence_legacy",
)


def verify_api_key(x_api_key: Optional[str]):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def evidence_url(path_string):
    """Return a safe URL for files inside one of the known evidence folders."""
    if not isinstance(path_string, str) or not path_string.strip():
        return None

    # Already normalized evidence links.
    if path_string.startswith("/evidence/"):
        remainder = path_string[len("/evidence/"):]
        if not remainder or ".." in Path(remainder).parts:
            return None
        # Preserve only known paths, including legacy handover links.
        return path_string

    path = Path(path_string)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    for folder_name, folder in EVIDENCE_DIRS.items():
        try:
            relative = path.relative_to(folder.resolve())
        except ValueError:
            continue
        if not path.is_file():
            return None
        return "/evidence/" + folder_name + "/" + quote(
            relative.as_posix(), safe="/"
        )
    return None


def convert_evidence_paths(result):
    """Normalize all attached evidence paths for external API clients."""
    missing = 0
    for collection in ("signals", "violations", "rule_candidates"):
        entries = result.get(collection, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            original = entry.get("evidence_frames", [])
            if not isinstance(original, list):
                continue
            converted = []
            for path_string in original:
                url = evidence_url(path_string)
                if url is not None:
                    converted.append(url)
                else:
                    missing += 1
            entry["evidence_frames"] = converted

    if missing:
        # Do not leak local filesystem paths through API responses.
        result.setdefault("analysis_warnings", []).append(
            f"{missing} evidence file(s) could not be published"
        )
    return result


def process_video_job(job_id, video_path):
    try:
        with jobs_lock:
            job = jobs[job_id]
            job["status"] = "processing"
            job["started_at"] = time.time()

        print(f"[JOB {job_id}] Processing {video_path.name}", flush=True)
        result = convert_evidence_paths(analyze_video(video_path))

        with jobs_lock:
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished_at"] = time.time()
        print(f"[JOB {job_id}] DONE", flush=True)

    except Exception as exc:
        traceback.print_exc()
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            jobs[job_id]["finished_at"] = time.time()
        print(f"[JOB {job_id}] FAILED", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Forte Process Intelligence"}


@app.post("/api/v1/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)

    # Handles both Unix and Windows-style client filenames.
    original_name = Path((file.filename or "").replace("\\", "/")).name
    if not original_name:
        raise HTTPException(status_code=400, detail="Filename is missing.")
    if Path(original_name).suffix.lower() != ".mov":
        raise HTTPException(status_code=400, detail="Only .mov videos are supported.")

    job_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{job_id}_{original_name}"
    bytes_written = 0
    try:
        with video_path.open("wb") as output:
            while True:
                chunk = await file.read(CHUNK_BYTES)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds {MAX_UPLOAD_MB} MB upload limit.",
                    )
                output.write(chunk)
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",  # Contract: processing / done / failed
            "video": original_name,
            "created_at": time.time(),
            "result": None,
            "error": None,
        }

    try:
        executor.submit(process_video_job, job_id, video_path)
    except RuntimeError as exc:
        with jobs_lock:
            jobs.pop(job_id, None)
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        response = {"job_id": job_id, "status": job["status"]}
        if job["status"] == "done":
            response["result"] = job["result"]
        elif job["status"] == "failed":
            response["error"] = job["error"]
        return response
