"""
FastAPI application — production-grade Speech-to-Text API.

Endpoints
---------
POST /transcribe   — submit audio (file upload OR URL)
GET  /status/{id}  — poll job status & progress
GET  /result/{id}  — fetch completed result
GET  /health       — liveness probe
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import get_config
from app.downloader import download_audio, validate_url
from app.queue import create_job, get_job_state
from app.security import APIKeyMiddleware

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Speech-to-Text API",
    version="1.0.0",
    description="Production-grade speech transcription powered by faster-whisper",
)

# Security middleware
app.add_middleware(APIKeyMiddleware)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    cfg = get_config()
    cfg.paths.ensure_dirs()
    logger.info("API started — directories ensured, config loaded")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
):
    """
    Submit an audio file or URL for transcription.

    Priority: file upload > audio_url.
    """
    cfg = get_config()

    # Parse metadata JSON string
    meta: dict = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON")

    # ----- Resolve audio source -----
    audio_path: Optional[str] = None

    if file is not None and file.filename:
        # Save uploaded file
        ext = Path(file.filename).suffix or ".audio"
        dest = Path(cfg.paths.upload_dir) / f"{uuid.uuid4().hex}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(dest, "wb") as out:
                shutil.copyfileobj(file.file, out)
        finally:
            await file.close()

        audio_path = str(dest.resolve())
        logger.info(f"Saved upload → {audio_path}")

    elif audio_url:
        if not validate_url(audio_url):
            raise HTTPException(status_code=400, detail="Invalid audio_url")
        try:
            audio_path = download_audio(audio_url)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to download audio: {exc}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either an audio file or audio_url",
        )

    # ----- Enqueue -----
    try:
        job_id = create_job(
            file_path=audio_path,
            metadata=meta,
            webhook_url=webhook_url,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
async def job_status(job_id: str):
    state = get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": state["status"],
        "progress": int(state.get("progress", 0)),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "error": state.get("error", ""),
    }


# ---------------------------------------------------------------------------
# GET /result/{job_id}
# ---------------------------------------------------------------------------

@app.get("/result/{job_id}")
async def job_result(job_id: str):
    state = get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if state["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail={"status": "failed", "error": state.get("error", "")},
        )

    if state["status"] != "completed":
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": state["status"],
                "progress": int(state.get("progress", 0)),
                "message": "Job is still processing",
            },
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "result": state.get("result", {}),
    }
