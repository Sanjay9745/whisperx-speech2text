"""
RQ Worker task — processes a single transcription job end-to-end.

Key design decisions:
  • The Whisper model is loaded ONCE when the worker process starts and
    reused for every job (via module-level singleton in whisper.py).
  • Audio is chunked (VAD or fixed), transcribed in batches, aligned,
    and optionally diarized.
  • Progress is tracked in Redis (0–100).
  • On completion a webhook is fired (if configured).
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from loguru import logger

from app.config import get_config
from app.queue import get_job_state, set_job_failed, set_job_result, update_job
from app.webhook import deliver_webhook_sync
from app.transcriber.chunker import load_audio, prepare_chunks
from app.transcriber.whisper import load_model, transcribe_chunk
from app.transcriber.align import align_segments
from app.transcriber.diarization import assign_speakers, diarize


# ---------------------------------------------------------------------------
# Ensure model is loaded at import time (worker boot).
# ---------------------------------------------------------------------------
def _warm_up():
    """Pre-load expensive models so the first job doesn't pay the cost."""
    try:
        load_model()
    except Exception as exc:
        logger.error(f"Model warm-up failed: {exc}")


# Called when the worker module is imported (once per process)
_warm_up()


# ---------------------------------------------------------------------------
# Main entry point (called by RQ)
# ---------------------------------------------------------------------------

def process_job(job_id: str) -> None:
    """Top-level handler invoked by RQ for each queued job."""
    logger.info(f"[{job_id}] Starting job")
    cfg = get_config()

    try:
        update_job(job_id, status="processing", progress=0)
        state = get_job_state(job_id)
        if state is None:
            raise RuntimeError("Job not found in Redis")

        file_path: str = state["file_path"]
        metadata: Dict[str, Any] = state.get("metadata", {})
        webhook_url: str = state.get("webhook_url", "")

        # ------------------------------------------------------------------
        # 1. Chunk audio
        # ------------------------------------------------------------------
        update_job(job_id, progress=5)
        logger.info(f"[{job_id}] Chunking audio: {file_path}")
        chunks = prepare_chunks(file_path)
        total_chunks = len(chunks)
        logger.info(f"[{job_id}] {total_chunks} chunks to transcribe")

        # ------------------------------------------------------------------
        # 2. Transcribe chunks (parallel with thread pool)
        # ------------------------------------------------------------------
        all_segments: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []
        detected_language: str = "en"

        progress_base = 10  # 10-80 for transcription
        progress_range = 70

        def _transcribe_one(idx: int):
            audio_arr, seg_start, seg_end = chunks[idx]
            result = transcribe_chunk(audio_arr)
            # Offset timestamps by chunk start
            for seg in result["segments"]:
                seg["start"] = round(seg["start"] + seg_start, 3)
                seg["end"] = round(seg["end"] + seg_start, 3)
                for w in seg.get("words", []):
                    w["start"] = round(w["start"] + seg_start, 3)
                    w["end"] = round(w["end"] + seg_start, 3)
            return idx, result

        max_parallel = min(cfg.performance.max_workers, total_chunks)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_transcribe_one, i): i for i in range(total_chunks)}
            results_map: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, res = fut.result()
                    results_map[idx] = res
                except Exception as chunk_err:
                    logger.error(f"[{job_id}] Chunk {idx} failed: {chunk_err}")
                    # Retry once
                    try:
                        _, res = _transcribe_one(idx)
                        results_map[idx] = res
                    except Exception as retry_err:
                        logger.error(
                            f"[{job_id}] Chunk {idx} retry failed: {retry_err}"
                        )
                        results_map[idx] = {
                            "text": "",
                            "segments": [],
                            "language": "en",
                        }
                completed += 1
                pct = progress_base + int(progress_range * completed / total_chunks)
                update_job(job_id, progress=pct)

        # Reassemble in order
        for i in range(total_chunks):
            r = results_map.get(i, {"text": "", "segments": [], "language": "en"})
            all_segments.extend(r["segments"])
            all_text_parts.append(r["text"])
            if r.get("language"):
                detected_language = r["language"]

        # ------------------------------------------------------------------
        # 3. WhisperX alignment
        # ------------------------------------------------------------------
        update_job(job_id, progress=82)
        logger.info(f"[{job_id}] Running alignment …")
        try:
            audio_np, sr = load_audio(file_path)
            all_segments = align_segments(audio_np, all_segments, detected_language)
        except Exception as align_err:
            logger.warning(f"[{job_id}] Alignment skipped: {align_err}")

        # ------------------------------------------------------------------
        # 4. Diarization
        # ------------------------------------------------------------------
        update_job(job_id, progress=88)
        speaker_turns: List[Dict[str, Any]] = []
        if cfg.diarization.enabled:
            logger.info(f"[{job_id}] Running diarization …")
            try:
                speaker_turns = diarize(file_path)
                all_segments = assign_speakers(all_segments, speaker_turns)
            except Exception as diar_err:
                logger.warning(f"[{job_id}] Diarization skipped: {diar_err}")

        # ------------------------------------------------------------------
        # 5. Build result
        # ------------------------------------------------------------------
        update_job(job_id, progress=95)

        # Flatten words
        all_words = []
        for seg in all_segments:
            all_words.extend(seg.get("words", []))

        result: Dict[str, Any] = {
            "text": " ".join(all_text_parts).strip(),
            "segments": all_segments,
            "words": all_words,
            "speakers": speaker_turns,
            "language": detected_language,
            "metadata": metadata,
        }

        # Persist result
        set_job_result(job_id, result)
        logger.info(f"[{job_id}] Completed successfully")

        # Save JSON to output dir
        _save_output(job_id, result)

        # ------------------------------------------------------------------
        # 6. Webhook
        # ------------------------------------------------------------------
        if webhook_url and cfg.webhook.enabled:
            logger.info(f"[{job_id}] Sending webhook to {webhook_url}")
            deliver_webhook_sync(
                webhook_url,
                {
                    "job_id": job_id,
                    "status": "completed",
                    "result": result,
                    "metadata": metadata,
                },
            )

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{job_id}] Job failed:\n{tb}")
        set_job_failed(job_id, str(exc))

        # Notify failure via webhook
        try:
            state = get_job_state(job_id)
            webhook_url = (state or {}).get("webhook_url", "")
            metadata = (state or {}).get("metadata", {})
            if webhook_url and cfg.webhook.enabled:
                deliver_webhook_sync(
                    webhook_url,
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "error": str(exc),
                        "metadata": metadata,
                    },
                )
        except Exception:
            logger.error(f"[{job_id}] Failed to send failure webhook")

    finally:
        # Clean up temp file if it came from a download
        try:
            state = get_job_state(job_id)
            fp = (state or {}).get("file_path", "")
            cfg = get_config()
            if fp and os.path.abspath(fp).startswith(
                os.path.abspath(cfg.paths.temp_dir)
            ):
                os.unlink(fp)
                logger.debug(f"[{job_id}] Cleaned up temp file {fp}")
        except Exception:
            pass


def _save_output(job_id: str, result: Dict[str, Any]):
    """Persist result JSON to the output directory."""
    cfg = get_config()
    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{job_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Result saved to {out_path}")
