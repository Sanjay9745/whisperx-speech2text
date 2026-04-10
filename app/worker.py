"""
RQ worker task that processes a single transcription job end to end.

The worker keeps model warm-up lazy, transcribes chunked audio, applies
alignment and diarization when possible, and now fails loudly when the
pipeline produces no usable transcript instead of silently returning zero
words.
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
from app.transcriber.align import align_segments
from app.transcriber.chunker import load_audio, prepare_chunks
from app.transcriber.diarization import assign_speakers, diarize
from app.transcriber.whisper import load_model, transcribe_chunk
from app.webhook import deliver_webhook_sync

_warmed_up = False


def _warm_up() -> None:
    """Load expensive models lazily so the worker can still boot quickly."""
    global _warmed_up
    if _warmed_up:
        return

    try:
        logger.info("Warming up Whisper model ...")
        load_model()
        _warmed_up = True
        logger.info("Whisper model ready")
    except Exception as exc:
        logger.error(f"Model warm-up failed (will retry on next job): {exc}")


def _build_text(segments: List[Dict[str, Any]], chunk_texts: List[str]) -> str:
    segment_parts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
    if segment_parts:
        return " ".join(segment_parts).strip()
    return " ".join(part for part in chunk_texts if part.strip()).strip()


def _flatten_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for seg in segments:
        for word in seg.get("words", []):
            if word.get("word", "").strip():
                words.append(word)
    return words


def _has_transcript_content(
    text: str,
    words: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> bool:
    return bool(text.strip() or words or any(seg.get("text", "").strip() for seg in segments))


def process_job(job_id: str) -> None:
    """Top-level handler invoked by RQ for each queued job."""
    logger.info(f"[{job_id}] Starting job")
    _warm_up()

    cfg = get_config()

    try:
        update_job(job_id, status="processing", progress=0)
        state = get_job_state(job_id)
        if state is None:
            raise RuntimeError("Job not found in Redis")

        file_path: str = state["file_path"]
        metadata: Dict[str, Any] = state.get("metadata", {})
        webhook_url: str = state.get("webhook_url", "")

        update_job(job_id, progress=5)
        logger.info(f"[{job_id}] Chunking audio: {file_path}")
        chunks = prepare_chunks(file_path)
        total_chunks = len(chunks)
        if total_chunks == 0:
            raise ValueError("No transcription chunks were produced from the audio.")
        logger.info(f"[{job_id}] {total_chunks} chunks to transcribe")

        all_segments: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []
        detected_language = "en"
        failed_chunks: List[Dict[str, Any]] = []

        progress_base = 10
        progress_range = 70

        def _transcribe_one(idx: int):
            audio_arr, chunk_start, _chunk_end = chunks[idx]
            result = transcribe_chunk(audio_arr)

            for seg in result.get("segments", []):
                seg["start"] = round(float(seg["start"]) + chunk_start, 3)
                seg["end"] = round(float(seg["end"]) + chunk_start, 3)
                for word in seg.get("words", []):
                    start = word.get("start")
                    end = word.get("end")
                    if start is None or end is None:
                        continue
                    word["start"] = round(float(start) + chunk_start, 3)
                    word["end"] = round(float(end) + chunk_start, 3)

            return idx, result

        max_parallel = max(1, min(cfg.performance.max_workers, total_chunks))
        completed = 0
        results_map: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(_transcribe_one, i): i for i in range(total_chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, result = future.result()
                    results_map[idx] = result
                except Exception as chunk_err:
                    logger.error(f"[{job_id}] Chunk {idx} failed: {chunk_err}")
                    try:
                        _, result = _transcribe_one(idx)
                        results_map[idx] = result
                    except Exception as retry_err:
                        logger.error(f"[{job_id}] Chunk {idx} retry failed: {retry_err}")
                        failed_chunks.append({"index": idx, "error": str(retry_err)})
                        results_map[idx] = {
                            "text": "",
                            "segments": [],
                            "language": "unknown",
                            "language_probability": 0.0,
                        }

                completed += 1
                pct = progress_base + int(progress_range * completed / total_chunks)
                update_job(job_id, progress=pct)

        non_empty_chunk_count = 0
        for index in range(total_chunks):
            result = results_map.get(
                index,
                {
                    "text": "",
                    "segments": [],
                    "language": "unknown",
                    "language_probability": 0.0,
                },
            )
            chunk_text = result.get("text", "").strip()
            segment_list = result.get("segments", [])

            if segment_list:
                all_segments.extend(segment_list)
            if chunk_text:
                all_text_parts.append(chunk_text)
            if chunk_text or any(seg.get("text", "").strip() for seg in segment_list):
                non_empty_chunk_count += 1
            if result.get("language") and result["language"] != "unknown":
                detected_language = result["language"]

        if failed_chunks and non_empty_chunk_count == 0 and not all_segments:
            first_error = failed_chunks[0]["error"]
            raise RuntimeError(f"All transcription chunks failed. First error: {first_error}")

        update_job(job_id, progress=82)
        logger.info(f"[{job_id}] Running alignment ...")
        try:
            audio_np, _sr = load_audio(file_path)
            all_segments = align_segments(audio_np, all_segments, detected_language)
        except Exception as align_err:
            logger.warning(f"[{job_id}] Alignment skipped: {align_err}")

        update_job(job_id, progress=88)
        speaker_turns: List[Dict[str, Any]] = []
        if cfg.diarization.enabled:
            logger.info(f"[{job_id}] Running diarization ...")
            try:
                speaker_turns = diarize(file_path)
                all_segments = assign_speakers(all_segments, speaker_turns)
            except Exception as diar_err:
                logger.warning(f"[{job_id}] Diarization skipped: {diar_err}")

        update_job(job_id, progress=95)

        final_text = _build_text(all_segments, all_text_parts)
        all_words = _flatten_words(all_segments)
        warnings: List[str] = []

        if failed_chunks:
            warnings.append(
                f"{len(failed_chunks)} chunk(s) failed during transcription and were omitted."
            )
        if final_text and not all_words:
            warnings.append("Transcript text was produced without word-level timestamps.")

        if not _has_transcript_content(final_text, all_words, all_segments):
            message = "No speech was detected in the audio after transcription."
            if cfg.performance.use_vad:
                message += " Try disabling VAD or using clearer audio input."
            raise ValueError(message)

        result: Dict[str, Any] = {
            "text": final_text,
            "segments": all_segments,
            "words": all_words,
            "speakers": speaker_turns,
            "language": detected_language,
            "metadata": metadata,
            "warnings": warnings,
            "stats": {
                "chunks_total": total_chunks,
                "chunks_with_content": non_empty_chunk_count,
                "chunks_failed": len(failed_chunks),
                "segment_count": len(all_segments),
                "word_count": len(all_words),
            },
        }

        set_job_result(job_id, result)
        logger.info(f"[{job_id}] Completed successfully")
        _save_output(job_id, result)

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
        try:
            state = get_job_state(job_id)
            file_path = (state or {}).get("file_path", "")
            cfg = get_config()
            if file_path and os.path.abspath(file_path).startswith(
                os.path.abspath(cfg.paths.temp_dir)
            ):
                os.unlink(file_path)
                logger.debug(f"[{job_id}] Cleaned up temp file {file_path}")
        except Exception:
            pass


def _save_output(job_id: str, result: Dict[str, Any]) -> None:
    """Persist the result JSON to the configured output directory."""
    cfg = get_config()
    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{job_id}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    logger.info(f"Result saved to {out_path}")
