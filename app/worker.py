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
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from loguru import logger

from app.config import get_config
from app.queue import get_job_state, set_job_failed, set_job_result, update_job
from app.result_formatter import build_formatted_transcript
from app.transcriber.align import align_segments
from app.transcriber.chunker import load_audio, prepare_chunks
from app.transcriber.diarization import assign_speakers, diarize, resegment_by_speakers
from app.transcriber.whisper import load_model, transcribe_chunk
from app.webhook import deliver_webhook_sync

_warmed_up = False


def _warm_up() -> None:
    """Load expensive models lazily so the worker can still boot quickly."""
    global _warmed_up
    if _warmed_up:
        return

    try:
        cfg = get_config()
        logger.info("Warming up Whisper model ...")
        load_model()

        # Log diarization config status for debugging
        if cfg.diarization.enabled:
            hf_token = cfg.diarization.hf_token
            if hf_token:
                logger.info(
                    f"Diarization is ENABLED (HF token: {hf_token[:8]}…)"
                )
            else:
                logger.warning(
                    "Diarization is ENABLED but WHISPER_HF_TOKEN is NOT SET. "
                    "Speaker labels will not be generated. "
                    "Set the WHISPER_HF_TOKEN environment variable."
                )
        else:
            logger.info("Diarization is DISABLED in config")

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


def _normalize_language(value: Any) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def _normalize_task(value: Any) -> str:
    task = str(value or "transcribe").strip().lower()
    return task if task in {"transcribe", "translate"} else "transcribe"


def _language_hints_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    hints: List[str] = []

    for key in ("language", "source_language", "language_hint"):
        value = _normalize_language(metadata.get(key))
        if value and value not in hints:
            hints.append(value)

    raw_hints = metadata.get("language_hints", [])
    if isinstance(raw_hints, str):
        raw_hints = [raw_hints]

    if isinstance(raw_hints, list):
        for value in raw_hints:
            candidate = _normalize_language(value)
            if candidate and candidate not in hints:
                hints.append(candidate)

    return hints


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
        source_language = _normalize_language(
            metadata.get("language") or metadata.get("source_language")
        ) or None
        task = _normalize_task(metadata.get("task"))
        initial_prompt = str(metadata.get("initial_prompt", "")).strip() or None
        language_hints = _language_hints_from_metadata(metadata)

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
            result = transcribe_chunk(
                audio_arr,
                language=source_language,
                task=task,
                initial_prompt=initial_prompt,
                language_hints=language_hints,
            )

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
        alignment_skipped_reason: str = ""
        if task == "translate":
            logger.info(f"[{job_id}] Skipping alignment — translate output cannot be force-aligned to source audio")
            alignment_skipped_reason = "translate"
        else:
            logger.info(f"[{job_id}] Running alignment ...")
            try:
                audio_np, _sr = load_audio(file_path)
                all_segments = align_segments(audio_np, all_segments, detected_language)
            except Exception as align_err:
                logger.warning(f"[{job_id}] Alignment skipped: {align_err}")
                alignment_skipped_reason = str(align_err)

        update_job(job_id, progress=88)
        speaker_turns: List[Dict[str, Any]] = []
        if cfg.diarization.enabled:
            logger.info(f"[{job_id}] Running diarization ...")
            try:
                # Resolve speaker-count hints: metadata > config > defaults
                diar_num = metadata.get("num_speakers") or cfg.diarization.num_speakers
                diar_min = metadata.get("min_speakers") or cfg.diarization.min_speakers
                diar_max = metadata.get("max_speakers") or cfg.diarization.max_speakers

                diar_kwargs: Dict[str, Any] = {}
                if diar_num:
                    diar_kwargs["num_speakers"] = int(diar_num)
                if diar_min and not diar_kwargs.get("num_speakers"):
                    diar_kwargs["min_speakers"] = int(diar_min)
                if diar_max and not diar_kwargs.get("num_speakers"):
                    diar_kwargs["max_speakers"] = int(diar_max)

                speaker_turns = diarize(file_path, **diar_kwargs)
                if speaker_turns:
                    logger.info(
                        f"[{job_id}] Diarization returned {len(speaker_turns)} turns, "
                        f"assigning speakers to {len(all_segments)} segments ..."
                    )
                    turn_speakers = {
                        str(turn.get("speaker", "")).strip()
                        for turn in speaker_turns
                        if str(turn.get("speaker", "")).strip()
                    }

                    def _project_segments(
                        segments: List[Dict[str, Any]],
                        *,
                        force_even_word_spread: bool,
                        prefer_specific_turns: bool,
                    ) -> List[Dict[str, Any]]:
                        projected = assign_speakers(
                            segments,
                            speaker_turns,
                            force_even_word_spread=force_even_word_spread,
                            prefer_specific_turns=prefer_specific_turns,
                        )
                        return resegment_by_speakers(projected)

                    projected_segments = _project_segments(
                        deepcopy(all_segments),
                        force_even_word_spread=(task == "translate"),
                        prefer_specific_turns=False,
                    )

                    projected_speakers = {
                        str(seg.get("speaker", "")).strip()
                        for seg in projected_segments
                        if str(seg.get("speaker", "")).strip() and seg.get("speaker") != "UNKNOWN"
                    }

                    if len(turn_speakers) > 1 and len(projected_speakers) <= 1:
                        logger.warning(
                            f"[{job_id}] Speaker turns indicate {len(turn_speakers)} speakers "
                            "but transcript segments collapsed to one speaker. "
                            "Retrying with stricter speaker-boundary projection ..."
                        )
                        strict_segments = _project_segments(
                            deepcopy(all_segments),
                            force_even_word_spread=True,
                            prefer_specific_turns=True,
                        )
                        strict_speakers = {
                            str(seg.get("speaker", "")).strip()
                            for seg in strict_segments
                            if str(seg.get("speaker", "")).strip() and seg.get("speaker") != "UNKNOWN"
                        }
                        if len(strict_speakers) > len(projected_speakers):
                            projected_segments = strict_segments
                            projected_speakers = strict_speakers
                            logger.info(
                                f"[{job_id}] Strict projection improved speaker coverage: "
                                f"{len(projected_speakers)} segment speaker(s)"
                            )

                    all_segments = projected_segments

                    assigned_count = sum(
                        1 for seg in all_segments
                        if str(seg.get("speaker", "")).strip()
                        and seg.get("speaker") != "UNKNOWN"
                    )
                    logger.info(
                        f"[{job_id}] Speakers assigned to {assigned_count}/{len(all_segments)} segments"
                    )
                else:
                    logger.warning(
                        f"[{job_id}] Diarization returned no speaker turns — "
                        "speakers will not be attached to segments. "
                        "Check the worker log above for diarization pipeline errors."
                    )
            except Exception as diar_err:
                import traceback as _tb
                logger.warning(
                    f"[{job_id}] Diarization failed: {diar_err}\n{_tb.format_exc()}"
                )
        else:
            logger.info(f"[{job_id}] Diarization is disabled in config")

        update_job(job_id, progress=95)

        final_text = _build_text(all_segments, all_text_parts)
        formatted_text = build_formatted_transcript(all_segments, final_text)
        all_words = _flatten_words(all_segments)
        warnings: List[str] = []

        if failed_chunks:
            warnings.append(
                f"{len(failed_chunks)} chunk(s) failed during transcription and were omitted."
            )
        if alignment_skipped_reason == "translate":
            warnings.append(
                "Alignment was skipped: translate mode produces English text that does not "
                "match source-language audio timestamps."
            )
        elif alignment_skipped_reason:
            warnings.append(f"Word-level alignment skipped: {alignment_skipped_reason}")
        if final_text and not all_words:
            warnings.append("Transcript text was produced without word-level timestamps.")
        if cfg.diarization.enabled and all_segments and not any(
            str(seg.get("speaker", "")).strip() for seg in all_segments
        ):
            hf_token_set = bool(cfg.diarization.hf_token)
            warnings.append(
                "Speaker labels were not attached to transcript segments. "
                + (
                    "WHISPER_HF_TOKEN is not set — diarization requires a valid HF token. "
                    "Get one at https://huggingface.co/settings/tokens"
                    if not hf_token_set
                    else "The HF token is set but no speaker turns were detected. "
                    "This is normal for very short audio (< 10 s) or single-speaker recordings. "
                    "For multi-speaker audio, check the worker log for pipeline errors."
                )
            )

        if not _has_transcript_content(final_text, all_words, all_segments):
            message = "No speech was detected in the audio after transcription."
            if cfg.performance.use_vad:
                message += " Try disabling VAD or using clearer audio input."
            raise ValueError(message)

        result: Dict[str, Any] = {
            "text": final_text,
            "formatted_text": formatted_text,
            "segments": all_segments,
            "words": all_words,
            "speakers": speaker_turns,
            "language": detected_language,
            "task": task,
            "metadata": metadata,
            "warnings": warnings,
            "stats": {
                "chunks_total": total_chunks,
                "chunks_with_content": non_empty_chunk_count,
                "chunks_failed": len(failed_chunks),
                "segment_count": len(all_segments),
                "speaker_turn_count": len(speaker_turns),
                "speaker_count": len(
                    {str(t.get("speaker", "")).strip() for t in speaker_turns if t.get("speaker")}
                ),
                "speakers_in_segments": len(
                    {str(seg.get("speaker", "")).strip() for seg in all_segments if seg.get("speaker")}
                ),
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
