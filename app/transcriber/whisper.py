"""
Faster-Whisper transcription engine.
Model is loaded ONCE per worker process and reused across all jobs.
Supports auto CUDA detection, FP16 / INT8 fallback.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger

from app.config import get_config

# ---------------------------------------------------------------------------
# Global model holder — loaded once per worker process
# ---------------------------------------------------------------------------

_model: Optional[WhisperModel] = None
_DEFAULT_FALLBACK_LANGUAGES = ("en",)


def _detect_device_and_dtype() -> Tuple[str, str]:
    """Return (device, compute_type) after probing available hardware."""
    cfg = get_config()
    device = cfg.model.device
    compute_type = cfg.model.compute_type

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU / int8")
        device = "cpu"
        compute_type = "int8"
    elif device == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7 and compute_type == "float16":
            logger.warning(
                f"GPU compute capability {cap} < 7.0 — switching to int8 instead of float16"
            )
            compute_type = "int8"

    return device, compute_type


def load_model() -> WhisperModel:
    """Load (or return cached) WhisperModel."""
    global _model
    if _model is not None:
        return _model

    cfg = get_config()
    device, compute_type = _detect_device_and_dtype()

    download_root = os.path.abspath(cfg.model.download_root)
    os.makedirs(download_root, exist_ok=True)

    logger.info(
        f"Loading faster-whisper model '{cfg.model.size}' "
        f"(device={device}, compute_type={compute_type}, "
        f"download_root={download_root})"
    )
    _model = WhisperModel(
        cfg.model.size,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=os.cpu_count() or 4,
    )
    logger.info("Whisper model loaded and cached")
    return _model


def _approximate_words(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """
    Build fallback word timings when text is present but the backend did not
    return per-word timestamps.
    """
    tokens = [token for token in text.split() if token.strip()]
    if not tokens or end <= start:
        return []

    step = (end - start) / len(tokens)
    words: List[Dict[str, Any]] = []
    for index, token in enumerate(tokens):
        word_start = start + (index * step)
        word_end = end if index == len(tokens) - 1 else start + ((index + 1) * step)
        words.append(
            {
                "word": token,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "estimated": True,
            }
        )
    return words


def _normalize_task(task: Optional[str]) -> str:
    if not task:
        return "transcribe"
    normalized = str(task).strip().lower()
    return normalized if normalized in {"transcribe", "translate"} else "transcribe"


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = str(language).strip().lower()
    return normalized or None


def _normalize_language_hints(
    language_hints: Optional[Iterable[str]],
) -> List[str]:
    normalized: List[str] = []
    for language in language_hints or []:
        candidate = _normalize_language(language)
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _has_content(result: Dict[str, Any]) -> bool:
    if result.get("text", "").strip():
        return True
    return any(seg.get("text", "").strip() for seg in result.get("segments", []))


def _run_transcription(
    model: WhisperModel,
    audio: "np.ndarray | str",
    *,
    language: Optional[str],
    task: str,
    initial_prompt: Optional[str],
) -> Dict[str, Any]:
    cfg = get_config()

    segments_iter, info = model.transcribe(
        audio,
        beam_size=cfg.accuracy.beam_size,
        best_of=cfg.accuracy.best_of,
        temperature=cfg.accuracy.temperature,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        # Chunks are already isolated with external VAD, so disabling these
        # filters avoids valid low-confidence speech getting dropped.
        compression_ratio_threshold=None,
        log_prob_threshold=None,
        no_speech_threshold=None,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )

    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for seg in segments_iter:
        segment_text = seg.text.strip()
        words: List[Dict[str, Any]] = []
        if seg.words:
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                word_text = w.word.strip()
                if not word_text:
                    continue
                words.append(
                    {
                        "word": word_text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                )

        if not words and segment_text:
            words = _approximate_words(segment_text, seg.start, seg.end)

        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": segment_text,
                "words": words,
            }
        )
        if segment_text:
            full_text_parts.append(segment_text)

    return {
        "text": " ".join(full_text_parts).strip(),
        "segments": segments,
        "language": info.language or language or "unknown",
        "language_probability": round(info.language_probability or 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Transcribe a single audio chunk
# ---------------------------------------------------------------------------

def transcribe_chunk(
    audio: "np.ndarray | str",
    *,
    language: Optional[str] = None,
    task: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    language_hints: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Transcribe a single audio chunk.
    *audio* may be a numpy float32 array (16 kHz mono) or a file path.
    Returns dict with keys: text, segments, language, language_probability.
    """
    model = load_model()
    normalized_language = _normalize_language(language)
    normalized_task = _normalize_task(task)
    normalized_hints = _normalize_language_hints(language_hints)

    result = _run_transcription(
        model,
        audio,
        language=normalized_language,
        task=normalized_task,
        initial_prompt=initial_prompt,
    )

    if _has_content(result) or normalized_language:
        return result

    candidate_languages: List[str] = []
    for candidate in normalized_hints + list(_DEFAULT_FALLBACK_LANGUAGES):
        if candidate not in candidate_languages:
            candidate_languages.append(candidate)

    for candidate_language in candidate_languages:
        if candidate_language == result.get("language"):
            continue

        logger.warning(
            "Transcription returned no text; retrying with language hint "
            f"'{candidate_language}' instead of auto-detected "
            f"'{result.get('language', 'unknown')}'"
        )
        retried = _run_transcription(
            model,
            audio,
            language=candidate_language,
            task=normalized_task,
            initial_prompt=initial_prompt,
        )
        if _has_content(retried):
            return retried
        if retried.get("language") and retried["language"] != "unknown":
            result = retried

    return result
