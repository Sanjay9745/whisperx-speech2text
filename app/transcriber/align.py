"""
Optional WhisperX-based word alignment.

When WhisperX is unavailable or returns incomplete data, the original
segments are preserved instead of replacing good transcription output with
empty words.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from loguru import logger

from app.config import get_config

try:
    import whisperx as _whisperx_probe  # noqa: F401

    _WHISPERX_AVAILABLE = True
    logger.info("whisperx found - word-level alignment enabled")
except ImportError:
    _WHISPERX_AVAILABLE = False
    logger.warning(
        "whisperx is NOT installed - word-level alignment will be skipped. "
        "Install it with: pip install --no-deps "
        "git+https://github.com/m-bain/whisperX.git@v3.8.5"
    )


def _segment_text_present(segments: List[Dict[str, Any]]) -> bool:
    return any(seg.get("text", "").strip() for seg in segments)


def _word_count(segments: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for seg in segments
        for word in seg.get("words", [])
        if word.get("word", "").strip()
    )


def _alignment_is_usable(
    original: List[Dict[str, Any]],
    aligned: List[Dict[str, Any]],
) -> bool:
    if not aligned:
        return False
    if _segment_text_present(original) and not _segment_text_present(aligned):
        return False
    if _word_count(original) > 0 and _word_count(aligned) == 0:
        return False
    return True


def align_segments(
    audio: Any,
    segments: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """
    Refine word timestamps with WhisperX when available.
    """
    if not _WHISPERX_AVAILABLE or not segments:
        return segments

    import whisperx

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu"

    try:
        logger.info(f"Running WhisperX alignment (language={language}, device={device})")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

        whisperx_segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in segments
        ]

        aligned = whisperx.align(
            whisperx_segments,
            align_model,
            align_metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )

        aligned_segments = aligned.get("segments", [])
        result: List[Dict[str, Any]] = []

        for index, aligned_segment in enumerate(aligned_segments):
            words = []
            for word in aligned_segment.get("words", []):
                word_text = word.get("word", "").strip()
                start = word.get("start")
                end = word.get("end")
                if not word_text or start is None or end is None:
                    continue
                words.append(
                    {
                        "word": word_text,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "score": round(word.get("score", 0.0), 4),
                    }
                )

            result.append(
                {
                    "id": index,
                    "start": round(aligned_segment.get("start", 0.0), 3),
                    "end": round(aligned_segment.get("end", 0.0), 3),
                    "text": aligned_segment.get("text", "").strip(),
                    "words": words,
                }
            )

        if not _alignment_is_usable(segments, result):
            logger.warning(
                "WhisperX alignment returned incomplete data; keeping original segments"
            )
            return segments

        logger.info(f"WhisperX alignment refined {len(result)} segments")
        return result

    except Exception as exc:
        logger.error(f"WhisperX alignment failed: {exc} - falling back to original segments")
        return segments
