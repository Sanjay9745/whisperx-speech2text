"""
WhisperX-based word-level alignment.
Refines timestamps after initial transcription.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.config import get_config


def align_segments(
    audio: Any,  # numpy array (float32, 16 kHz, mono)
    segments: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """
    Use WhisperX alignment to refine word timestamps.
    Falls back gracefully if alignment fails.
    """
    try:
        import whisperx
    except ImportError:
        logger.warning("whisperx not installed — skipping alignment")
        return segments

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu"

    try:
        logger.info(f"Running WhisperX alignment (language={language}, device={device})")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

        # WhisperX expects segments in its own format
        whisperx_segments = []
        for seg in segments:
            whisperx_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
            )

        aligned = whisperx.align(
            whisperx_segments,
            align_model,
            align_metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )

        aligned_segments = aligned.get("segments", [])

        # Merge aligned words back into our segment format
        result = []
        for i, aseg in enumerate(aligned_segments):
            words = []
            for w in aseg.get("words", []):
                words.append(
                    {
                        "word": w.get("word", "").strip(),
                        "start": round(w.get("start", 0), 3),
                        "end": round(w.get("end", 0), 3),
                        "score": round(w.get("score", 0), 4),
                    }
                )
            result.append(
                {
                    "id": i,
                    "start": round(aseg.get("start", 0), 3),
                    "end": round(aseg.get("end", 0), 3),
                    "text": aseg.get("text", "").strip(),
                    "words": words,
                }
            )

        logger.info(f"Alignment refined {len(result)} segments")
        return result

    except Exception as exc:
        logger.error(f"WhisperX alignment failed: {exc}. Using original segments.")
        return segments
