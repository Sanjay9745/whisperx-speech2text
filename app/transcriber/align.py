"""
WhisperX word-level alignment (optional).

WhisperX is installed separately (after torch) because it conflicts with
the system torch version when resolved by pip normally.  This module
gracefully degrades: if whisperx is not importable the original segments
are returned untouched.

Install command (run AFTER torch is installed):
    pip install --no-deps git+https://github.com/m-bain/whisperX.git@v3.3.1
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from loguru import logger

from app.config import get_config

# Check once at module load time so the warning appears in startup logs.
try:
    import whisperx as _whisperx_probe  # noqa: F401
    _WHISPERX_AVAILABLE = True
    logger.info("whisperx found — word-level alignment enabled")
except ImportError:
    _WHISPERX_AVAILABLE = False
    logger.warning(
        "whisperx is NOT installed — word-level alignment will be skipped. "
        "Install it with: pip install --no-deps git+https://github.com/m-bain/whisperX.git@v3.3.1"
    )


def align_segments(
    audio: Any,  # numpy float32 array, 16 kHz mono
    segments: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """
    Use WhisperX to refine word-level timestamps.
    Returns the original *segments* unchanged when whisperx is unavailable
    or when alignment raises an exception.
    """
    if not _WHISPERX_AVAILABLE:
        return segments

    if not segments:
        return segments

    import whisperx  # safe: we checked availability above

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() and cfg.model.device == "cuda" else "cpu"

    try:
        logger.info(f"Running WhisperX alignment (language={language}, device={device})")

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

        # WhisperX expects a minimal segment format
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
        for i, aseg in enumerate(aligned_segments):
            words = [
                {
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "score": round(w.get("score", 0.0), 4),
                }
                for w in aseg.get("words", [])
            ]
            result.append(
                {
                    "id": i,
                    "start": round(aseg.get("start", 0.0), 3),
                    "end": round(aseg.get("end", 0.0), 3),
                    "text": aseg.get("text", "").strip(),
                    "words": words,
                }
            )

        logger.info(f"WhisperX alignment refined {len(result)} segments")
        return result

    except Exception as exc:
        logger.error(f"WhisperX alignment failed: {exc} — falling back to original segments")
        return segments

