"""
Faster-Whisper transcription engine.
Model is loaded ONCE and reused across all jobs.
Supports auto CUDA detection, FP16 / INT8 fallback.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger

from app.config import get_config

# ---------------------------------------------------------------------------
# Global model holder — loaded once per worker process
# ---------------------------------------------------------------------------

_model: Optional[WhisperModel] = None


def _detect_device_and_dtype() -> Tuple[str, str]:
    """Return (device, compute_type) after probing hardware."""
    cfg = get_config()
    device = cfg.model.device
    compute_type = cfg.model.compute_type

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU / int8")
        device = "cpu"
        compute_type = "int8"
    elif device == "cuda":
        # Verify FP16 support
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7 and compute_type == "float16":
            logger.warning(
                f"GPU compute capability {cap} < 7.0 — using int8 instead of float16"
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


# ---------------------------------------------------------------------------
# Transcribe a single audio chunk (numpy or file path)
# ---------------------------------------------------------------------------


def transcribe_chunk(
    audio: np.ndarray | str,
    *,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe a single audio chunk.
    *audio* may be a numpy float32 array (16 kHz mono) or a file path.
    Returns dict with keys: text, segments, language.
    """
    cfg = get_config()
    model = load_model()

    segments_iter, info = model.transcribe(
        audio,
        beam_size=cfg.accuracy.beam_size,
        best_of=cfg.accuracy.best_of,
        temperature=cfg.accuracy.temperature,
        language=language,
        vad_filter=False,  # we handle VAD externally
        word_timestamps=True,
    )

    segments = []
    full_text_parts: List[str] = []

    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                )
        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
        )
        full_text_parts.append(seg.text.strip())

    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
    }
