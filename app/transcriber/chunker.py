"""
Audio chunker — splits audio into smaller pieces using VAD timestamps
or fixed-duration windows for long-audio support.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger

from app.config import get_config
from app.transcriber.vad import vad_detector


def load_audio(path: str, sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to *sr*."""
    import librosa

    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr


def chunk_by_vad(
    audio: np.ndarray,
    sr: int,
    vad_segments: List[dict],
    max_chunk_sec: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Merge VAD segments into chunks no longer than *max_chunk_sec*.
    Returns list of (audio_array, start_sec, end_sec).
    """
    chunks: List[Tuple[np.ndarray, float, float]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for seg in vad_segments:
        s, e = seg["start"], seg["end"]
        if current_start is None:
            current_start, current_end = s, e
            continue

        # Would appending this segment exceed max chunk length?
        if e - current_start > max_chunk_sec:
            # Flush current
            si, ei = int(current_start * sr), int(current_end * sr)
            chunks.append((audio[si:ei], current_start, current_end))
            current_start, current_end = s, e
        else:
            current_end = e

    # Flush remaining
    if current_start is not None:
        si, ei = int(current_start * sr), int(current_end * sr)
        chunks.append((audio[si:ei], current_start, current_end))

    return chunks


def chunk_fixed(
    audio: np.ndarray,
    sr: int,
    chunk_sec: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """Fallback: fixed-duration windows (no VAD)."""
    total = len(audio) / sr
    chunks = []
    start = 0.0
    while start < total:
        end = min(start + chunk_sec, total)
        si, ei = int(start * sr), int(end * sr)
        chunks.append((audio[si:ei], start, end))
        start = end
    return chunks


def prepare_chunks(
    audio_path: str,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    High-level: load audio → optionally run VAD → return chunks.
    """
    cfg = get_config()
    audio, sr = load_audio(audio_path)
    max_chunk = cfg.performance.chunk_duration_sec

    if cfg.performance.use_vad:
        logger.info("Running VAD before chunking …")
        vad_segments = vad_detector.detect(audio_path)
        if vad_segments:
            chunks = chunk_by_vad(audio, sr, vad_segments, max_chunk)
            logger.info(f"Chunked into {len(chunks)} VAD-guided pieces")
            return chunks
        logger.warning("VAD returned no segments, falling back to fixed chunking")

    chunks = chunk_fixed(audio, sr, max_chunk)
    logger.info(f"Chunked into {len(chunks)} fixed-duration pieces")
    return chunks


def save_chunk_to_file(
    chunk: np.ndarray, sr: int, out_dir: str
) -> str:
    """Write a numpy audio chunk to a temp wav file. Returns file path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(path, chunk, sr)
    return path
