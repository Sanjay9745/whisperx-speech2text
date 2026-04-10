"""
Audio chunking helpers for long-form transcription.

The chunker loads and validates audio, optionally uses VAD-guided chunking,
and falls back to fixed windows when VAD is unavailable or too aggressive.
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

_MIN_AUDIO_SAMPLES = 1_600
_MIN_CHUNK_SEC = 1.0
_VAD_PAD_SEC = 0.2


def load_audio(path: str, sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """Load audio, resample to ``sr``, and validate the sample buffer."""
    import librosa

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, _orig_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size < _MIN_AUDIO_SAMPLES:
        raise ValueError("Audio is empty or too short to transcribe after loading.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid sample values.")

    return audio, sr


def _slice_audio(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    total_sec = len(audio) / sr
    start_sec = max(0.0, float(start_sec))
    end_sec = min(total_sec, float(end_sec))
    if end_sec <= start_sec:
        return np.asarray([], dtype=np.float32)

    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    return np.asarray(audio[start_idx:end_idx], dtype=np.float32)


def _pad_vad_segments(
    vad_segments: List[dict],
    *,
    total_sec: float,
    pad_sec: float,
) -> List[dict]:
    """Add a little context around VAD segments and merge overlaps."""
    padded: List[dict] = []
    for seg in vad_segments:
        start = max(0.0, float(seg["start"]) - pad_sec)
        end = min(total_sec, float(seg["end"]) + pad_sec)
        if end <= start:
            continue

        if padded and start <= padded[-1]["end"]:
            padded[-1]["end"] = max(padded[-1]["end"], end)
        else:
            padded.append({"start": start, "end": end})

    return padded


def _merge_small_chunks(
    audio: np.ndarray,
    sr: int,
    chunks: List[Tuple[np.ndarray, float, float]],
    *,
    min_chunk_sec: float,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Merge very short neighboring chunks so Whisper receives enough context.
    """
    if not chunks:
        return []

    merged: List[Tuple[np.ndarray, float, float]] = []
    pending_start = chunks[0][1]
    pending_end = chunks[0][2]

    for _, start, end in chunks[1:]:
        if pending_end - pending_start < min_chunk_sec:
            pending_end = end
            continue

        pending_audio = _slice_audio(audio, sr, pending_start, pending_end)
        if pending_audio.size:
            merged.append((pending_audio, pending_start, pending_end))
        pending_start, pending_end = start, end

    if pending_end - pending_start < min_chunk_sec and merged:
        _, previous_start, _ = merged.pop()
        pending_start = previous_start

    pending_audio = _slice_audio(audio, sr, pending_start, pending_end)
    if pending_audio.size:
        merged.append((pending_audio, pending_start, pending_end))

    return merged


def _normalize_chunks(
    audio: np.ndarray,
    sr: int,
    chunks: List[Tuple[np.ndarray, float, float]],
) -> List[Tuple[np.ndarray, float, float]]:
    cleaned: List[Tuple[np.ndarray, float, float]] = []
    for _, start, end in chunks:
        chunk_audio = _slice_audio(audio, sr, start, end)
        if chunk_audio.size == 0 or end <= start:
            continue
        cleaned.append((chunk_audio, round(start, 3), round(end, 3)))
    return cleaned


def chunk_by_vad(
    audio: np.ndarray,
    sr: int,
    vad_segments: List[dict],
    max_chunk_sec: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Merge VAD segments into chunks no longer than ``max_chunk_sec``.
    Returns ``(audio_array, start_sec, end_sec)`` tuples.
    """
    chunks: List[Tuple[np.ndarray, float, float]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for seg in vad_segments:
        start = float(seg["start"])
        end = float(seg["end"])

        if current_start is None:
            current_start, current_end = start, end
            continue

        if end - current_start > max_chunk_sec:
            chunks.append(
                (
                    _slice_audio(audio, sr, current_start, current_end),
                    current_start,
                    current_end,
                )
            )
            current_start, current_end = start, end
        else:
            current_end = end

    if current_start is not None and current_end is not None:
        chunks.append(
            (
                _slice_audio(audio, sr, current_start, current_end),
                current_start,
                current_end,
            )
        )

    return chunks


def chunk_fixed(
    audio: np.ndarray,
    sr: int,
    chunk_sec: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """Fallback fixed-duration windowing."""
    total_sec = len(audio) / sr
    chunks: List[Tuple[np.ndarray, float, float]] = []
    start = 0.0
    while start < total_sec:
        end = min(start + chunk_sec, total_sec)
        chunks.append((_slice_audio(audio, sr, start, end), start, end))
        start = end
    return chunks


def prepare_chunks(audio_path: str) -> List[Tuple[np.ndarray, float, float]]:
    """Load audio and return normalized chunks ready for transcription."""
    cfg = get_config()
    audio, sr = load_audio(audio_path)
    max_chunk_sec = max(1, cfg.performance.chunk_duration_sec)
    total_sec = len(audio) / sr

    if cfg.performance.use_vad:
        logger.info("Running VAD before chunking ...")
        try:
            vad_segments = vad_detector.detect(audio_path)
        except Exception as exc:
            logger.warning(f"VAD failed, falling back to fixed chunking: {exc}")
            vad_segments = []

        if vad_segments:
            padded_segments = _pad_vad_segments(
                vad_segments,
                total_sec=total_sec,
                pad_sec=_VAD_PAD_SEC,
            )
            chunks = chunk_by_vad(audio, sr, padded_segments, max_chunk_sec)
            chunks = _merge_small_chunks(
                audio,
                sr,
                chunks,
                min_chunk_sec=_MIN_CHUNK_SEC,
            )
            chunks = _normalize_chunks(audio, sr, chunks)
            if chunks:
                logger.info(f"Chunked into {len(chunks)} VAD-guided pieces")
                return chunks

        logger.warning("VAD returned no usable segments, falling back to fixed chunking")

    chunks = _normalize_chunks(audio, sr, chunk_fixed(audio, sr, max_chunk_sec))
    if not chunks:
        raise ValueError("No audio chunks could be produced from the input file.")

    logger.info(f"Chunked into {len(chunks)} fixed-duration pieces")
    return chunks


def save_chunk_to_file(chunk: np.ndarray, sr: int, out_dir: str) -> str:
    """Write a numpy audio chunk to a temp WAV file and return its path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(path, chunk, sr)
    return path
