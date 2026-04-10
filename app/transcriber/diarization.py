"""
Speaker diarization using pyannote.audio.
Assigns speaker labels to transcription segments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.config import get_config

_pipeline: Optional[Any] = None


def _load_pipeline():
    """Load pyannote diarization pipeline (cached)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    cfg = get_config()
    if not cfg.diarization.enabled:
        return None
    if not cfg.diarization.hf_token:
        logger.warning(
            "Diarization enabled but hf_token is empty — skipping diarization"
        )
        return None

    try:
        from pyannote.audio import Pipeline

        device = (
            torch.device("cuda")
            if torch.cuda.is_available() and cfg.model.device == "cuda"
            else torch.device("cpu")
        )

        logger.info("Loading pyannote diarization pipeline …")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=cfg.diarization.hf_token,
        )
        _pipeline = _pipeline.to(device)
        logger.info("Diarization pipeline loaded")
        return _pipeline
    except Exception as exc:
        logger.error(f"Failed to load diarization pipeline: {exc}")
        return None


def diarize(audio_path: str) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on *audio_path*.
    Returns list of {start, end, speaker}.
    """
    pipeline = _load_pipeline()
    if pipeline is None:
        return []

    try:
        logger.info(f"Running diarization on {audio_path} …")
        diarization_result = pipeline(audio_path)

        turns: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            turns.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                }
            )

        logger.info(f"Diarization found {len(set(t['speaker'] for t in turns))} speakers")
        return turns

    except Exception as exc:
        logger.error(f"Diarization failed: {exc}")
        return []


def assign_speakers(
    segments: List[Dict[str, Any]],
    speaker_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Assign a speaker label to each segment based on overlap with
    diarization turns.
    """
    if not speaker_turns:
        return segments

    def _find_speaker(start: float, end: float) -> str:
        best_overlap = 0.0
        best_speaker = "UNKNOWN"
        for turn in speaker_turns:
            overlap_start = max(start, turn["start"])
            overlap_end = min(end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        return best_speaker

    for seg in segments:
        seg["speaker"] = _find_speaker(seg["start"], seg["end"])
        # Also tag words if present
        for w in seg.get("words", []):
            w["speaker"] = _find_speaker(w.get("start", seg["start"]), w.get("end", seg["end"]))

    return segments
