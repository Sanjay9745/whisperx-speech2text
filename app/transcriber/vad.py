"""
VAD (Voice Activity Detection) using Silero-VAD.
Returns speech timestamps used for intelligent chunking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger


class VoiceActivityDetector:
    """Silero-VAD wrapper — loaded once, reused across jobs."""

    def __init__(self):
        self._model: Optional[Any] = None
        self._get_speech_timestamps: Optional[Any] = None
        self._read_audio: Optional[Any] = None

    def load(self):
        if self._model is not None:
            return
        logger.info("Loading Silero-VAD model …")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        (
            self._get_speech_timestamps,
            _,  # save_audio
            self._read_audio,
            *_,
        ) = utils
        logger.info("Silero-VAD loaded")

    def detect(
        self,
        audio_path: str,
        *,
        sampling_rate: int = 16_000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Return list of ``{start: <sec>, end: <sec>}`` speech segments.
        """
        self.load()
        wav = self._read_audio(audio_path, sampling_rate=sampling_rate)
        timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        # silero-vad <5.x returns {start: int_samples, end: int_samples}.
        # silero-vad >=5.x returns {start: float_seconds, end: float_seconds}.
        # Detect which format we have: if values look like sample counts
        # (integers >> 1) divide by sampling_rate; otherwise use as-is.
        segments = []
        for ts in timestamps:
            s, e = ts["start"], ts["end"]
            # Sample counts are always > 1 for any reasonable audio;
            # second-based values for speech are always < ~3600.
            if isinstance(s, (int, float)) and s > sampling_rate:
                # Old API — values are sample indices
                s = s / sampling_rate
                e = e / sampling_rate
            segments.append({"start": float(s), "end": float(e)})
        logger.info(f"VAD found {len(segments)} speech segments in {audio_path}")
        return segments


# Module-level singleton
vad_detector = VoiceActivityDetector()
