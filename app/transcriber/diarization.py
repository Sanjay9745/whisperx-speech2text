"""
Speaker diarization using pyannote.audio.
Assigns speaker labels to transcription segments.

Compatibility notes
-------------------
* pyannote.audio 4.x - use ``token=`` (not the deprecated ``use_auth_token=``).
* torch 2.6+ - changed torch.load default to weights_only=True, which breaks
  pyannote checkpoint loading (OmegaConf objects not in safe allowlist).
  We patch torch.load before importing pyannote so it defaults to False.
* speechbrain (pulled in by pyannote) - torchaudio.list_audio_backends()
  was removed in torchaudio 2.9+; see app/transcriber/diarization.py patch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.config import get_config

_pipeline: Optional[Any] = None
_SPEAKER_LABEL_PATTERN = re.compile(r"speaker[_\s-]*(\d+)$", re.IGNORECASE)


def _apply_torch_load_patch() -> None:
    """
    Patch torch.load to FORCE weights_only=False.

    torch 2.6+ changed the default to True for security, but:
    - pyannote.audio checkpoints contain OmegaConf / TorchVersion objects not in
      PyTorch's safe-globals allowlist → UnpicklingError.
    - pytorch-lightning (used internally by pyannote) calls torch.load with
      weights_only=True EXPLICITLY, so setdefault() is not enough — we must
      OVERRIDE the value to False unconditionally.

    We guard against double-patching by checking for our sentinel attribute.
    This is safe because we only load trusted HuggingFace official models.
    """
    # Guard: don't patch more than once (avoids recursive wrapper chains)
    if getattr(torch.load, "_pyannote_patched", False):
        logger.debug("torch.load already patched — skipping")
        return

    try:
        # Version check — only needed on torch >= 2.6
        try:
            from packaging.version import Version
            need_patch = Version(torch.__version__.split("+")[0]) >= Version("2.6.0")
        except ImportError:
            parts = torch.__version__.split("+")[0].split(".")
            major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            need_patch = (major, minor) >= (2, 6)

        if not need_patch:
            logger.debug(f"torch.load patch not needed (torch {torch.__version__})")
            return

        _orig = torch.load

        def _patched_load(*args, **kwargs):
            # FORCE weights_only=False — override even if caller passed True explicitly.
            # Required because pytorch-lightning passes weights_only=True directly.
            kwargs["weights_only"] = False
            return _orig(*args, **kwargs)

        _patched_load._pyannote_patched = True  # type: ignore[attr-defined]
        torch.load = _patched_load  # type: ignore[method-assign]
        logger.debug(f"torch.load weights_only=False patch applied (torch {torch.__version__})")

    except Exception as exc:
        logger.warning(f"torch.load patch failed: {exc} — pyannote may fail to load checkpoints")


def _load_pipeline():
    """Load pyannote diarization pipeline (cached)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    cfg = get_config()
    if not cfg.diarization.enabled:
        logger.info("Diarization is disabled in config")
        return None

    hf_token = cfg.diarization.hf_token
    if not hf_token:
        logger.error(
            "Diarization enabled but hf_token is empty. "
            "Set WHISPER_HF_TOKEN env var or diarization.hf_token in config.yaml. "
            "Speaker labels will NOT be generated."
        )
        return None

    # Apply torch.load compat patch BEFORE importing pyannote.
    _apply_torch_load_patch()

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        logger.error(
            f"pyannote.audio is not installed or failed to import: {exc}. "
            "Install it with: pip install pyannote.audio==4.0.1"
        )
        return None

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and cfg.model.device == "cuda"
        else torch.device("cpu")
    )

    try:
        logger.info(
            f"Loading pyannote diarization pipeline "
            f"(token={hf_token[:8]}…, device={device}) ..."
        )
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        _pipeline = _pipeline.to(device)
        logger.info("Diarization pipeline loaded successfully")
        return _pipeline
    except Exception as exc:
        import traceback
        logger.error(
            f"Failed to load diarization pipeline: {exc}\n"
            f"{traceback.format_exc()}\n"
            "Common fixes:\n"
            "  1. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept model terms at https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Ensure your HF token has 'read' scope\n"
            "  4. Check that pyannote.audio is compatible with your torch version"
        )
        return None


def _load_audio_for_diarization(audio_path: str) -> Dict[str, Any]:
    """
    Pre-load audio into memory as a dict accepted by pyannote Pipeline.

    pyannote's internal ``Audio.crop()`` relies on ffmpeg / torchaudio to read
    the file, and its duration comes from container metadata.  For container
    formats like **.mp4 / .mkv / .webm** the reported duration can exceed the
    actual number of decoded samples, causing::

        "requested chunk [...] resulted in N samples instead of expected M"

    By loading with **librosa** (which always returns exactly the decoded
    samples) and passing the waveform as an in-memory dict we bypass
    pyannote's file I/O entirely, eliminating the mismatch.

    Returns ``{"waveform": Tensor(1, T), "sample_rate": 16000}``.
    """
    from app.transcriber.chunker import load_audio  # already handles all formats

    logger.info(f"Pre-loading audio for diarization: {audio_path}")
    audio_np, sr = load_audio(audio_path, sr=16_000)
    duration_sec = len(audio_np) / sr
    logger.info(
        f"Audio loaded: {duration_sec:.1f}s, {len(audio_np)} samples @ {sr} Hz"
    )

    waveform = torch.from_numpy(audio_np).unsqueeze(0).float()  # (1, T)
    return {"waveform": waveform, "sample_rate": sr}


def diarize(audio_path: str) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on *audio_path*.
    Returns list of {start, end, speaker}.
    """
    pipeline = _load_pipeline()
    if pipeline is None:
        logger.warning(
            "Diarization pipeline is not available — returning empty speaker turns. "
            "Check earlier log messages for the root cause."
        )
        return []

    try:
        # Pre-load audio into memory to avoid pyannote's ffmpeg duration
        # mismatch with container formats (.mp4, .mkv, .webm, etc.).
        audio_input = _load_audio_for_diarization(audio_path)
        logger.info(f"Running diarization on {audio_path} ...")
        diarization_result = pipeline(audio_input)

        # pyannote.audio 4.x returns DiarizeOutput; 3.x returns Annotation.
        # DiarizeOutput wraps the Annotation in .speaker_diarization.
        annotation = getattr(
            diarization_result, "speaker_diarization", diarization_result
        )
        logger.debug(
            f"Pipeline returned {type(diarization_result).__name__}; "
            f"using {type(annotation).__name__} for itertracks"
        )

        raw_turns: List[Dict[str, Any]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_turns.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                }
            )

        if not raw_turns:
            logger.warning(
                "Diarization completed but found zero speaker turns. "
                "The audio may be too short or contain only one speaker."
            )
            return []

        turns = _normalize_turn_speakers(raw_turns)
        unique_speakers = set(t['speaker'] for t in turns)
        logger.info(
            f"Diarization found {len(unique_speakers)} speaker(s) "
            f"with {len(turns)} turn(s): {unique_speakers}"
        )
        return turns

    except Exception as exc:
        import traceback
        logger.error(f"Diarization failed: {exc}\n{traceback.format_exc()}")
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

    turns = sorted(
        speaker_turns,
        key=lambda turn: (_safe_time(turn.get("start")), _safe_time(turn.get("end"))),
    )

    def _find_speaker(start: Any, end: Any) -> Dict[str, Any]:
        segment_start = _safe_time(start)
        segment_end = _safe_time(end, segment_start)
        if segment_end < segment_start:
            segment_end = segment_start

        midpoint = segment_start + ((segment_end - segment_start) / 2.0)
        best_overlap = 0.0
        best_turn: Optional[Dict[str, Any]] = None
        nearest_gap = float("inf")
        nearest_turn: Optional[Dict[str, Any]] = None

        for turn in turns:
            turn_start = _safe_time(turn.get("start"))
            turn_end = _safe_time(turn.get("end"), turn_start)
            overlap_start = max(segment_start, turn_start)
            overlap_end = min(segment_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

            if turn_start <= midpoint <= turn_end:
                nearest_gap = 0.0
                nearest_turn = turn

            if overlap == 0.0:
                gap = min(abs(segment_start - turn_end), abs(segment_end - turn_start))
                if gap < nearest_gap:
                    nearest_gap = gap
                    nearest_turn = turn

        return best_turn or nearest_turn or {"speaker": "UNKNOWN"}

    for seg in segments:
        matched_turn = _find_speaker(seg.get("start", 0.0), seg.get("end", 0.0))
        seg["speaker"] = matched_turn.get("speaker", "UNKNOWN")
        speaker_id = str(matched_turn.get("speaker_id", "")).strip()
        if speaker_id:
            seg["speaker_id"] = speaker_id

        for word in seg.get("words", []):
            matched_word_turn = _find_speaker(
                word.get("start", seg.get("start", 0.0)),
                word.get("end", seg.get("end", 0.0)),
            )
            word["speaker"] = matched_word_turn.get("speaker", seg["speaker"])
            word_speaker_id = str(matched_word_turn.get("speaker_id", "")).strip()
            if word_speaker_id:
                word["speaker_id"] = word_speaker_id

    return segments


def _safe_time(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display_speaker_name(raw_label: Any, fallback_index: int) -> str:
    raw = str(raw_label or "").strip()
    if not raw:
        return f"Speaker {fallback_index}"

    match = _SPEAKER_LABEL_PATTERN.search(raw)
    if match:
        return f"Speaker {int(match.group(1)) + 1}"

    if raw.lower().startswith("speaker "):
        return raw.title()

    return raw


def _normalize_turn_speakers(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_map: Dict[str, str] = {}
    next_index = 1
    normalized: List[Dict[str, Any]] = []

    for turn in sorted(
        turns,
        key=lambda item: (_safe_time(item.get("start")), _safe_time(item.get("end"))),
    ):
        raw_speaker = str(turn.get("speaker", "")).strip()
        if raw_speaker not in label_map:
            label_map[raw_speaker] = _display_speaker_name(raw_speaker, next_index)
            next_index += 1

        normalized_turn = {
            "start": round(_safe_time(turn.get("start")), 3),
            "end": round(_safe_time(turn.get("end")), 3),
            "speaker": label_map[raw_speaker],
        }
        if raw_speaker:
            normalized_turn["speaker_id"] = raw_speaker

        normalized.append(normalized_turn)

    return normalized
