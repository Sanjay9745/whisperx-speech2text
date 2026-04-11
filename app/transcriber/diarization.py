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
from collections import Counter
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


def diarize(
    audio_path: str,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on *audio_path*.

    Parameters
    ----------
    min_speakers / max_speakers / num_speakers
        Hints passed directly to pyannote's pipeline.  ``num_speakers``
        takes precedence when set.  These dramatically improve accuracy
        for conversational audio where the speaker count is known.

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

        # Build optional kwargs for speaker count hints
        pipeline_kwargs: Dict[str, Any] = {}
        if num_speakers is not None and num_speakers > 0:
            pipeline_kwargs["num_speakers"] = num_speakers
            logger.info(f"Diarization: num_speakers={num_speakers}")
        else:
            if min_speakers is not None and min_speakers > 0:
                pipeline_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None and max_speakers > 0:
                pipeline_kwargs["max_speakers"] = max_speakers
            if pipeline_kwargs:
                logger.info(f"Diarization: {pipeline_kwargs}")

        logger.info(f"Running diarization on {audio_path} ...")
        diarization_result = pipeline(audio_input, **pipeline_kwargs)

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
    Assign a speaker label to each segment (and each word) based on
    overlap with diarization turns.

    Strategy
    --------
    1. Build a sorted list of diarization turns.
    2. For every word in a segment, find the turn with the greatest overlap.
       - If word timestamps are *collapsed* (all identical — common in
         ``translate`` mode where alignment is skipped), distribute the
         words evenly across the segment's time range so each word gets a
         distinct virtual timestamp.
    3. Set the **segment-level** speaker to the speaker that owns the
       majority of words in that segment.  This is more accurate than
       picking the turn with the longest overlap against the entire
       segment range, because a shorter minority-speaker turn that
       dominates the first or second half of the segment will be
       correctly captured.
    """
    if not speaker_turns:
        return segments

    turns = sorted(
        speaker_turns,
        key=lambda turn: (_safe_time(turn.get("start")), _safe_time(turn.get("end"))),
    )

    def _find_speaker_at(point_start: float, point_end: float) -> Dict[str, Any]:
        """Find the best speaker turn for a given time range."""
        best_overlap = 0.0
        best_turn: Optional[Dict[str, Any]] = None
        nearest_gap = float("inf")
        nearest_turn: Optional[Dict[str, Any]] = None

        for turn in turns:
            turn_start = _safe_time(turn.get("start"))
            turn_end = _safe_time(turn.get("end"), turn_start)
            overlap_start = max(point_start, turn_start)
            overlap_end = min(point_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

            # For zero-length points, check containment
            if point_start == point_end and turn_start <= point_start <= turn_end:
                if best_overlap == 0.0:
                    best_overlap = 0.001  # tiny but nonzero
                    best_turn = turn

            if overlap == 0.0:
                gap = min(abs(point_start - turn_end), abs(point_end - turn_start))
                if gap < nearest_gap:
                    nearest_gap = gap
                    nearest_turn = turn

        return best_turn or nearest_turn or {"speaker": "UNKNOWN"}

    for seg in segments:
        seg_start = _safe_time(seg.get("start", 0.0))
        seg_end = _safe_time(seg.get("end", seg_start))
        words = seg.get("words", [])

        if words:
            # Detect whether word timestamps are collapsed (all identical)
            word_starts = [_safe_time(w.get("start", seg_start)) for w in words]
            word_ends = [_safe_time(w.get("end", seg_end)) for w in words]
            timestamps_collapsed = (
                len(set(word_starts)) <= 2 and len(words) > 3
            )

            if timestamps_collapsed and len(words) > 1:
                # Distribute words evenly across the segment's time range
                seg_dur = max(seg_end - seg_start, 0.001)
                step = seg_dur / len(words)
                for i, word in enumerate(words):
                    virt_start = seg_start + i * step
                    virt_end = seg_start + (i + 1) * step
                    matched = _find_speaker_at(virt_start, virt_end)
                    word["speaker"] = matched.get("speaker", "UNKNOWN")
                    spk_id = str(matched.get("speaker_id", "")).strip()
                    if spk_id:
                        word["speaker_id"] = spk_id
            else:
                # Normal path — word timestamps are usable
                for word in words:
                    w_start = _safe_time(word.get("start", seg_start))
                    w_end = _safe_time(word.get("end", w_start))
                    if w_end < w_start:
                        w_end = w_start
                    matched = _find_speaker_at(w_start, w_end)
                    word["speaker"] = matched.get("speaker", "UNKNOWN")
                    spk_id = str(matched.get("speaker_id", "")).strip()
                    if spk_id:
                        word["speaker_id"] = spk_id

            # Determine segment speaker from word majority
            speaker_counts = Counter(
                str(w.get("speaker", "UNKNOWN")) for w in words
            )
            majority_speaker = speaker_counts.most_common(1)[0][0]
            seg["speaker"] = majority_speaker
            # Find matching speaker_id
            for w in words:
                if w.get("speaker") == majority_speaker and w.get("speaker_id"):
                    seg["speaker_id"] = w["speaker_id"]
                    break
        else:
            # No words — fall back to segment-level overlap
            matched = _find_speaker_at(seg_start, seg_end)
            seg["speaker"] = matched.get("speaker", "UNKNOWN")
            spk_id = str(matched.get("speaker_id", "")).strip()
            if spk_id:
                seg["speaker_id"] = spk_id

    # Log summary
    seg_speakers = Counter(str(s.get("speaker", "")) for s in segments)
    logger.info(
        f"Speaker assignment complete: {len(segments)} segments → "
        f"{dict(seg_speakers)}"
    )

    return segments


def resegment_by_speakers(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Split segments at speaker-change boundaries.

    Whisper creates segments based on *content* boundaries (sentences,
    pauses) — NOT speaker changes.  This means a single Whisper segment
    may contain words from two speakers when they speak in quick
    succession (common in conversations).

    After ``assign_speakers()`` has labeled each **word** with its speaker,
    this function walks through every segment and splits it whenever the
    speaker changes between consecutive words.  The result is a list of
    segments where **each segment has exactly one speaker**.

    Segments without words are passed through unchanged.
    """
    if not segments:
        return segments

    new_segments: List[Dict[str, Any]] = []
    seg_id = 1

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # No words → keep as-is
            seg["id"] = seg_id
            new_segments.append(seg)
            seg_id += 1
            continue

        # Group consecutive words by speaker
        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = [words[0]]
        current_speaker = str(words[0].get("speaker", ""))

        for w in words[1:]:
            w_speaker = str(w.get("speaker", ""))
            if w_speaker == current_speaker:
                current_group.append(w)
            else:
                groups.append(current_group)
                current_group = [w]
                current_speaker = w_speaker

        groups.append(current_group)

        if len(groups) == 1:
            # No speaker change — keep segment intact
            seg["id"] = seg_id
            new_segments.append(seg)
            seg_id += 1
            continue

        # Multiple speakers in this segment → split
        for group_words in groups:
            group_speaker = str(group_words[0].get("speaker", "UNKNOWN"))
            group_speaker_id = ""
            for gw in group_words:
                sid = str(gw.get("speaker_id", "")).strip()
                if sid:
                    group_speaker_id = sid
                    break

            group_text = " ".join(
                str(w.get("word", "")).strip() for w in group_words
            ).strip()

            # Determine time range from word timestamps
            starts = [
                _safe_time(w.get("start"))
                for w in group_words
                if w.get("start") is not None
            ]
            ends = [
                _safe_time(w.get("end"))
                for w in group_words
                if w.get("end") is not None
            ]
            group_start = min(starts) if starts else _safe_time(seg.get("start"))
            group_end = max(ends) if ends else _safe_time(seg.get("end"))

            new_seg: Dict[str, Any] = {
                "id": seg_id,
                "start": round(group_start, 3),
                "end": round(group_end, 3),
                "text": group_text,
                "words": group_words,
                "speaker": group_speaker,
            }
            if group_speaker_id:
                new_seg["speaker_id"] = group_speaker_id
            new_segments.append(new_seg)
            seg_id += 1

    if len(new_segments) != len(segments):
        logger.info(
            f"Re-segmented by speaker changes: {len(segments)} → "
            f"{len(new_segments)} segments"
        )

    return new_segments


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
