"""
Speaker diarization using NVIDIA NeMo.
Assigns speaker labels to transcription segments.

Uses NeMo's NeuralDiarizer (MSDD — Multi-Scale Diarization Decoder) with
TitaNet speaker embeddings and MarbleNet VAD.  No HuggingFace token is
required — all models are downloaded from NVIDIA NGC automatically.

Compatibility notes
-------------------
* nemo_toolkit[asr] — provides NeuralDiarizer, ClusteringDiarizer, TitaNet,
  MarbleNet VAD, and all speaker-diarization utilities.
* Models are cached in the NeMo cache directory (~/.cache/nemo or the path
  set by NEMO_CACHE_DIR env var).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.config import get_config

_SPEAKER_LABEL_PATTERN = re.compile(r"speaker[_\s-]*(\d+)$", re.IGNORECASE)


def _build_nemo_config(device: torch.device, out_dir: str, yaml_cache_dir: Optional[str] = None) -> Any:
    """
    Build a NeMo OmegaConf configuration for NeuralDiarizer.

    We download NeMo's official ``diar_infer_telephonic.yaml`` and patch
    only the fields we need to change.  This avoids missing required fields
    that arise when constructing the config from scratch.

    Parameters
    ----------
    out_dir : str
        Per-job output directory (set as ``diarizer.out_dir``).
    yaml_cache_dir : str, optional
        Directory to cache the downloaded YAML so it survives per-job
        cleanup.  Defaults to *out_dir* when not provided.
    """
    from omegaconf import OmegaConf
    import urllib.request

    # Try to load official NeMo YAML config (most reliable approach).
    # Cache in yaml_cache_dir (the base dir) so it survives per-job cleanup.
    cache_dir = yaml_cache_dir or out_dir
    yaml_url = (
        "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/"
        "speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    )
    yaml_cache = os.path.join(cache_dir, "diar_infer_telephonic.yaml")

    if not os.path.exists(yaml_cache):
        try:
            logger.info(f"Downloading NeMo diarization config from: {yaml_url}")
            urllib.request.urlretrieve(yaml_url, yaml_cache)
        except Exception as dl_err:
            logger.warning(
                f"Could not download NeMo YAML config ({dl_err}); "
                "falling back to built-in config."
            )
            yaml_cache = None

    if yaml_cache and os.path.exists(yaml_cache):
        nemo_cfg = OmegaConf.load(yaml_cache)
    else:
        # Fallback: construct minimal config from scratch
        nemo_cfg = OmegaConf.create({
            "device": device.type,
            "num_workers": 1,
            "sample_rate": 16000,
            "batch_size": 64,
            "diarizer": {
                "manifest_filepath": None,
                "out_dir": out_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.875,
                        "onset": 0.8,
                        "offset": 0.6,
                        "pad_onset": 0.05,
                        "pad_offset": -0.05,
                        "min_duration_on": 0.2,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 20,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                    },
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "sigmoid_threshold": [0.7, 1.0],
                    },
                },
            },
        })

    # Always patch these fields regardless of how the config was loaded
    OmegaConf.set_struct(nemo_cfg, False)
    nemo_cfg.device = device.type
    nemo_cfg.diarizer.out_dir = out_dir
    nemo_cfg.diarizer.oracle_vad = False
    nemo_cfg.diarizer.vad.model_path = "vad_multilingual_marblenet"
    nemo_cfg.diarizer.vad.parameters.onset = 0.8
    nemo_cfg.diarizer.vad.parameters.offset = 0.6
    nemo_cfg.diarizer.vad.parameters.pad_offset = -0.05
    nemo_cfg.diarizer.speaker_embeddings.model_path = "titanet_large"
    nemo_cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
    nemo_cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
    nemo_cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.25]
    nemo_cfg.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
    nemo_cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]
    nemo_cfg.num_workers = 1

    return nemo_cfg

def _create_manifest(
    audio_path: str,
    out_dir: str,
    num_speakers: Optional[int] = None,
) -> str:
    """
    Create a NeMo-style manifest JSON file for the audio.

    NeMo diarizers expect a JSON-lines manifest with fields:
    audio_filepath, offset, duration, label, text, num_speakers,
    rttm_filepath, uem_filepath.
    """
    import soundfile as sf

    # Get audio duration
    try:
        info = sf.info(audio_path)
        duration = info.duration
    except Exception:
        # Fallback: load with librosa
        from app.transcriber.chunker import load_audio
        audio_np, sr = load_audio(audio_path, sr=16_000)
        duration = len(audio_np) / sr

    manifest_entry = {
        "audio_filepath": os.path.abspath(audio_path),
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }

    manifest_path = os.path.join(out_dir, "input_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entry, f)
        f.write("\n")

    return manifest_path


def _convert_audio_to_wav(audio_path: str, out_dir: str) -> str:
    """
    Convert audio to 16 kHz mono WAV if needed.  NeMo works best with WAV.

    Returns the path to use (original if already suitable, else converted).
    """
    import soundfile as sf

    wav_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(audio_path))[0] + "_16k.wav",
    )

    if audio_path.lower().endswith(".wav"):
        try:
            info = sf.info(audio_path)
            if info.samplerate == 16000 and info.channels == 1:
                return audio_path
        except Exception:
            pass

    from app.transcriber.chunker import load_audio

    logger.info(f"Converting audio to 16 kHz mono WAV for NeMo: {audio_path}")
    audio_np, sr = load_audio(audio_path, sr=16_000)
    sf.write(wav_path, audio_np, 16000, subtype="PCM_16")
    logger.info(f"Audio converted: {wav_path} ({len(audio_np) / sr:.1f}s)")
    return wav_path


def _parse_rttm_file(rttm_path: str) -> List[Dict[str, Any]]:
    """
    Parse an RTTM file produced by NeMo and return speaker turns.

    RTTM format (space-separated):
    SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    turns: List[Dict[str, Any]] = []

    if not os.path.exists(rttm_path):
        logger.warning(f"RTTM file not found: {rttm_path}")
        return turns

    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                turns.append({
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                    "speaker": speaker,
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse RTTM line: {line} — {e}")

    return turns


def diarize(
    audio_path: str,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on *audio_path* using NVIDIA NeMo.

    Parameters
    ----------
    min_speakers / max_speakers / num_speakers
        Hints for speaker count.  ``num_speakers`` takes precedence when
        set.  These dramatically improve accuracy for conversational audio
        where the speaker count is known.

    Returns list of ``{start, end, speaker}``.
    """
    cfg = get_config()

    if not cfg.diarization.enabled:
        logger.warning("Diarization is disabled — returning empty speaker turns.")
        return []

    try:
        from nemo.collections.asr.models import NeuralDiarizer
    except ImportError as exc:
        logger.error(
            f"nemo_toolkit[asr] is not installed: {exc}. "
            "Install it with: pip install 'nemo_toolkit[asr]'"
        )
        return []

    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and cfg.model.device == "cuda"
        else torch.device("cpu")
    )

    # Use a unique temp directory per call to avoid cross-job collisions.
    # The base nemo_diarization dir holds the downloaded YAML config only.
    nemo_base_dir = os.path.join(cfg.paths.temp_dir, "nemo_diarization")
    os.makedirs(nemo_base_dir, exist_ok=True)

    # Per-call isolated output dir so concurrent jobs never clash
    job_out_dir = tempfile.mkdtemp(prefix="nemo_job_", dir=nemo_base_dir)

    converted_wav: Optional[str] = None

    try:
        # 1. Convert audio to 16 kHz mono WAV (NeMo requires this)
        wav_path = _convert_audio_to_wav(audio_path, job_out_dir)
        if wav_path != audio_path:
            converted_wav = wav_path

        # 2. Create NeMo manifest (JSON-lines)
        manifest_path = _create_manifest(
            wav_path, job_out_dir, num_speakers=num_speakers
        )

        # 3. Build NeMo config (downloads official YAML once, then patches)
        nemo_cfg = _build_nemo_config(device, job_out_dir, yaml_cache_dir=nemo_base_dir)
        nemo_cfg.diarizer.manifest_filepath = manifest_path

        if num_speakers is not None and num_speakers > 0:
            # Oracle mode: tell NeMo exactly how many speakers there are
            nemo_cfg.diarizer.clustering.parameters.oracle_num_speakers = True
            logger.info(f"Diarization: num_speakers={num_speakers} (oracle mode)")
        else:
            nemo_cfg.diarizer.clustering.parameters.oracle_num_speakers = False
            if max_speakers is not None and max_speakers > 0:
                nemo_cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
                logger.info(f"Diarization: max_speakers={max_speakers}")
            else:
                # Default cap — avoids over-segmentation on short calls
                nemo_cfg.diarizer.clustering.parameters.max_num_speakers = 8

        # 4. Run NeMo NeuralDiarizer
        logger.info(
            f"Running NeMo diarization on: {audio_path} "
            f"(device={device.type}, out_dir={job_out_dir})"
        )
        diarizer = NeuralDiarizer(cfg=nemo_cfg).to(device)
        diarizer.diarize()

        # 5. Find the RTTM output — NeMo names it after the audio basename
        #    but we must search the folder because the exact name depends on
        #    whether we converted the file (may have a _16k suffix or not).
        raw_turns = _find_and_parse_rttm(job_out_dir)

        if not raw_turns:
            logger.warning(
                "Diarization completed but found zero speaker turns in RTTM. "
                "The audio may be too short or contain only noise."
            )
            return []

        turns = _normalize_turn_speakers(raw_turns)
        unique_speakers = set(t["speaker"] for t in turns)
        logger.info(
            f"Diarization found {len(unique_speakers)} speaker(s) "
            f"with {len(turns)} turn(s): {unique_speakers}"
        )
        return turns

    except Exception as exc:
        import traceback
        logger.error(f"Diarization failed: {exc}\n{traceback.format_exc()}")
        return []
    finally:
        # Clean up the per-job temp directory entirely
        try:
            shutil.rmtree(job_out_dir, ignore_errors=True)
        except Exception:
            pass


def _find_and_parse_rttm(out_dir: str) -> List[Dict[str, Any]]:
    """
    Scan *out_dir*/pred_rttms/ for any ``.rttm`` file and parse the first
    one found.  This is more robust than constructing the exact filename
    because NeMo's naming depends on the audio_filepath basename.
    """
    rttm_dir = os.path.join(out_dir, "pred_rttms")

    if not os.path.isdir(rttm_dir):
        logger.warning(f"NeMo pred_rttms directory not found: {rttm_dir}")
        return []

    rttm_files = [
        os.path.join(rttm_dir, f)
        for f in os.listdir(rttm_dir)
        if f.endswith(".rttm")
    ]

    if not rttm_files:
        logger.warning(f"No .rttm files found in: {rttm_dir}")
        return []

    if len(rttm_files) > 1:
        logger.warning(
            f"Multiple RTTM files found in {rttm_dir}; using first: {rttm_files[0]}"
        )

    logger.info(f"Parsing RTTM: {rttm_files[0]}")
    return _parse_rttm_file(rttm_files[0])


# ---------------------------------------------------------------------------
# Speaker-assignment & re-segmentation (unchanged from original)
# ---------------------------------------------------------------------------

def assign_speakers(
    segments: List[Dict[str, Any]],
    speaker_turns: List[Dict[str, Any]],
    *,
    force_even_word_spread: bool = False,
    prefer_specific_turns: bool = False,
) -> List[Dict[str, Any]]:
    """
    Assign a speaker label to each segment (and each word) based on
    overlap with diarization turns.

    Segments are first labeled at the word level and only then reduced
    back to segment speaker runs. This avoids flattening a multi-speaker
    segment down to a single majority speaker too early.
    """
    if not speaker_turns:
        return segments

    turns = sorted(
        speaker_turns,
        key=lambda turn: (_safe_time(turn.get("start")), _safe_time(turn.get("end"))),
    )

    def _find_speaker_at(point_start: float, point_end: float) -> Dict[str, Any]:
        """Find the best speaker turn for a given time range."""
        midpoint = point_start + ((point_end - point_start) / 2.0)
        containing_turns: List[Dict[str, Any]] = []
        overlapping_turns: List[Dict[str, Any]] = []
        nearest_distance = float("inf")
        nearest_turn: Optional[Dict[str, Any]] = None

        for turn in turns:
            turn_start = _safe_time(turn.get("start"))
            turn_end = _safe_time(turn.get("end"), turn_start)
            overlap_start = max(point_start, turn_start)
            overlap_end = min(point_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if turn_start <= midpoint <= turn_end:
                containing_turns.append(turn)
            if overlap > 0.0:
                overlapping_turns.append(turn)

            distance = min(abs(midpoint - turn_start), abs(midpoint - turn_end))
            if turn_start <= midpoint <= turn_end:
                distance = 0.0
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_turn = turn

        if containing_turns:
            if prefer_specific_turns:
                return min(
                    containing_turns,
                    key=lambda turn: (
                        _safe_time(turn.get("end")) - _safe_time(turn.get("start")),
                        abs(
                            midpoint - (
                                (_safe_time(turn.get("start")) + _safe_time(turn.get("end"))) / 2.0
                            )
                        ),
                    ),
                )
            return max(
                containing_turns,
                key=lambda turn: (
                    _overlap_ratio(turn, point_start, point_end),
                    _overlap_amount(turn, point_start, point_end),
                ),
            )

        if overlapping_turns:
            if prefer_specific_turns:
                return max(
                    overlapping_turns,
                    key=lambda turn: (
                        _overlap_ratio(turn, point_start, point_end),
                        -(_safe_time(turn.get("end")) - _safe_time(turn.get("start"))),
                        _overlap_amount(turn, point_start, point_end),
                    ),
                )
            return max(
                overlapping_turns,
                key=lambda turn: (
                    _overlap_amount(turn, point_start, point_end),
                    _overlap_ratio(turn, point_start, point_end),
                ),
            )

        return nearest_turn or {"speaker": "UNKNOWN"}

    for seg in segments:
        seg_start = _safe_time(seg.get("start", 0.0))
        seg_end = _safe_time(seg.get("end", seg_start))
        words = seg.get("words", [])

        if words:
            timestamps_unreliable = force_even_word_spread or _word_timestamps_unreliable(
                words,
                seg_start,
                seg_end,
            )

            if timestamps_unreliable and len(words) > 1:
                seg_dur = max(seg_end - seg_start, 0.001)
                step = seg_dur / len(words)
                for i, word in enumerate(words):
                    virt_start = seg_start + i * step
                    virt_end = seg_start + (i + 1) * step
                    word["start"] = round(virt_start, 3)
                    word["end"] = round(virt_end, 3)
                    matched = _find_speaker_at(virt_start, virt_end)
                    word["speaker"] = matched.get("speaker", "UNKNOWN")
                    spk_id = str(matched.get("speaker_id", "")).strip()
                    if spk_id:
                        word["speaker_id"] = spk_id
            else:
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

            first_labeled_word = next(
                (word for word in words if str(word.get("speaker", "")).strip()),
                words[0],
            )
            seg["speaker"] = str(first_labeled_word.get("speaker", "UNKNOWN"))
            first_speaker_id = str(first_labeled_word.get("speaker_id", "")).strip()
            if first_speaker_id:
                seg["speaker_id"] = first_speaker_id
        else:
            matched = _find_speaker_at(seg_start, seg_end)
            seg["speaker"] = matched.get("speaker", "UNKNOWN")
            spk_id = str(matched.get("speaker_id", "")).strip()
            if spk_id:
                seg["speaker_id"] = spk_id

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
    pauses) — NOT speaker changes.  After ``assign_speakers()`` has labeled
    each **word** with its speaker, this function splits whenever the
    speaker changes between consecutive words.
    """
    if not segments:
        return segments

    new_segments: List[Dict[str, Any]] = []
    seg_id = 1

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            seg["id"] = seg_id
            new_segments.append(seg)
            seg_id += 1
            continue

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
            seg["id"] = seg_id
            new_segments.append(seg)
            seg_id += 1
            continue

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_time(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _overlap_amount(turn: Dict[str, Any], start: float, end: float) -> float:
    turn_start = _safe_time(turn.get("start"))
    turn_end = _safe_time(turn.get("end"), turn_start)
    return max(0.0, min(end, turn_end) - max(start, turn_start))


def _overlap_ratio(turn: Dict[str, Any], start: float, end: float) -> float:
    duration = max(end - start, 0.001)
    return _overlap_amount(turn, start, end) / duration


def _word_timestamps_unreliable(
    words: List[Dict[str, Any]],
    seg_start: float,
    seg_end: float,
) -> bool:
    if len(words) <= 1:
        return False

    starts = [_safe_time(word.get("start", seg_start)) for word in words]
    ends = [_safe_time(word.get("end", seg_end)) for word in words]
    unique_starts = len({round(value, 3) for value in starts})
    zero_length = sum(1 for start, end in zip(starts, ends) if end <= start)
    non_monotonic = sum(
        1 for index in range(1, len(starts)) if starts[index] < starts[index - 1]
    )

    seg_duration = max(seg_end - seg_start, 0.001)
    covered_duration = max(ends) - min(starts) if ends and starts else 0.0

    if unique_starts <= max(2, len(words) // 3):
        return True
    if zero_length >= max(2, len(words) // 3):
        return True
    if non_monotonic > 0:
        return True
    if seg_duration > 1.0 and covered_duration < (seg_duration * 0.45):
        return True
    return False


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
