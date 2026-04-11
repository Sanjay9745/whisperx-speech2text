"""
Helpers for building human-friendly transcript text from raw segments.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def format_timestamp(value: Any) -> str:
    """Return a timestamp in ``HH:MM:SS.mmm`` format."""
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02}:{minutes:02}:{remainder:06.3f}"


def build_formatted_transcript(
    segments: Iterable[Dict[str, Any]],
    fallback_text: str = "",
) -> str:
    """
    Build a display-ready transcript with timestamps and speaker labels.

    Example:
    ``[00:00:01.250 -> 00:00:03.980] Speaker 1: Hello there``
    """
    lines: List[str] = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = format_timestamp(segment.get("start"))
        end = format_timestamp(segment.get("end"))
        speaker = str(segment.get("speaker", "")).strip()
        prefix = f"[{start} -> {end}]"

        if speaker:
            lines.append(f"{prefix} {speaker}: {text}")
        else:
            lines.append(f"{prefix} {text}")

    formatted = "\n".join(lines).strip()
    return formatted or fallback_text.strip()
