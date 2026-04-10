"""
Audio downloader — streams a URL to a temp file.
Validates URL, enforces size guard, and returns local file path.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

from app.config import get_config

# 4 GB ceiling (can be raised)
_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
_CHUNK_SIZE = 1024 * 256  # 256 KB


def validate_url(url: str) -> bool:
    """Basic URL sanity check."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def download_audio(url: str) -> str:
    """
    Stream-download *url* into the configured temp directory.
    Returns the absolute path of the downloaded file.
    Raises on invalid URL or HTTP errors.
    """
    if not validate_url(url):
        raise ValueError(f"Invalid audio URL: {url}")

    cfg = get_config()
    temp_dir = Path(cfg.paths.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Derive a sensible extension from the URL (fallback: .audio)
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".audio"
    if len(ext) > 6:
        ext = ".audio"
    dest = temp_dir / f"{uuid.uuid4().hex}{ext}"

    logger.info(f"Downloading audio from {url} → {dest}")

    downloaded = 0
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()

            # Optional: check Content-Length header
            cl = resp.headers.get("content-length")
            if cl and int(cl) > _MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Remote file too large ({int(cl)} bytes > {_MAX_DOWNLOAD_BYTES})"
                )

            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(_CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > _MAX_DOWNLOAD_BYTES:
                        f.close()
                        os.unlink(dest)
                        raise ValueError(
                            f"Download exceeded max size ({_MAX_DOWNLOAD_BYTES} bytes)"
                        )
                    f.write(chunk)

    logger.info(f"Downloaded {downloaded:,} bytes → {dest}")
    return str(dest.resolve())
