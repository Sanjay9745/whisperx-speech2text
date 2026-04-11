"""
Configuration loader — reads config.yaml once, exposes a typed singleton.

Environment variable overrides (all optional, highest priority):
  WHISPER_MODEL_SIZE          — model size: tiny/base/small/medium/large-v2/large-v3
  WHISPER_DEVICE              — cuda | cpu
  WHISPER_COMPUTE_TYPE        — float16 | int8 | float32
  WHISPER_DOWNLOAD_ROOT       — path to cache model weights
  WHISPER_BATCH_SIZE          — int, transcription batch size
  WHISPER_MAX_WORKERS         — int, parallel transcription threads
  WHISPER_CHUNK_DURATION      — int, max chunk duration in seconds
  WHISPER_USE_VAD             — 1/true/yes to enable VAD chunking
  WHISPER_BEAM_SIZE           — int, beam search width
  WHISPER_TEMPERATURE         — float, sampling temperature
  WHISPER_BEST_OF             — int, best-of for sampling
  WHISPER_HF_TOKEN            — Hugging Face token for pyannote diarization
  WHISPER_DIARIZATION_ENABLED — 1/true/yes to enable diarization
  WHISPER_DIARIZATION_MIN_SPEAKERS — int, minimum expected speakers (e.g. 2)
  WHISPER_DIARIZATION_MAX_SPEAKERS — int, maximum expected speakers (e.g. 5)
  WHISPER_WEBHOOK_ENABLED     — 1/true/yes to enable webhook delivery
  WHISPER_WEBHOOK_TIMEOUT     — int, webhook HTTP timeout seconds
  WHISPER_WEBHOOK_RETRY_COUNT — int, number of webhook retry attempts
  WHISPER_API_KEY_ENABLED     — 1/true/yes to enforce API key auth
  WHISPER_API_KEYS            — comma-separated list of valid API keys
  WHISPER_UPLOAD_DIR          — path to uploaded audio files
  WHISPER_OUTPUT_DIR          — path to transcription result JSONs
  WHISPER_TEMP_DIR            — path for temporary downloaded files
  REDIS_URL                   — full Redis connection URL
                                (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _bool_env(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


def _opt_int(val: Any) -> Optional[int]:
    """Convert a value to int if non-empty, otherwise return None."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Typed config containers
# ---------------------------------------------------------------------------

class ModelConfig:
    def __init__(self, data: dict):
        self.size: str = data.get("size", "large-v2")
        self.device: str = data.get("device", "cuda")
        self.compute_type: str = data.get("compute_type", "float16")
        self.download_root: str = data.get("download_root", "./models")


class PerformanceConfig:
    def __init__(self, data: dict):
        self.batch_size: int = int(data.get("batch_size", 16))
        self.max_workers: int = int(data.get("max_workers", 1))
        self.chunk_duration_sec: int = int(data.get("chunk_duration_sec", 30))
        self.use_vad: bool = bool(data.get("use_vad", True))
        self.queue_soft_limit: int = int(data.get("queue_soft_limit", 10_000))
        raw_hard = data.get("queue_hard_limit", None)
        self.queue_hard_limit: Optional[int] = int(raw_hard) if raw_hard is not None else None


class AccuracyConfig:
    def __init__(self, data: dict):
        self.beam_size: int = int(data.get("beam_size", 5))
        self.temperature: float = float(data.get("temperature", 0.0))
        self.best_of: int = int(data.get("best_of", 5))


class DiarizationConfig:
    def __init__(self, data: dict):
        self.enabled: bool = bool(data.get("enabled", True))
        self.hf_token: str = data.get("hf_token", "")
        # Optional speaker count hints passed to pyannote pipeline
        self.min_speakers: Optional[int] = _opt_int(data.get("min_speakers"))
        self.max_speakers: Optional[int] = _opt_int(data.get("max_speakers"))


class WebhookConfig:
    def __init__(self, data: dict):
        self.enabled: bool = bool(data.get("enabled", True))
        self.timeout_sec: int = int(data.get("timeout_sec", 15))
        self.retry_count: int = int(data.get("retry_count", 3))


class SecurityConfig:
    def __init__(self, data: dict):
        self.api_key_enabled: bool = bool(data.get("api_key_enabled", True))
        self.api_keys: List[str] = data.get("api_keys", [])


class PathsConfig:
    def __init__(self, data: dict):
        self.upload_dir: str = data.get("upload_dir", "./uploads")
        self.output_dir: str = data.get("output_dir", "./outputs")
        self.temp_dir: str = data.get("temp_dir", "./temp")

    def ensure_dirs(self):
        for d in (self.upload_dir, self.output_dir, self.temp_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig:
    """Application-wide configuration singleton."""

    _instance: Optional["AppConfig"] = None

    def __init__(self, raw: Dict[str, Any]):
        self.model = ModelConfig(raw.get("model", {}))
        self.performance = PerformanceConfig(raw.get("performance", {}))
        self.accuracy = AccuracyConfig(raw.get("accuracy", {}))
        self.diarization = DiarizationConfig(raw.get("diarization", {}))
        self.webhook = WebhookConfig(raw.get("webhook", {}))
        self.security = SecurityConfig(raw.get("security", {}))
        self.paths = PathsConfig(raw.get("paths", {}))

        # Apply env-var overrides (highest priority)
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    def _apply_env_overrides(self):
        """Override any config value with the corresponding WHISPER_* env var."""

        # --- Model ---
        if v := os.getenv("WHISPER_MODEL_SIZE"):
            self.model.size = v
        if v := os.getenv("WHISPER_DEVICE"):
            self.model.device = v
        if v := os.getenv("WHISPER_COMPUTE_TYPE"):
            self.model.compute_type = v
        if v := os.getenv("WHISPER_DOWNLOAD_ROOT"):
            self.model.download_root = v

        # --- Performance ---
        if v := os.getenv("WHISPER_BATCH_SIZE"):
            self.performance.batch_size = int(v)
        if v := os.getenv("WHISPER_MAX_WORKERS"):
            self.performance.max_workers = int(v)
        if v := os.getenv("WHISPER_CHUNK_DURATION"):
            self.performance.chunk_duration_sec = int(v)
        if v := os.getenv("WHISPER_USE_VAD"):
            self.performance.use_vad = _bool_env(v)

        # --- Accuracy ---
        if v := os.getenv("WHISPER_BEAM_SIZE"):
            self.accuracy.beam_size = int(v)
        if v := os.getenv("WHISPER_TEMPERATURE"):
            self.accuracy.temperature = float(v)
        if v := os.getenv("WHISPER_BEST_OF"):
            self.accuracy.best_of = int(v)

        # --- Diarization ---
        if v := os.getenv("WHISPER_HF_TOKEN"):
            self.diarization.hf_token = v
        if v := os.getenv("WHISPER_DIARIZATION_ENABLED"):
            self.diarization.enabled = _bool_env(v)
        if v := os.getenv("WHISPER_DIARIZATION_MIN_SPEAKERS"):
            self.diarization.min_speakers = _opt_int(v)
        if v := os.getenv("WHISPER_DIARIZATION_MAX_SPEAKERS"):
            self.diarization.max_speakers = _opt_int(v)

        # --- Webhook ---
        if v := os.getenv("WHISPER_WEBHOOK_ENABLED"):
            self.webhook.enabled = _bool_env(v)
        if v := os.getenv("WHISPER_WEBHOOK_TIMEOUT"):
            self.webhook.timeout_sec = int(v)
        if v := os.getenv("WHISPER_WEBHOOK_RETRY_COUNT"):
            self.webhook.retry_count = int(v)

        # --- Security ---
        if v := os.getenv("WHISPER_API_KEY_ENABLED"):
            self.security.api_key_enabled = _bool_env(v)
        if v := os.getenv("WHISPER_API_KEYS"):
            # Accept comma-separated list: "key1,key2,key3"
            self.security.api_keys = [k.strip() for k in v.split(",") if k.strip()]

        # --- Paths ---
        if v := os.getenv("WHISPER_UPLOAD_DIR"):
            self.paths.upload_dir = v
        if v := os.getenv("WHISPER_OUTPUT_DIR"):
            self.paths.output_dir = v
        if v := os.getenv("WHISPER_TEMP_DIR"):
            self.paths.temp_dir = v

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        if cls._instance is not None:
            return cls._instance
        cfg_path = path or _CONFIG_PATH
        if not cfg_path.exists():
            logger.warning(f"config.yaml not found at {cfg_path} — using defaults + env vars")
            raw: Dict[str, Any] = {}
        else:
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {cfg_path}")
        cls._instance = cls(raw)
        return cls._instance

    @classmethod
    def reset(cls):
        """Force re-load on next call to get_config() — useful in tests."""
        cls._instance = None


def get_config() -> AppConfig:
    """Convenience accessor — returns the singleton AppConfig."""
    return AppConfig.load()

