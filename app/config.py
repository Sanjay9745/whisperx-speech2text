"""
Configuration loader — reads config.yaml once, exposes a typed singleton.
Environment variable overrides are supported via WHISPER_* prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# ---------------------------------------------------------------------------
# Dataclass-style containers (plain classes to avoid extra deps)
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
        self.max_workers: int = int(data.get("max_workers", 4))
        self.chunk_duration_sec: int = int(data.get("chunk_duration_sec", 20))
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


class WebhookConfig:
    def __init__(self, data: dict):
        self.enabled: bool = bool(data.get("enabled", True))
        self.timeout_sec: int = int(data.get("timeout_sec", 10))
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

        # Apply env-var overrides
        self._apply_env_overrides()
        # Ensure directories exist
        self.paths.ensure_dirs()

    # ------------------------------------------------------------------
    def _apply_env_overrides(self):
        """Allow overriding key settings via environment variables."""
        if v := os.getenv("WHISPER_MODEL_SIZE"):
            self.model.size = v
        if v := os.getenv("WHISPER_DEVICE"):
            self.model.device = v
        if v := os.getenv("WHISPER_COMPUTE_TYPE"):
            self.model.compute_type = v
        if v := os.getenv("WHISPER_BATCH_SIZE"):
            self.performance.batch_size = int(v)
        if v := os.getenv("WHISPER_MAX_WORKERS"):
            self.performance.max_workers = int(v)
        if v := os.getenv("WHISPER_HF_TOKEN"):
            self.diarization.hf_token = v
        if v := os.getenv("WHISPER_DIARIZATION_ENABLED"):
            self.diarization.enabled = v.lower() in ("1", "true", "yes")
        if v := os.getenv("REDIS_URL"):
            pass  # handled directly where redis connects

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        if cls._instance is not None:
            return cls._instance
        cfg_path = path or _CONFIG_PATH
        if not cfg_path.exists():
            logger.warning(f"Config file not found at {cfg_path}, using defaults")
            raw: Dict[str, Any] = {}
        else:
            with open(cfg_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {cfg_path}")
        cls._instance = cls(raw)
        return cls._instance

    @classmethod
    def reset(cls):
        """For testing only."""
        cls._instance = None


def get_config() -> AppConfig:
    """Convenience accessor."""
    return AppConfig.load()
