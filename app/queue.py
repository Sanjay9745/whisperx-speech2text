"""
Redis connection + RQ queue helpers.
Stores job state, progress, and results in Redis.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import redis
from rq import Queue
from loguru import logger

from app.config import get_config

# ---------------------------------------------------------------------------
# Redis connection (singleton)
# ---------------------------------------------------------------------------

_redis_conn: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_conn
    if _redis_conn is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_conn = redis.Redis.from_url(url, decode_responses=True)
        _redis_conn.ping()
        logger.info(f"Connected to Redis at {url}")
    return _redis_conn


def get_queue() -> Queue:
    """Return the default RQ queue."""
    return Queue("transcription", connection=get_redis(), default_timeout=-1)


# ---------------------------------------------------------------------------
# Job state helpers  (stored in Redis hashes:  job:<job_id>)
# ---------------------------------------------------------------------------

_KEY = "job:{job_id}"


def _key(job_id: str) -> str:
    return _KEY.format(job_id=job_id)


def create_job(
    *,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None,
) -> str:
    """Register a new job in Redis and enqueue it via RQ.  Returns job_id."""
    cfg = get_config()
    r = get_redis()
    q = get_queue()

    job_id = str(uuid.uuid4())

    # Soft-limit warning
    queue_len = len(q)
    if queue_len >= cfg.performance.queue_soft_limit:
        logger.warning(
            f"Queue length ({queue_len}) exceeds soft limit "
            f"({cfg.performance.queue_soft_limit})"
        )

    # Hard-limit enforcement (if set)
    if (
        cfg.performance.queue_hard_limit is not None
        and queue_len >= cfg.performance.queue_hard_limit
    ):
        raise RuntimeError(
            f"Queue full: {queue_len} >= hard limit {cfg.performance.queue_hard_limit}"
        )

    state: Dict[str, str] = {
        "status": "queued",
        "progress": "0",
        "file_path": file_path,
        "metadata": json.dumps(metadata or {}),
        "webhook_url": webhook_url or "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "result": "",
        "error": "",
    }
    r.hset(_key(job_id), mapping=state)

    # Enqueue the worker task
    q.enqueue(
        "app.worker.process_job",
        job_id,
        job_id=job_id,
        job_timeout=-1,  # no timeout — long audio
        result_ttl=86400,  # keep RQ meta 24 h
    )
    logger.info(f"Job {job_id} queued (queue length: {queue_len + 1})")
    return job_id


# ---------------------------------------------------------------------------
# State read / write
# ---------------------------------------------------------------------------


def get_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    r = get_redis()
    data = r.hgetall(_key(job_id))
    if not data:
        return None
    # Deserialize metadata & result
    data["metadata"] = json.loads(data.get("metadata", "{}"))
    if data.get("result"):
        data["result"] = json.loads(data["result"])
    return data


def update_job(job_id: str, **fields: Any):
    r = get_redis()
    updates: Dict[str, str] = {"updated_at": datetime.utcnow().isoformat()}
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            updates[k] = json.dumps(v)
        else:
            updates[k] = str(v)
    r.hset(_key(job_id), mapping=updates)


def set_job_result(job_id: str, result: Dict[str, Any]):
    update_job(job_id, status="completed", progress=100, result=result)


def set_job_failed(job_id: str, error: str):
    update_job(job_id, status="failed", error=error)
