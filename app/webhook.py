"""
Webhook delivery with retries and timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from app.config import get_config


async def deliver_webhook(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: Optional[int] = None,
    retry_count: Optional[int] = None,
) -> bool:
    """
    POST *payload* as JSON to *url*.
    Returns True on success, False after all retries exhausted.
    """
    cfg = get_config().webhook
    _timeout = timeout_sec or cfg.timeout_sec
    _retries = retry_count or cfg.retry_count

    for attempt in range(1, _retries + 1):
        try:
            async with httpx.AsyncClient(timeout=_timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code < 300:
                    logger.info(
                        f"Webhook delivered to {url} (attempt {attempt})"
                    )
                    return True
                logger.warning(
                    f"Webhook {url} returned {resp.status_code} "
                    f"(attempt {attempt}/{_retries})"
                )
        except Exception as exc:
            logger.error(
                f"Webhook {url} failed (attempt {attempt}/{_retries}): {exc}"
            )

        # Exponential back-off between retries
        if attempt < _retries:
            await asyncio.sleep(2 ** attempt)

    logger.error(f"Webhook delivery to {url} exhausted all {_retries} retries")
    return False


def deliver_webhook_sync(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_sec: Optional[int] = None,
    retry_count: Optional[int] = None,
) -> bool:
    """Synchronous wrapper for use inside RQ workers."""
    cfg = get_config().webhook
    _timeout = timeout_sec or cfg.timeout_sec
    _retries = retry_count or cfg.retry_count

    for attempt in range(1, _retries + 1):
        try:
            with httpx.Client(timeout=_timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code < 300:
                    logger.info(
                        f"Webhook delivered to {url} (attempt {attempt})"
                    )
                    return True
                logger.warning(
                    f"Webhook {url} returned {resp.status_code} "
                    f"(attempt {attempt}/{_retries})"
                )
        except Exception as exc:
            logger.error(
                f"Webhook {url} failed (attempt {attempt}/{_retries}): {exc}"
            )

        if attempt < _retries:
            import time
            time.sleep(2 ** attempt)

    logger.error(f"Webhook delivery to {url} exhausted all {_retries} retries")
    return False
