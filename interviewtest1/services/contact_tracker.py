"""
Redis-backed sliding window contact frequency tracker.

Tracks how many times a given email address has contacted support
within a configurable rolling time window (default: 7 days).

Falls back to an in-memory dict if Redis is unavailable — suitable
for local development and testing without a Redis instance.

Public API:
    record_contact(email: str, window_days: int = 7) -> None
    get_contact_count(email: str, window_days: int = 7) -> int
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

import redis as redis_lib

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------

def _contact_key(email: str) -> str:
    """Redis sorted set key for a given email address."""
    return f"support:contacts:{email}"


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

class _InMemoryContactTracker:
    """
    Thread-unsafe in-memory fallback for development / testing.
    Stores (email → list[timestamp]) and prunes on read.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = defaultdict(list)

    def record(self, email: str) -> None:
        self._store[email].append(time.time())

    def count(self, email: str, window_days: int) -> int:
        cutoff = time.time() - (window_days * 86400)
        # Prune old entries
        self._store[email] = [t for t in self._store[email] if t >= cutoff]
        return len(self._store[email])


# ---------------------------------------------------------------------------
# Redis-backed tracker
# ---------------------------------------------------------------------------

class _RedisContactTracker:
    """
    Redis sorted-set based sliding window tracker.

    - Key: ``support:contacts:{email}``
    - Score: Unix timestamp (float)
    - Members: unique "{timestamp}:{uuid}" strings to allow multiple contacts
      at the same second.
    """

    def __init__(self, client: redis_lib.Redis) -> None:
        self._r = client

    def record(self, email: str, now: Optional[float] = None) -> None:
        import uuid
        now = now or time.time()
        key = _contact_key(email)
        member = f"{now}:{uuid.uuid4().hex}"
        self._r.zadd(key, {member: now})
        # Cleanup entries older than 30 days (generous buffer)
        cutoff = now - 30 * 86400
        self._r.zremrangebyscore(key, "-inf", cutoff)

    def count(self, email: str, window_days: int, now: Optional[float] = None) -> int:
        now = now or time.time()
        cutoff = now - (window_days * 86400)
        key = _contact_key(email)
        return self._r.zcount(key, cutoff, "+inf")


# ---------------------------------------------------------------------------
# Factory: choose Redis or fallback
# ---------------------------------------------------------------------------

_tracker: _RedisContactTracker | _InMemoryContactTracker | None = None


def _get_tracker() -> _RedisContactTracker | _InMemoryContactTracker:
    global _tracker
    if _tracker is not None:
        return _tracker

    if settings.redis_url:
        try:
            client = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
            client.ping()
            _tracker = _RedisContactTracker(client)
            logger.info("Contact tracker: using Redis at %s", settings.redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory tracker.", exc
            )
            _tracker = _InMemoryContactTracker()
    else:
        logger.info("No REDIS_URL configured — using in-memory contact tracker.")
        _tracker = _InMemoryContactTracker()

    return _tracker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_contact(email: str, window_days: int = 7) -> None:
    """Record an incoming contact from the given email address."""
    _get_tracker().record(email)


def get_contact_count(email: str, window_days: int = 7) -> int:
    """Return the number of contacts from *email* in the last *window_days* days."""
    return _get_tracker().count(email, window_days)
