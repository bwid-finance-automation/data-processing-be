"""Redis-based rate limiting service.

This module provides distributed rate limiting using Redis for:
- Request rate limiting per IP
- API endpoint rate limiting
- User-based rate limiting

Falls back to in-memory storage when Redis is unavailable.
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict

import redis.asyncio as redis

from app.core.unified_config import get_unified_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Distributed rate limiter using Redis with in-memory fallback.

    Uses sliding window algorithm for accurate rate limiting.
    Falls back gracefully to in-memory storage when Redis is unavailable.
    """

    _instance: Optional["RateLimiter"] = None
    _client: Optional[redis.Redis] = None

    # In-memory fallback storage
    _memory_store: Dict[str, list] = defaultdict(list)

    def __init__(
        self,
        requests_per_minute: int = 100,
        window_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per window
            window_seconds: Time window in seconds (default: 60)
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._client = None
        self._connected = False

    @classmethod
    async def get_instance(
        cls,
        requests_per_minute: int = 100,
        window_seconds: int = 60,
    ) -> "RateLimiter":
        """Get singleton instance with lazy initialization."""
        if cls._instance is None:
            cls._instance = cls(requests_per_minute, window_seconds)
            await cls._instance.connect()
        return cls._instance

    async def connect(self) -> None:
        """Connect to Redis server."""
        config = get_unified_config().redis

        if not config.enabled:
            logger.info("Redis disabled, using in-memory rate limiting")
            return

        try:
            self._client = redis.from_url(
                config.url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Rate limiter Redis connection failed: {e}. Using in-memory fallback.")
            self._client = None
            self._connected = False

    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None

    async def is_allowed(self, key: str) -> Tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Unique identifier (e.g., IP address, user ID)

        Returns:
            Tuple of (is_allowed, current_count, remaining)
        """
        if self.is_redis_available:
            return await self._check_redis(key)
        else:
            return self._check_memory(key)

    async def _check_redis(self, key: str) -> Tuple[bool, int, int]:
        """
        Check rate limit using Redis.

        Uses atomic INCR with EXPIRE for thread-safe counting.
        """
        try:
            redis_key = f"ratelimit:{key}"
            current_time = int(time.time())
            window_key = f"{redis_key}:{current_time // self.window_seconds}"

            # Atomic increment
            pipe = self._client.pipeline()
            pipe.incr(window_key)
            pipe.expire(window_key, self.window_seconds * 2)  # Extra buffer for safety
            results = await pipe.execute()

            current_count = results[0]
            remaining = max(0, self.requests_per_minute - current_count)
            is_allowed = current_count <= self.requests_per_minute

            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for {key}: {current_count}/{self.requests_per_minute}"
                )

            return is_allowed, current_count, remaining

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}. Falling back to memory.")
            self._connected = False
            return self._check_memory(key)

    def _check_memory(self, key: str) -> Tuple[bool, int, int]:
        """
        Check rate limit using in-memory storage.

        Uses sliding window with timestamp list.
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Clean old entries
        self._memory_store[key] = [
            ts for ts in self._memory_store[key]
            if ts > window_start
        ]

        current_count = len(self._memory_store[key])
        remaining = max(0, self.requests_per_minute - current_count)

        if current_count >= self.requests_per_minute:
            logger.warning(
                f"Rate limit exceeded for {key}: {current_count}/{self.requests_per_minute} (in-memory)"
            )
            return False, current_count, remaining

        # Add current request
        self._memory_store[key].append(current_time)
        current_count += 1
        remaining = max(0, self.requests_per_minute - current_count)

        return True, current_count, remaining

    async def get_stats(self, key: str) -> Dict:
        """
        Get current rate limit stats for a key.

        Returns:
            Dict with current_count, limit, remaining, and reset_time
        """
        if self.is_redis_available:
            try:
                current_time = int(time.time())
                window_key = f"ratelimit:{key}:{current_time // self.window_seconds}"
                current_count = int(await self._client.get(window_key) or 0)
            except Exception:
                current_count = len(self._memory_store.get(key, []))
        else:
            current_count = len(self._memory_store.get(key, []))

        return {
            "current_count": current_count,
            "limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - current_count),
            "reset_seconds": self.window_seconds,
            "backend": "redis" if self.is_redis_available else "memory",
        }

    async def reset(self, key: str) -> None:
        """
        Reset rate limit counter for a key.

        Useful for testing or administrative purposes.
        """
        if self.is_redis_available:
            try:
                pattern = f"ratelimit:{key}:*"
                keys = await self._client.keys(pattern)
                if keys:
                    await self._client.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to reset Redis rate limit for {key}: {e}")

        # Also clear memory store
        if key in self._memory_store:
            del self._memory_store[key]

    def cleanup_memory_store(self) -> int:
        """
        Clean up expired entries from in-memory store.

        Returns:
            Number of keys cleaned up
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        cleaned = 0

        keys_to_delete = []
        for key, timestamps in self._memory_store.items():
            # Keep only recent timestamps
            self._memory_store[key] = [ts for ts in timestamps if ts > window_start]
            if not self._memory_store[key]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._memory_store[key]
            cleaned += 1

        return cleaned


# Convenience function
async def get_rate_limiter(
    requests_per_minute: int = 100,
    window_seconds: int = 60,
) -> RateLimiter:
    """Get the rate limiter singleton instance."""
    return await RateLimiter.get_instance(requests_per_minute, window_seconds)
