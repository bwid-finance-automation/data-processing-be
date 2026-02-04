"""Redis caching service for OCR and AI results."""

import hashlib
import json
from typing import Optional, Any, Dict

import redis.asyncio as redis

from app.core.unified_config import get_unified_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class RedisCacheService:
    """Async Redis cache service for OCR and AI results.

    Features:
    - Singleton pattern with lazy initialization
    - Graceful degradation when Redis is unavailable
    - SHA256 content hashing for cache keys
    - Configurable TTL per cache type
    """

    _instance: Optional["RedisCacheService"] = None
    _client: Optional[redis.Redis] = None

    def __init__(self):
        self.config = get_unified_config().redis
        self._client = None
        self._connected = False

    @classmethod
    async def get_instance(cls) -> "RedisCacheService":
        """Get singleton instance with lazy initialization."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.connect()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
        cls._client = None

    async def connect(self) -> None:
        """Connect to Redis server."""
        if not self.config.enabled:
            logger.info("Redis caching is disabled via configuration")
            return

        try:
            self._client = redis.from_url(
                self.config.url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.url}")
        except redis.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self._client = None
            self._connected = False
        except Exception as e:
            logger.warning(f"Redis initialization error: {e}. Caching will be disabled.")
            self._client = None
            self._connected = False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                await self._client.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected and available."""
        return self._connected and self._client is not None

    @staticmethod
    def hash_content(content: bytes) -> str:
        """Generate SHA256 hash of content for use as cache key.

        Args:
            content: Binary content to hash

        Returns:
            Hexadecimal SHA256 hash string
        """
        return hashlib.sha256(content).hexdigest()

    # ==================== OCR Cache Methods ====================

    async def get_ocr_result(self, file_hash: str) -> Optional[str]:
        """Get cached OCR result by file content hash.

        Args:
            file_hash: SHA256 hash of the PDF file content

        Returns:
            Cached OCR text if found, None otherwise
        """
        if not self.is_connected:
            return None

        try:
            key = f"ocr:{file_hash}"
            result = await self._client.get(key)
            if result:
                logger.debug(f"OCR cache hit for hash {file_hash[:16]}...")
            return result
        except Exception as e:
            logger.warning(f"Redis get_ocr_result error: {e}")
            return None

    async def set_ocr_result(self, file_hash: str, text: str) -> bool:
        """Cache OCR result with TTL.

        Args:
            file_hash: SHA256 hash of the PDF file content
            text: Extracted OCR text to cache

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            key = f"ocr:{file_hash}"
            await self._client.setex(key, self.config.ocr_cache_ttl, text)
            logger.debug(f"OCR result cached for hash {file_hash[:16]}... (TTL: {self.config.ocr_cache_ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Redis set_ocr_result error: {e}")
            return False

    # ==================== AI Analysis Cache Methods ====================

    async def get_ai_result(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached AI analysis result.

        Args:
            prompt_hash: SHA256 hash of the prompt/input data

        Returns:
            Cached result dict if found, None otherwise
        """
        if not self.is_connected:
            return None

        try:
            key = f"ai:{prompt_hash}"
            data = await self._client.get(key)
            if data:
                logger.debug(f"AI cache hit for hash {prompt_hash[:16]}...")
                return json.loads(data)
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode cached AI result: {e}")
            return None
        except Exception as e:
            logger.warning(f"Redis get_ai_result error: {e}")
            return None

    async def set_ai_result(self, prompt_hash: str, result: Dict[str, Any]) -> bool:
        """Cache AI analysis result with TTL.

        Args:
            prompt_hash: SHA256 hash of the prompt/input data
            result: Analysis result dict to cache

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            key = f"ai:{prompt_hash}"
            await self._client.setex(
                key,
                self.config.ai_cache_ttl,
                json.dumps(result, ensure_ascii=False)
            )
            logger.debug(f"AI result cached for hash {prompt_hash[:16]}... (TTL: {self.config.ai_cache_ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Redis set_ai_result error: {e}")
            return False

    # ==================== Classification Cache Methods ====================

    async def get_classification(self, cache_key: str) -> Optional[str]:
        """Get cached AI classification result.

        Args:
            cache_key: MD5 hash of description+is_receipt

        Returns:
            Cached category string if found, None otherwise
        """
        if not self.is_connected:
            return None

        try:
            key = f"classify:{cache_key}"
            result = await self._client.get(key)
            if result:
                logger.debug(f"Classification cache hit for {cache_key[:12]}...")
            return result
        except Exception as e:
            logger.warning(f"Redis get_classification error: {e}")
            return None

    async def get_classifications_bulk(self, cache_keys: list[str]) -> Dict[str, str]:
        """Get multiple cached classification results in one round-trip.

        Args:
            cache_keys: List of MD5 hash keys

        Returns:
            Dict mapping cache_key -> category for found entries
        """
        if not self.is_connected or not cache_keys:
            return {}

        try:
            redis_keys = [f"classify:{k}" for k in cache_keys]
            values = await self._client.mget(redis_keys)
            results = {}
            for key, val in zip(cache_keys, values):
                if val is not None:
                    results[key] = val
            if results:
                logger.info(f"Classification cache: {len(results)}/{len(cache_keys)} hits")
            return results
        except Exception as e:
            logger.warning(f"Redis get_classifications_bulk error: {e}")
            return {}

    async def set_classifications_bulk(self, entries: Dict[str, str]) -> bool:
        """Cache multiple classification results with TTL.

        Args:
            entries: Dict mapping cache_key -> category

        Returns:
            True if cached successfully
        """
        if not self.is_connected or not entries:
            return False

        try:
            pipe = self._client.pipeline()
            for cache_key, category in entries.items():
                pipe.setex(f"classify:{cache_key}", self.config.ai_cache_ttl, category)
            await pipe.execute()
            logger.info(f"Cached {len(entries)} classification results (TTL: {self.config.ai_cache_ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Redis set_classifications_bulk error: {e}")
            return False

    async def clear_classification_cache(self) -> int:
        """Clear all classification cache entries.

        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match="classify:*"):
                keys.append(key)
            if keys:
                await self._client.delete(*keys)
            logger.info(f"Cleared {len(keys)} classification cache entries")
            return len(keys)
        except Exception as e:
            logger.warning(f"Redis clear_classification_cache error: {e}")
            return 0

    # ==================== Utility Methods ====================

    async def delete_key(self, key: str) -> bool:
        """Delete a specific cache key.

        Args:
            key: Full cache key to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    async def clear_ocr_cache(self) -> int:
        """Clear all OCR cache entries.

        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match="ocr:*"):
                keys.append(key)
            if keys:
                await self._client.delete(*keys)
            logger.info(f"Cleared {len(keys)} OCR cache entries")
            return len(keys)
        except Exception as e:
            logger.warning(f"Redis clear_ocr_cache error: {e}")
            return 0

    async def clear_ai_cache(self) -> int:
        """Clear all AI cache entries.

        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match="ai:*"):
                keys.append(key)
            if keys:
                await self._client.delete(*keys)
            logger.info(f"Cleared {len(keys)} AI cache entries")
            return len(keys)
        except Exception as e:
            logger.warning(f"Redis clear_ai_cache error: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats (ocr_count, ai_count, connected status)
        """
        stats = {
            "connected": self.is_connected,
            "enabled": self.config.enabled,
            "ocr_count": 0,
            "ai_count": 0,
            "classify_count": 0,
        }

        if not self.is_connected:
            return stats

        try:
            # Count OCR entries
            ocr_count = 0
            async for _ in self._client.scan_iter(match="ocr:*"):
                ocr_count += 1
            stats["ocr_count"] = ocr_count

            # Count AI entries
            ai_count = 0
            async for _ in self._client.scan_iter(match="ai:*"):
                ai_count += 1
            stats["ai_count"] = ai_count

            # Count classification entries
            classify_count = 0
            async for _ in self._client.scan_iter(match="classify:*"):
                classify_count += 1
            stats["classify_count"] = classify_count

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")

        return stats


# Singleton accessor function
async def get_cache_service() -> RedisCacheService:
    """Get the Redis cache service singleton instance.

    Returns:
        RedisCacheService instance (connected or gracefully degraded)
    """
    return await RedisCacheService.get_instance()
