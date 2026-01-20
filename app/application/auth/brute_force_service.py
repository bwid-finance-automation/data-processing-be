"""Brute force protection service using Redis."""

from typing import Optional

from app.infrastructure.cache.redis_cache import RedisCacheService, get_cache_service
from app.core.unified_config import get_unified_config
from app.shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class AccountLockedError(Exception):
    """Raised when account is locked due to too many failed attempts."""

    def __init__(self, message: str, remaining_seconds: int = 0):
        self.message = message
        self.remaining_seconds = remaining_seconds
        super().__init__(message)


class BruteForceProtectionService:
    """
    Service for tracking and preventing brute force login attacks.

    Uses Redis to store failed attempt counts with auto-expiration.
    Key pattern: login_failed:{identifier}
    """

    KEY_PREFIX = "login_failed:"

    def __init__(self):
        self._cache: Optional[RedisCacheService] = None

    @property
    def config(self):
        """Get brute force config from unified config."""
        return get_unified_config().brute_force

    async def _get_cache(self) -> RedisCacheService:
        """Get or initialize cache service."""
        if self._cache is None:
            self._cache = await get_cache_service()
        return self._cache

    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for the identifier (email/username)."""
        return f"{self.KEY_PREFIX}{identifier.lower()}"

    async def is_locked(self, identifier: str) -> bool:
        """
        Check if an account is currently locked out.

        Args:
            identifier: Username or email to check

        Returns:
            True if account is locked, False otherwise
        """
        if not self.config.enabled:
            return False

        cache = await self._get_cache()
        if not cache.is_connected:
            # Graceful degradation: allow login if Redis is unavailable
            logger.warning("Redis unavailable, skipping brute force check")
            return False

        key = self._get_key(identifier)
        try:
            data = await cache._client.get(key)
            if data:
                attempts = int(data)
                return attempts >= self.config.max_failed_attempts
        except Exception as e:
            logger.warning(f"Error checking brute force lock: {e}")

        return False

    async def get_remaining_lockout_time(self, identifier: str) -> Optional[int]:
        """
        Get remaining lockout time in seconds.

        Args:
            identifier: Username or email

        Returns:
            Remaining seconds if locked, None if not locked
        """
        if not self.config.enabled:
            return None

        cache = await self._get_cache()
        if not cache.is_connected:
            return None

        key = self._get_key(identifier)
        try:
            ttl = await cache._client.ttl(key)
            if ttl > 0:
                data = await cache._client.get(key)
                if data and int(data) >= self.config.max_failed_attempts:
                    return ttl
        except Exception as e:
            logger.warning(f"Error getting lockout time: {e}")

        return None

    async def record_failed_attempt(self, identifier: str) -> int:
        """
        Record a failed login attempt.

        Args:
            identifier: Username or email

        Returns:
            Current number of failed attempts
        """
        if not self.config.enabled:
            return 0

        cache = await self._get_cache()
        if not cache.is_connected:
            logger.warning("Redis unavailable, cannot record failed attempt")
            return 0

        key = self._get_key(identifier)
        lockout_seconds = self.config.lockout_duration_minutes * 60

        try:
            # Increment counter and set/extend TTL
            pipe = cache._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, lockout_seconds)
            results = await pipe.execute()

            attempts = results[0]

            if attempts >= self.config.max_failed_attempts:
                logger.warning(
                    f"Account locked due to brute force: {identifier} "
                    f"({attempts} attempts, locked for {self.config.lockout_duration_minutes} minutes)"
                )
            else:
                logger.info(
                    f"Failed login attempt for {identifier}: "
                    f"{attempts}/{self.config.max_failed_attempts}"
                )

            return attempts

        except Exception as e:
            logger.warning(f"Error recording failed attempt: {e}")
            return 0

    async def reset_attempts(self, identifier: str) -> bool:
        """
        Reset failed attempts counter after successful login.

        Args:
            identifier: Username or email

        Returns:
            True if reset successful, False otherwise
        """
        if not self.config.enabled:
            return True

        cache = await self._get_cache()
        if not cache.is_connected:
            return False

        key = self._get_key(identifier)
        try:
            await cache._client.delete(key)
            logger.debug(f"Reset brute force counter for {identifier}")
            return True
        except Exception as e:
            logger.warning(f"Error resetting attempts: {e}")
            return False

    async def get_attempts(self, identifier: str) -> int:
        """
        Get current number of failed attempts.

        Args:
            identifier: Username or email

        Returns:
            Number of failed attempts
        """
        cache = await self._get_cache()
        if not cache.is_connected:
            return 0

        key = self._get_key(identifier)
        try:
            data = await cache._client.get(key)
            return int(data) if data else 0
        except Exception as e:
            logger.warning(f"Error getting attempts: {e}")
            return 0


# Singleton accessor
_brute_force_service: Optional[BruteForceProtectionService] = None


async def get_brute_force_service() -> BruteForceProtectionService:
    """Get brute force protection service singleton."""
    global _brute_force_service
    if _brute_force_service is None:
        _brute_force_service = BruteForceProtectionService()
    return _brute_force_service
