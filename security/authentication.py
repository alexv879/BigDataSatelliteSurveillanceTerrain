"""
Authentication and Authorization
API key authentication, JWT tokens, and rate limiting
"""
import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger


# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class APIKeyAuth:
    """
    API Key Authentication
    Provides simple API key-based authentication
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize API key authentication

        Args:
            api_keys: Dictionary of api_key -> user_id
        """
        self.api_keys = api_keys or {}
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def generate_api_key(self, user_id: str) -> str:
        """
        Generate a new API key

        Args:
            user_id: User identifier

        Returns:
            New API key
        """
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = user_id
        logger.info(f"Generated API key for user: {user_id}")
        return api_key

    def revoke_api_key(self, api_key: str):
        """
        Revoke an API key

        Args:
            api_key: API key to revoke
        """
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            del self.api_keys[api_key]
            logger.info(f"Revoked API key for user: {user_id}")

    async def validate(self, api_key: str = Security(APIKeyHeader(name="X-API-Key"))):
        """
        Validate API key

        Args:
            api_key: API key from header

        Returns:
            User ID if valid

        Raises:
            HTTPException: If API key invalid
        """
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )

        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )

        return self.api_keys[api_key]


class JWTAuth:
    """
    JWT Token Authentication
    Provides JWT-based authentication with refresh tokens
    """

    def __init__(self, secret_key: str = SECRET_KEY):
        """
        Initialize JWT authentication

        Args:
            secret_key: Secret key for signing tokens
        """
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()

    def hash_password(self, password: str) -> str:
        """
        Hash a password

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password to compare

        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            data: Data to encode in token
            expires_delta: Optional expiration time delta

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)

        return encoded_jwt

    async def validate(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> dict:
        """
        Validate JWT token

        Args:
            credentials: HTTP authorization credentials

        Returns:
            Decoded token data

        Raises:
            HTTPException: If token invalid
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization required"
            )

        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[ALGORITHM]
            )
            return payload

        except JWTError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or expired token"
            )


class RateLimiter:
    """
    Rate Limiter
    Prevents abuse by limiting request rates
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            requests_per_day: Max requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day

        # Storage: {user_id: {window: [timestamps]}}
        self.requests: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {
                'minute': [],
                'hour': [],
                'day': []
            }
        )

    def _clean_old_requests(self, user_id: str):
        """
        Remove old request timestamps

        Args:
            user_id: User identifier
        """
        now = time.time()

        # Clean minute window (last 60 seconds)
        self.requests[user_id]['minute'] = [
            ts for ts in self.requests[user_id]['minute']
            if now - ts < 60
        ]

        # Clean hour window (last 3600 seconds)
        self.requests[user_id]['hour'] = [
            ts for ts in self.requests[user_id]['hour']
            if now - ts < 3600
        ]

        # Clean day window (last 86400 seconds)
        self.requests[user_id]['day'] = [
            ts for ts in self.requests[user_id]['day']
            if now - ts < 86400
        ]

    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limits

        Args:
            user_id: User identifier

        Returns:
            True if within limits, False if exceeded

        Raises:
            HTTPException: If rate limit exceeded
        """
        self._clean_old_requests(user_id)

        minute_count = len(self.requests[user_id]['minute'])
        hour_count = len(self.requests[user_id]['hour'])
        day_count = len(self.requests[user_id]['day'])

        if minute_count >= self.requests_per_minute:
            logger.warning(
                f"User {user_id} exceeded per-minute rate limit "
                f"({minute_count}/{self.requests_per_minute})"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: max {self.requests_per_minute} requests per minute"
            )

        if hour_count >= self.requests_per_hour:
            logger.warning(
                f"User {user_id} exceeded per-hour rate limit "
                f"({hour_count}/{self.requests_per_hour})"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: max {self.requests_per_hour} requests per hour"
            )

        if day_count >= self.requests_per_day:
            logger.warning(
                f"User {user_id} exceeded per-day rate limit "
                f"({day_count}/{self.requests_per_day})"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: max {self.requests_per_day} requests per day"
            )

        # Record this request
        now = time.time()
        self.requests[user_id]['minute'].append(now)
        self.requests[user_id]['hour'].append(now)
        self.requests[user_id]['day'].append(now)

        return True

    def get_rate_limit_status(self, user_id: str) -> dict:
        """
        Get current rate limit status for user

        Args:
            user_id: User identifier

        Returns:
            Dictionary with current usage
        """
        self._clean_old_requests(user_id)

        return {
            'minute': {
                'used': len(self.requests[user_id]['minute']),
                'limit': self.requests_per_minute,
                'remaining': max(0, self.requests_per_minute - len(self.requests[user_id]['minute']))
            },
            'hour': {
                'used': len(self.requests[user_id]['hour']),
                'limit': self.requests_per_hour,
                'remaining': max(0, self.requests_per_hour - len(self.requests[user_id]['hour']))
            },
            'day': {
                'used': len(self.requests[user_id]['day']),
                'limit': self.requests_per_day,
                'remaining': max(0, self.requests_per_day - len(self.requests[user_id]['day']))
            }
        }


def rate_limit(limiter: RateLimiter):
    """
    Decorator for rate limiting

    Args:
        limiter: RateLimiter instance

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user_id: str = "anonymous", **kwargs):
            limiter.check_rate_limit(user_id)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
