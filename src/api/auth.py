"""
auth.py
=======
Stage E: Secured Output API & Visualization — auth layer (architecture doc
Sec. 3-E, "FastAPI with OAuth2/JWT authentication, Rate Limiting")

Purpose
-------
Provides JWT issuance/verification and a simple in-memory sliding-window
rate limiter for the FastAPI app in `api/main.py`.

# SWAP_POINT: in production, token *issuance* happens at the central
# identity provider (Keycloak / AWS Cognito, per architecture doc Sec. 4 —
# "Identity Management"), not inside this service. This service should only
# *verify* tokens issued elsewhere (validate signature against the IdP's
# public key / JWKS endpoint). `issue_token` below exists only so this
# prototype is runnable end-to-end without standing up a real Keycloak
# instance; it is marked dev-only and should not exist in production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import jwt
from fastapi import HTTPException, Request, status

from src.security.secrets_manager import SecretsManager, get_default_secrets_manager

JWT_ALGORITHM = "HS256"
JWT_DEFAULT_TTL_SECONDS = 3600


@dataclass
class TokenClaims:
    sub: str
    scopes: List[str]
    tenant: str


class AuthService:
    def __init__(self, secrets: SecretsManager = None):
        self._secrets = secrets or get_default_secrets_manager()

    def _signing_secret(self) -> str:
        return self._secrets.get_secret("secretsmanager://api/jwt-signing-secret", requested_by="AuthService")

    def issue_token(self, sub: str, scopes: List[str], tenant: str,
                     ttl_seconds: int = JWT_DEFAULT_TTL_SECONDS) -> str:
        """DEV-ONLY token issuance — see module docstring SWAP_POINT."""
        now = int(time.time())
        payload = {
            "sub": sub,
            "scopes": scopes,
            "tenant": tenant,
            "iat": now,
            "exp": now + ttl_seconds,
        }
        return jwt.encode(payload, self._signing_secret(), algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(token, self._signing_secret(), algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return TokenClaims(sub=payload["sub"], scopes=payload.get("scopes", []), tenant=payload.get("tenant", ""))


class RateLimiter:
    """Simple sliding-window rate limiter, keyed by subject (user/service id)."""

    def __init__(self, limit_per_min: int = 120):
        self.limit_per_min = limit_per_min
        self._calls: Dict[str, List[float]] = {}

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60
        calls = [t for t in self._calls.get(key, []) if t >= window_start]
        if len(calls) >= self.limit_per_min:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({self.limit_per_min}/min)",
            )
        calls.append(now)
        self._calls[key] = calls


_auth_service: Optional[AuthService] = None
_rate_limiter: Optional[RateLimiter] = None


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def extract_bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    return header[len("Bearer "):]
