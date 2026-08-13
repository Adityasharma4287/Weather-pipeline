"""
secrets_manager.py
===================
Stage: Cross-cutting (Security & Compliance, Sec. 4 of the architecture doc)

Purpose
-------
No component in this pipeline is allowed to hold a raw credential (API key,
DB password, signing key) in code or config. Instead, every component asks
for a credential by *reference* (e.g. "secretsmanager://ingestion/gfs-api-key")
and this module resolves it at call time.

In production this class would be a thin wrapper around AWS Secrets Manager /
HashiCorp Vault / GCP Secret Manager. Swap point is marked below with
SWAP_POINT so it's obvious what to replace when moving off the local dev
backend.

Design notes
------------
- Credentials are never logged. `get_secret` logs *that* a secret was
  accessed (ref, caller, timestamp) via the audit log, but never the value.
- The in-memory backend used here is for local development / this prototype
  only. It is explicitly not suitable for production use.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from src.security.audit_log import AuditLog


class SecretNotFoundError(KeyError):
    """Raised when a secret reference cannot be resolved."""


@dataclass(frozen=True)
class SecretRef:
    """Parsed representation of a `secretsmanager://<namespace>/<name>` URI."""

    namespace: str
    name: str

    @classmethod
    def parse(cls, uri: str) -> "SecretRef":
        prefix = "secretsmanager://"
        if not uri.startswith(prefix):
            raise ValueError(
                f"Invalid secret reference '{uri}'. "
                f"Expected format: '{prefix}<namespace>/<name>'"
            )
        remainder = uri[len(prefix):]
        if "/" not in remainder:
            raise ValueError(f"Invalid secret reference '{uri}': missing '/'")
        namespace, name = remainder.split("/", 1)
        return cls(namespace=namespace, name=name)


class SecretsManager:
    """
    Local-development secrets backend.

    # SWAP_POINT: replace the internal `_store` dict + `_seed_dev_secrets`
    # with real calls to boto3 `secretsmanager.get_secret_value`, or the
    # HashiCorp Vault / GCP Secret Manager SDK, to go to production.
    """

    def __init__(self, audit_log: Optional[AuditLog] = None):
        self._lock = threading.Lock()
        self._store: Dict[str, str] = {}
        self._audit = audit_log or AuditLog()
        self._seed_dev_secrets()

    def _seed_dev_secrets(self) -> None:
        """
        Populate a few deterministic dev-only secrets so the pipeline is
        runnable out of the box. Values are intentionally fake and are
        pulled from environment variables when present, so a real deployment
        can override them without touching code.
        """
        defaults = {
            "ingestion/gfs-api-key": os.environ.get("GFS_API_KEY", "dev-gfs-key-not-real"),
            "ingestion/hrrr-api-key": os.environ.get("HRRR_API_KEY", "dev-hrrr-key-not-real"),
            "ingestion/radar-api-key": os.environ.get("RADAR_API_KEY", "dev-radar-key-not-real"),
            "signing/artifact-signing-key": os.environ.get(
                "ARTIFACT_SIGNING_KEY", "dev-signing-key-do-not-use-in-prod"
            ),
            "api/jwt-signing-secret": os.environ.get(
                "JWT_SIGNING_SECRET", "dev-jwt-secret-do-not-use-in-prod"
            ),
        }
        with self._lock:
            self._store.update(defaults)

    def get_secret(self, ref: str, requested_by: str = "unknown") -> str:
        """
        Resolve a secret reference to its value.

        Parameters
        ----------
        ref: str
            A `secretsmanager://namespace/name` URI.
        requested_by: str
            Identifier of the calling component, for audit purposes.

        Raises
        ------
        SecretNotFoundError if the reference does not resolve.
        """
        parsed = SecretRef.parse(ref)
        key = f"{parsed.namespace}/{parsed.name}"
        with self._lock:
            value = self._store.get(key)

        # Audit the *access*, never the *value*.
        self._audit.record(
            actor=requested_by,
            action="secret_access",
            resource=key,
            metadata={"found": value is not None},
        )

        if value is None:
            raise SecretNotFoundError(f"No secret registered for '{ref}'")
        return value

    def put_secret(self, namespace: str, name: str, value: str, requested_by: str = "unknown") -> None:
        """Register/rotate a secret. Used by tests and bootstrap scripts."""
        key = f"{namespace}/{name}"
        with self._lock:
            self._store[key] = value
        self._audit.record(
            actor=requested_by,
            action="secret_write",
            resource=key,
            metadata={},
        )


# Module-level singleton for convenience across the pipeline; components
# should prefer dependency injection in tests.
_default_manager: Optional[SecretsManager] = None


def get_default_secrets_manager() -> SecretsManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SecretsManager()
    return _default_manager
