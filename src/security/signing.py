"""
signing.py
==========
Stage: Cross-cutting (Security & Compliance, Sec. 4 — "Model Integrity")

Purpose
-------
Every artifact that crosses a trust boundary in this pipeline (a model
checkpoint, an intermediate forecast, a promoted/publishable forecast) is
signed at write time and verified at read time. This prevents:
  - A tampered model checkpoint from being silently loaded for inference.
  - A forecast artifact being modified in the object store between the
    downscaling stage and the verification stage.
  - An unverified / unsigned artifact from being promoted to the public API.

Implementation
---------------
Uses HMAC-SHA256 keyed by a secret pulled from the SecretsManager
(`signing/artifact-signing-key`). HMAC (symmetric) is used here rather than
asymmetric signing (e.g. RSA/Ed25519) to keep the prototype dependency-free;
the interface (`sign_bytes` / `verify_bytes`) is intentionally the same
shape it would be for an asymmetric scheme, so swapping in a KMS-backed
asymmetric signer later is a drop-in change.

# SWAP_POINT: replace `hmac.new(...)` with a call to a cloud KMS `Sign`/
# `Verify` API (asymmetric, key never leaves the KMS) for production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Dict

from src.security.secrets_manager import SecretsManager, get_default_secrets_manager


class SignatureVerificationError(Exception):
    """Raised when an artifact's signature does not match its content."""


@dataclass
class SignedArtifact:
    payload: Dict[str, Any]
    signature: str
    signer: str

    def to_json(self) -> str:
        return json.dumps({"payload": self.payload, "signature": self.signature, "signer": self.signer})

    @classmethod
    def from_json(cls, raw: str) -> "SignedArtifact":
        d = json.loads(raw)
        return cls(payload=d["payload"], signature=d["signature"], signer=d["signer"])


class ArtifactSigner:
    """Signs and verifies pipeline artifacts (model checkpoints, forecast outputs)."""

    def __init__(self, secrets: SecretsManager = None, signer_id: str = "weather-pipeline"):
        self._secrets = secrets or get_default_secrets_manager()
        self._signer_id = signer_id

    def _key(self) -> bytes:
        return self._secrets.get_secret(
            "secretsmanager://signing/artifact-signing-key", requested_by="ArtifactSigner"
        ).encode("utf-8")

    @staticmethod
    def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
        # Deterministic serialization so the same logical payload always
        # produces the same signature, regardless of dict insertion order.
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def sign(self, payload: Dict[str, Any]) -> SignedArtifact:
        digest = hmac.new(self._key(), self._canonical_bytes(payload), hashlib.sha256).hexdigest()
        return SignedArtifact(payload=payload, signature=digest, signer=self._signer_id)

    def verify(self, artifact: SignedArtifact) -> bool:
        expected = hmac.new(self._key(), self._canonical_bytes(artifact.payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, artifact.signature):
            raise SignatureVerificationError(
                f"Signature mismatch for artifact signed by '{artifact.signer}'. "
                "Refusing to load — possible tampering."
            )
        return True
