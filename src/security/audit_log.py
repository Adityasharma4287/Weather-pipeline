"""
audit_log.py
============
Stage: Cross-cutting (Security & Compliance, Sec. 4 — "Logging & Auditing")

Purpose
-------
Every pipeline stage transition (data fetch, model run, artifact promotion,
API read) must be recorded in an append-only, immutable log with actor,
timestamp, resource, and action, so an incident can be reconstructed without
relying on mutable application logs.

This implementation appends newline-delimited JSON records to a local file
opened in append-only mode, and additionally computes a running hash chain
(`prev_hash` -> `record_hash`) so any tampering with a historical entry is
detectable by re-walking the chain. This is the same idea used by
write-once/append-only object storage in production (e.g. S3 Object Lock),
implemented locally so the pipeline is fully runnable without cloud
credentials.

# SWAP_POINT: in production, `_append_line` should write to a WORM
# (write-once-read-many) object store / dedicated audit-log service instead
# of a local file, and hash-chain verification should run as a periodic
# background job.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "/tmp/weather_pipeline_audit.log")


@dataclass
class AuditRecord:
    timestamp: str
    actor: str
    action: str
    resource: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    record_hash: str = ""


class AuditIntegrityError(Exception):
    """Raised when the audit log hash chain does not validate."""


class AuditLog:
    """Append-only, hash-chained audit log."""

    def __init__(self, path: str = DEFAULT_AUDIT_LOG_PATH):
        self._path = path
        self._lock = threading.Lock()
        # Ensure the file exists so reads never fail on a fresh checkout.
        if not os.path.exists(self._path):
            open(self._path, "a").close()

    @staticmethod
    def _hash_record(prev_hash: str, timestamp: str, actor: str, action: str,
                      resource: str, metadata: Dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "prev_hash": prev_hash,
                "timestamp": timestamp,
                "actor": actor,
                "action": action,
                "resource": resource,
                "metadata": metadata,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _last_hash(self) -> str:
        last = ""
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = json.loads(line)["record_hash"]
        return last

    def record(self, actor: str, action: str, resource: str,
               metadata: Optional[Dict[str, Any]] = None) -> AuditRecord:
        """Append a new, hash-chained audit record. Thread-safe."""
        metadata = metadata or {}
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            prev_hash = self._last_hash()
            record_hash = self._hash_record(prev_hash, timestamp, actor, action, resource, metadata)
            record = AuditRecord(
                timestamp=timestamp,
                actor=actor,
                action=action,
                resource=resource,
                metadata=metadata,
                prev_hash=prev_hash,
                record_hash=record_hash,
            )
            with open(self._path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        return record

    def read_all(self) -> List[AuditRecord]:
        records = []
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(AuditRecord(**json.loads(line)))
        return records

    def verify_integrity(self) -> bool:
        """
        Re-walk the hash chain. Returns True if intact, raises
        AuditIntegrityError with details if any record was tampered with
        or removed.
        """
        prev_hash = ""
        for i, rec in enumerate(self.read_all()):
            expected = self._hash_record(
                prev_hash, rec.timestamp, rec.actor, rec.action, rec.resource, rec.metadata
            )
            if rec.prev_hash != prev_hash or rec.record_hash != expected:
                raise AuditIntegrityError(
                    f"Audit log tampering detected at record #{i} (resource={rec.resource})"
                )
            prev_hash = rec.record_hash
        return True
