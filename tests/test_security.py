import os
import tempfile

import pytest

from src.security.audit_log import AuditIntegrityError, AuditLog
from src.security.secrets_manager import SecretNotFoundError, SecretsManager
from src.security.signing import ArtifactSigner, SignatureVerificationError


def test_secrets_manager_resolves_seeded_dev_secret():
    mgr = SecretsManager()
    value = mgr.get_secret("secretsmanager://ingestion/gfs-api-key", requested_by="test")
    assert isinstance(value, str) and len(value) > 0


def test_secrets_manager_raises_on_unknown_ref():
    mgr = SecretsManager()
    with pytest.raises(SecretNotFoundError):
        mgr.get_secret("secretsmanager://nope/does-not-exist", requested_by="test")


def test_secrets_manager_rejects_malformed_uri():
    mgr = SecretsManager()
    with pytest.raises(ValueError):
        mgr.get_secret("not-a-valid-uri", requested_by="test")


def test_audit_log_append_and_read(tmp_path):
    path = str(tmp_path / "audit.log")
    log = AuditLog(path)
    log.record(actor="tester", action="unit_test", resource="widget", metadata={"k": "v"})
    log.record(actor="tester", action="unit_test_2", resource="widget2")
    records = log.read_all()
    assert len(records) == 2
    assert records[0].action == "unit_test"
    assert records[1].prev_hash == records[0].record_hash


def test_audit_log_integrity_detects_tampering(tmp_path):
    path = str(tmp_path / "audit.log")
    log = AuditLog(path)
    log.record(actor="tester", action="a", resource="r1")
    log.record(actor="tester", action="b", resource="r2")
    assert log.verify_integrity() is True

    # Tamper with the file directly.
    with open(path, "r") as f:
        lines = f.readlines()
    import json
    tampered = json.loads(lines[0])
    tampered["resource"] = "TAMPERED"
    lines[0] = json.dumps(tampered) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)

    with pytest.raises(AuditIntegrityError):
        log.verify_integrity()


def test_artifact_signing_roundtrip():
    signer = ArtifactSigner(signer_id="test-signer")
    artifact = signer.sign({"a": 1, "b": [1, 2, 3]})
    assert signer.verify(artifact) is True


def test_artifact_signing_detects_tampering():
    signer = ArtifactSigner(signer_id="test-signer")
    artifact = signer.sign({"a": 1})
    artifact.payload["a"] = 999  # tamper after signing
    with pytest.raises(SignatureVerificationError):
        signer.verify(artifact)
