from __future__ import annotations

import json
from pathlib import Path

import pytest

from arctic_route_risk import RiskExplanationArtifactStore
from arctic_route_risk.errors import PublicationConflictError


def _sidecar() -> dict:
    return {
        "schema_version": "risk-explanation.v1",
        "publication_status": "UNAVAILABLE",
        "identity": {"risk_window_id": "risk-window-sha256-" + "a" * 64},
    }


def test_sidecar_store_publishes_content_addressed_artifact_and_manifest(tmp_path: Path) -> None:
    store = RiskExplanationArtifactStore(tmp_path / "explanation")
    publication = store.publish(_sidecar())

    manifest, sidecar = store.read(publication["manifest_path"])
    assert manifest["status"] == "PUBLISHED"
    assert manifest["artifact_id"].startswith("risk-explanation-sha256-")
    assert sidecar == _sidecar()
    assert publication["artifact_path"].name == f"{manifest['artifact_id']}.json"

    second = store.publish(_sidecar())
    assert second["artifact_path"] == publication["artifact_path"]
    assert second["manifest_path"] == publication["manifest_path"]


def test_sidecar_store_fails_closed_on_tampered_bytes(tmp_path: Path) -> None:
    store = RiskExplanationArtifactStore(tmp_path / "explanation")
    publication = store.publish(_sidecar())
    artifact = publication["artifact_path"]
    artifact.write_text(
        json.dumps({**_sidecar(), "tampered": True}),
        encoding="utf-8",
    )
    with pytest.raises(PublicationConflictError, match="digest mismatch"):
        store.read(publication["manifest_path"])


def test_sidecar_store_rejects_identity_collision(tmp_path: Path) -> None:
    store = RiskExplanationArtifactStore(tmp_path / "explanation")
    store.publish(_sidecar())
    changed = _sidecar()
    changed["publication_status"] = "PARTIAL"
    with pytest.raises(
        PublicationConflictError, match="immutable ID already has different content"
    ):
        store.publish(changed)
