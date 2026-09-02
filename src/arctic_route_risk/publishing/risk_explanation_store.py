"""Immutable publication store for B-owned ``risk-explanation.v1`` artifacts.

The sidecar is deliberately stored separately from ``RiskFrame`` commits.  A
consumer may opt into the explanation artifact only after the manifest binds
the exact RiskWindow identity and the content-addressed JSON bytes verify.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from arctic_route_risk.errors import PublicationConflictError

SCHEMA_VERSION = "risk-explanation-manifest.v1"
SIDECAR_SCHEMA_VERSION = "risk-explanation.v1"
_WINDOW_ID = re.compile(r"^risk-window-sha256-[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PublicationConflictError(
                f"immutable ID already has different content: {path}"
            )
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise PublicationConflictError(
                        f"immutable ID already has different content: {path}"
                    ) from None
            _fsync_directory(path.parent)
        finally:
            if temporary_name is not None:
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name)
    except OSError as exc:
        raise PublicationConflictError(f"cannot publish immutable artifact: {path}") from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RiskExplanationArtifactStore:
    """Filesystem-backed, content-addressed sidecar and manifest store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.manifests = self.root / "manifests"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)

    def publish(self, sidecar: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(sidecar, dict) or sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
            raise PublicationConflictError("risk explanation sidecar schema is unsupported")
        identity = sidecar.get("identity")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("risk_window_id"), str)
            or _WINDOW_ID.fullmatch(identity["risk_window_id"]) is None
        ):
            raise PublicationConflictError("risk explanation sidecar identity is missing")
        artifact_bytes = _canonical_bytes(sidecar)
        digest = _sha256_bytes(artifact_bytes)
        artifact_id = f"risk-explanation-sha256-{digest}"
        artifact_name = f"{artifact_id}.json"
        # Manifests live in ``root/manifests`` while immutable payloads live in
        # the sibling ``root/artifacts`` directory.  Keep the manifest
        # self-contained with a relative path, but do not duplicate the large
        # payload under every manifest directory.
        artifact_relative = f"../artifacts/{artifact_name}"
        artifact_path = self.artifacts / artifact_name
        _write_once(artifact_path, artifact_bytes)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PUBLISHED",
            "artifact_id": artifact_id,
            "artifact_sha256": digest,
            "artifact_path": artifact_relative,
            "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
            "identity": identity,
        }
        manifest_path = self.manifests / f"{identity['risk_window_id']}.json"
        _write_once(manifest_path, _canonical_bytes(manifest))
        return {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "artifact_path": artifact_path,
        }

    @staticmethod
    def read(manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest_file = Path(manifest_path)
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationConflictError(
                f"missing or invalid risk explanation manifest: {manifest_file}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("status") != "PUBLISHED"
            or manifest.get("sidecar_schema_version") != SIDECAR_SCHEMA_VERSION
        ):
            raise PublicationConflictError("risk explanation manifest is unsupported")
        relative = manifest.get("artifact_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise PublicationConflictError("risk explanation artifact path is not relative")
        artifact = (manifest_file.parent / relative).resolve()
        store_root = manifest_file.parent.parent.resolve()
        try:
            artifact.relative_to(store_root)
        except ValueError as exc:
            raise PublicationConflictError(
                "risk explanation artifact escapes manifest root"
            ) from exc
        try:
            artifact_bytes = artifact.read_bytes()
            sidecar = json.loads(artifact_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationConflictError(
                f"missing or invalid risk explanation artifact: {artifact}"
            ) from exc
        if _sha256_bytes(artifact_bytes) != manifest.get("artifact_sha256"):
            raise PublicationConflictError("risk explanation artifact digest mismatch")
        artifact_id = manifest.get("artifact_id")
        expected_artifact_id = f"risk-explanation-sha256-{manifest.get('artifact_sha256', '')}"
        if artifact_id != expected_artifact_id or artifact.name != f"{artifact_id}.json":
            raise PublicationConflictError("risk explanation artifact identity mismatch")
        if not isinstance(sidecar, dict) or sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
            raise PublicationConflictError("risk explanation artifact schema mismatch")
        if sidecar.get("identity") != manifest.get("identity"):
            raise PublicationConflictError("risk explanation manifest identity mismatch")
        manifest_identity = manifest.get("identity")
        risk_window_id = (
            manifest_identity.get("risk_window_id")
            if isinstance(manifest_identity, dict)
            else None
        )
        if (
            not isinstance(risk_window_id, str)
            or _WINDOW_ID.fullmatch(risk_window_id) is None
            or manifest_file.name != f"{risk_window_id}.json"
        ):
            raise PublicationConflictError("risk explanation manifest identity is invalid")
        return manifest, sidecar


__all__ = ["SCHEMA_VERSION", "SIDECAR_SCHEMA_VERSION", "RiskExplanationArtifactStore"]
