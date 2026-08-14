"""Safe conversion of the delivered legacy checkpoint to safetensors."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from arctic_route_risk.errors import RiskPipelineError
from arctic_route_risk.modeling.contracts import (
    MODEL_ARTIFACT_INVALID,
    ModelArtifactManifest,
    ModelArtifactPolicy,
    model_error,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_artifact_policy(
    policy: ModelArtifactPolicy | str | Path | Mapping[str, Any] | None,
) -> ModelArtifactPolicy:
    if policy is None:
        return ModelArtifactPolicy()
    if isinstance(policy, ModelArtifactPolicy):
        return policy
    if isinstance(policy, Mapping):
        return ModelArtifactPolicy.from_document(policy)
    candidate = Path(policy)
    if candidate.exists():
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise model_error(MODEL_ARTIFACT_INVALID, f"cannot read policy: {candidate}") from exc
        return ModelArtifactPolicy.from_document(document)
    if str(policy) == "legacy_cnn_one_step_v1":
        return ModelArtifactPolicy()
    raise model_error(MODEL_ARTIFACT_INVALID, f"unknown policy: {policy}")


def _normalise_member(name: str) -> str:
    return name.replace("\\", "/")


def _validate_zip_members(archive: zipfile.ZipFile, policy: ModelArtifactPolicy) -> zipfile.ZipInfo:
    matches: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    for info in archive.infolist():
        normalised = _normalise_member(info.filename).rstrip("/")
        if not normalised:
            continue
        parts = normalised.split("/")
        drive_path = len(parts[0]) == 2 and parts[0][1] == ":"
        if (
            normalised.startswith("/")
            or drive_path
            or ".." in parts
            or "\x00" in normalised
        ):
            raise model_error(MODEL_ARTIFACT_INVALID, f"unsafe ZIP member path: {info.filename!r}")
        if normalised in seen:
            raise model_error(MODEL_ARTIFACT_INVALID, f"duplicate ZIP member: {normalised}")
        seen.add(normalised)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise model_error(
                MODEL_ARTIFACT_INVALID, f"symlink ZIP member is forbidden: {info.filename!r}"
            )
        if normalised == policy.member_path:
            matches.append(info)
    if len(matches) != 1:
        raise model_error(
            MODEL_ARTIFACT_INVALID,
            f"expected exactly one checkpoint member {policy.member_path!r}, found {len(matches)}",
        )
    return matches[0]


def _load_torch_checkpoint(checkpoint: bytes, policy: ModelArtifactPolicy) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise model_error(MODEL_ARTIFACT_INVALID, "model-cpu extra is required for intake") from exc
    try:
        # Explicitly constrained: no pickle-bearing default, and always CPU.
        loaded = torch.load(io.BytesIO(checkpoint), weights_only=True, map_location="cpu")
    except Exception as exc:
        raise model_error(
            MODEL_ARTIFACT_INVALID, "checkpoint failed restricted weights-only CPU load"
        ) from exc
    if not isinstance(loaded, Mapping):
        raise model_error(MODEL_ARTIFACT_INVALID, "checkpoint outer value is not a mapping")
    if set(loaded) != set(policy.expected_outer_keys):
        raise model_error(MODEL_ARTIFACT_INVALID, "checkpoint outer keys differ from policy")
    if loaded.get("in_channels") != 1 or loaded.get("hidden_channels") != 32:
        raise model_error(
            MODEL_ARTIFACT_INVALID, "checkpoint architecture metadata differs from policy"
        )
    if loaded.get("target") != policy.expected_target:
        raise model_error(MODEL_ARTIFACT_INVALID, "checkpoint target differs from policy")
    state = loaded.get("model_state_dict")
    if not isinstance(state, Mapping) or set(state) != set(policy.expected_state_keys):
        raise model_error(MODEL_ARTIFACT_INVALID, "checkpoint tensor keys differ from policy")
    for key in policy.expected_state_keys:
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise model_error(MODEL_ARTIFACT_INVALID, f"checkpoint value is not a tensor: {key}")
        if tuple(tensor.shape) != tuple(policy.expected_state_shapes[key]):
            raise model_error(MODEL_ARTIFACT_INVALID, f"checkpoint shape differs for {key}")
        if tensor.dtype != torch.float32:
            raise model_error(MODEL_ARTIFACT_INVALID, f"checkpoint dtype differs for {key}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise model_error(
                MODEL_ARTIFACT_INVALID, f"checkpoint contains non-finite values: {key}"
            )
    return {
        key: state[key].detach().to(device="cpu").contiguous() for key in policy.expected_state_keys
    }


def _manifest_for(
    *,
    policy: ModelArtifactPolicy,
    zip_sha256: str,
    checkpoint_sha256: str,
    safetensors_sha256: str,
    parameter_count: int,
) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        manifest_version="b.model-artifact-manifest.v1",
        artifact_id="legacy-cnn-one-step-v1",
        policy_id=policy.policy_id,
        backend_id="legacy_cnn_one_step",
        model_version="legacy_cnn_one_step_v1",
        source_zip_sha256=zip_sha256,
        source_checkpoint_sha256=checkpoint_sha256,
        source_member=policy.member_path,
        safetensors_sha256=safetensors_sha256,
        tensor_keys=policy.expected_state_keys,
        tensor_shapes=policy.expected_state_shapes,
        dtype="float32",
        parameter_count=parameter_count,
        native_grid_step_degrees=policy.native_grid_step_degrees,
        predicted_valid_time=None,
        time_step_status="unknown",
        calibration_status="experimental_unverified",
        runtime="cpu_only",
        upstream_license="not_provided",
        distribution_authorized_on="2026-08-14",
    )


def _notice(manifest: ModelArtifactManifest) -> str:
    return f"""# Legacy CNN asset notice

This directory contains a converted, pickle-free copy of the user-delivered
legacy one-step CNN checkpoint. It is an experimental, CPU-only shadow asset;
it is not the formal B rule backend and its cadence is intentionally unknown.

- Converted artifact: `model.safetensors`
- Safetensors SHA-256: `{manifest.safetensors_sha256}`
- Source ZIP SHA-256: `{manifest.source_zip_sha256}`
- Source checkpoint SHA-256: `{manifest.source_checkpoint_sha256}`
- Source member: `{manifest.source_member}`
- Conversion: `torch.load(weights_only=True, map_location="cpu")` followed by
  safetensors serialization and bitwise re-load verification.
- Upstream license: not provided with the delivery.
- Public redistribution: explicitly authorized by the user on 2026-08-14 for
  this public Work Package B repository.

The original ZIP, checkpoint, training data, scripts and bytecode are not
included in the repository. Do not infer an hourly forecast, calibration,
confidence, hard mask, route, or navigation guarantee from this asset.
"""


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _existing_manifest(output: Path) -> ModelArtifactManifest | None:
    manifest_path = output / "manifest.json"
    model_path = output / "model.safetensors"
    notice_path = output / "MODEL_ASSET_NOTICE.md"
    if not output.exists():
        return None
    if not output.is_dir() or not (
        manifest_path.exists() and model_path.exists() and notice_path.exists()
    ):
        raise model_error(MODEL_ARTIFACT_INVALID, f"output exists but is incomplete: {output}")
    try:
        manifest = ModelArtifactManifest.from_document(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, RiskPipelineError) as exc:
        raise model_error(
            MODEL_ARTIFACT_INVALID, f"existing manifest is invalid: {output}"
        ) from exc
    if _sha256_file(model_path) != manifest.safetensors_sha256:
        raise model_error(
            MODEL_ARTIFACT_INVALID, "existing safetensors hash does not match manifest"
        )
    return manifest


def intake_legacy_cnn_zip(
    zip_path: str | Path,
    policy: ModelArtifactPolicy | str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path = "models/legacy_cnn_one_step_v1",
) -> ModelArtifactManifest:
    """Convert the exact delivered checkpoint without extracting or executing it."""

    source = Path(zip_path)
    output = Path(output_dir)
    resolved_policy = resolve_model_artifact_policy(policy)
    if not source.is_file():
        raise model_error(MODEL_ARTIFACT_INVALID, f"ZIP does not exist: {source}")
    zip_digest = _sha256_file(source)
    if zip_digest != resolved_policy.source_zip_sha256:
        raise model_error(MODEL_ARTIFACT_INVALID, "source ZIP SHA-256 does not match policy")
    existing = _existing_manifest(output)
    if existing is not None:
        if (
            existing.source_zip_sha256 != zip_digest
            or existing.policy_id != resolved_policy.policy_id
        ):
            raise model_error(MODEL_ARTIFACT_INVALID, "output contains a different artifact")
        return existing
    try:
        with zipfile.ZipFile(source, "r") as archive:
            member = _validate_zip_members(archive, resolved_policy)
            checkpoint = archive.read(member)
    except zipfile.BadZipFile as exc:
        raise model_error(MODEL_ARTIFACT_INVALID, "source is not a valid ZIP") from exc
    checkpoint_digest = _sha256_bytes(checkpoint)
    if checkpoint_digest != resolved_policy.source_checkpoint_sha256:
        raise model_error(MODEL_ARTIFACT_INVALID, "checkpoint SHA-256 does not match policy")
    state = _load_torch_checkpoint(checkpoint, resolved_policy)
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise model_error(
            MODEL_ARTIFACT_INVALID, "model-cpu extra is required for conversion"
        ) from exc
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    metadata = {
        "format": "safetensors",
        "artifact_id": "legacy-cnn-one-step-v1",
        "source_checkpoint_sha256": checkpoint_digest,
        "time_step_status": "unknown",
    }
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        tensor_path = temp_dir / "model.safetensors"
        save_file(state, str(tensor_path), metadata=metadata)
        reloaded = load_file(str(tensor_path), device="cpu")
        if set(reloaded) != set(resolved_policy.expected_state_keys):
            raise model_error(MODEL_ARTIFACT_INVALID, "safetensors keys changed during conversion")
        for key in resolved_policy.expected_state_keys:
            if not torch.equal(state[key], reloaded[key]):
                raise model_error(
                    MODEL_ARTIFACT_INVALID, f"tensor changed during conversion: {key}"
                )
        tensor_digest = _sha256_file(tensor_path)
        manifest = _manifest_for(
            policy=resolved_policy,
            zip_sha256=zip_digest,
            checkpoint_sha256=checkpoint_digest,
            safetensors_sha256=tensor_digest,
            parameter_count=parameter_count,
        )
        _write_json(temp_dir / "manifest.json", manifest.to_document())
        (temp_dir / "MODEL_ASSET_NOTICE.md").write_text(_notice(manifest), encoding="utf-8")
        try:
            os.replace(temp_dir, output)
        except FileExistsError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            existing = _existing_manifest(output)
            if existing is None or existing.to_document() != manifest.to_document():
                raise model_error(
                    MODEL_ARTIFACT_INVALID, "output appeared with different content"
                ) from None
            return existing
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
