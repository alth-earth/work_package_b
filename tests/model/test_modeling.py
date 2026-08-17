"""Short, opt-in tests for the converted legacy CPU model asset."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

from arctic_route_risk import (
    LegacyCnnOneStepBackend,
    ModelArtifactManifest,
    ModelArtifactPolicy,
    RiskModelInput,
    RuleBaselineBackend,
)
from arctic_route_risk.errors import RiskPipelineError
from arctic_route_risk.modeling.artifacts import _validate_zip_members
from arctic_route_risk.service import _demo_unvalidated_risk

pytestmark = pytest.mark.model

ROOT = Path(__file__).parents[2]
ASSET = ROOT / "models/legacy_cnn_one_step_v1"


def _cnn_input(*, size: tuple[int, int] = (7, 9), value: float = 0.25) -> RiskModelInput:
    ny, nx = size
    latitude = 70.0 + np.arange(ny, dtype=np.float64) * 0.05
    longitude = 30.0 + np.arange(nx, dtype=np.float64) * 0.05
    values = np.full((ny, nx), value, dtype=np.float32)
    return RiskModelInput.from_comprehensive_risk(values, latitude, longitude)


def test_manifest_and_safetensors_are_self_consistent() -> None:
    manifest = ModelArtifactManifest.from_document(
        json.loads((ASSET / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest.parameter_count == 18_849
    assert manifest.time_step_status == "unknown"
    assert manifest.predicted_valid_time is None
    assert manifest.runtime == "cpu_only"
    assert hashlib.sha256((ASSET / "model.safetensors").read_bytes()).hexdigest() == (
        manifest.safetensors_sha256
    )
    tensors = load_file(str(ASSET / "model.safetensors"), device="cpu")
    assert tuple(tensors) == manifest.tensor_keys
    assert sum(tensor.numel() for tensor in tensors.values()) == manifest.parameter_count


def test_cpu_backend_is_deterministic_and_cannot_claim_hourly_output() -> None:
    backend = LegacyCnnOneStepBackend(ASSET)
    model_input = _cnn_input()
    first = backend.infer(model_input)
    second = backend.infer(model_input)
    assert first.risk_score.dtype == np.float32
    assert not first.risk_score.flags.writeable
    assert first.predicted_valid_time is None
    assert first.time_step_status == "unknown"
    assert np.all(np.isfinite(first.risk_score))
    assert np.all((first.risk_score >= 0.0) & (first.risk_score <= 1.0))
    np.testing.assert_array_equal(first.risk_score, second.risk_score)


@pytest.mark.parametrize(
    ("values", "latitude", "longitude", "variables", "expected"),
    [
        (np.full((7, 9), np.nan), None, None, None, "model_input_incompatible"),
        (np.full((7, 9), 1.1), None, None, None, "model_input_incompatible"),
        (np.full((7, 9), 0.2), np.arange(7) * 0.1 + 70, None, None, "model_input_incompatible"),
        (np.full((7, 9), 0.2), None, None, {"extra": np.zeros((7, 9))}, "model_input_incompatible"),
    ],
)
def test_cpu_backend_rejects_invalid_input(
    values: np.ndarray,
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
    variables: dict[str, np.ndarray] | None,
    expected: str,
) -> None:
    backend = LegacyCnnOneStepBackend(ASSET)
    if latitude is None:
        latitude = 70.0 + np.arange(7, dtype=np.float64) * 0.05
    if longitude is None:
        longitude = 30.0 + np.arange(9, dtype=np.float64) * 0.05
    if variables is None:
        variables = {"comprehensive_risk": values}
    model_input = RiskModelInput(variables=variables, latitude=latitude, longitude=longitude)
    with pytest.raises(RiskPipelineError, match=f"^{expected}:"):
        backend.infer(model_input)


def test_cpu_backend_rejects_declared_output_time() -> None:
    backend = LegacyCnnOneStepBackend(ASSET)
    model_input = replace(_cnn_input(), time_step_status="hourly")
    with pytest.raises(RiskPipelineError, match=r"^model_input_incompatible:"):
        backend.infer(model_input)


def test_rule_adapter_matches_existing_rule_array() -> None:
    shape = (2, 3)
    values = {
        "ice_concentration": np.full(shape, 0.4),
        "ice_thickness": np.full(shape, 1.0),
        "ice_type": np.full(shape, 2.0),
        "ice_edge": np.full(shape, 0.2),
        "ice_drift_u": np.full(shape, 0.1),
        "ice_drift_v": np.full(shape, 0.05),
        "significant_wave_height": np.full(shape, 2.0),
        "ocean_current_u": np.full(shape, 0.4),
        "ocean_current_v": np.full(shape, 0.2),
        "wind_u10": np.full(shape, 5.0),
        "wind_v10": np.full(shape, 2.0),
        "air_temperature_2m": np.full(shape, 270.0),
        "visibility": np.full(shape, 8_000.0),
        "sea_surface_height": np.full(shape, 0.3),
        "land_sea_mask": np.ones(shape),
    }
    latitude = np.array([70.0, 70.1])
    longitude = np.array([30.0, 30.1, 30.2])
    model_input = RiskModelInput(variables=values, latitude=latitude, longitude=longitude)
    expected, _, _, _, _ = _demo_unvalidated_risk(
        values,
        source_confidence=1.0,
        model_config=RuleBaselineBackend().model_config,
    )
    RuleBaselineBackend().assert_risk_score_equivalent(model_input, expected)


def test_policy_rejects_wrong_source_hash_before_conversion(tmp_path: Path) -> None:
    policy = replace(ModelArtifactPolicy(), source_zip_sha256="0" * 64)
    fake_zip = tmp_path / "wrong.zip"
    fake_zip.write_bytes(b"not the delivered archive")
    from arctic_route_risk import intake_legacy_cnn_zip

    with pytest.raises(RiskPipelineError, match=r"^model_artifact_invalid:"):
        intake_legacy_cnn_zip(fake_zip, policy=policy, output_dir=tmp_path / "out")


def test_zip_member_gate_rejects_duplicate_and_traversal() -> None:
    policy = ModelArtifactPolicy()
    for names in (
        (policy.member_path, policy.member_path),
        ("../escape.pth", policy.member_path),
    ):
        stream = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(stream, "w") as archive:
                for name in names:
                    archive.writestr(name, b"x")
        stream.seek(0)
        with (
            zipfile.ZipFile(stream) as archive,
            pytest.raises(RiskPipelineError, match=r"^model_artifact_invalid:"),
        ):
            _validate_zip_members(archive, policy)
