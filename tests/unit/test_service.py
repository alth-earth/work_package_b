from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from arctic_route_planning.contracts import validate_canonical_risk_id

from arctic_route_risk import (
    BInputEnvelope,
    GridCompatibilityError,
    RiskBuildRequest,
    RiskBuildService,
    RiskPipelineError,
    TargetGridConfig,
    load_risk_build_configuration,
    model_config_digest,
)
from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.service import (
    _demo_unvalidated_risk,
    _polar_stereographic_xy,
    _regrid_variable,
)

GENERATED = datetime(2026, 8, 2, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).parents[2]


def _frames(formal_fixture):
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    request = RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)
    return RiskBuildService(utc_now=lambda: GENERATED).build_window(request)


def test_builds_complete_hourly_c_contract_window(formal_fixture) -> None:
    frames = _frames(formal_fixture)

    assert len(frames) == 3
    assert frames[0].valid_time == formal_fixture.context.simulation_start
    assert frames[-1].valid_time == formal_fixture.context.simulation_end
    assert all(
        (right.valid_time - left.valid_time).total_seconds() == 3600
        for left, right in pairwise(frames)
    )
    for frame in frames:
        validate_canonical_risk_id(frame)
        assert frame.generated_at == GENERATED
        assert frame.payload.attrs["calibration_status"] == "demo_unvalidated"
        assert frame.payload["hard_mask"].dtype == np.bool_
        assert float(frame.payload["environment_speed_factor"].min()) > 0
        risk = frame.payload["risk_score"].values
        level = frame.payload["risk_level"].values
        finite = np.isfinite(risk)
        np.testing.assert_array_equal(
            level[finite],
            np.clip(np.floor(risk[finite] * 5) + 1, 1, 5).astype(np.uint8),
        )
        assert all(source.issue_time <= frame.as_of_time for source in frame.source_summary)


def test_land_sea_mask_uses_nearest_spatial_alignment(formal_fixture) -> None:
    source = formal_fixture.prepared.frames["land_sea_mask"][0]
    latitude = np.linspace(70.0, 72.0, 9)
    longitude = np.linspace(10.0, 12.0, 9)

    values = _regrid_variable(
        source,
        "land_sea_mask",
        latitude,
        longitude,
        categorical=True,
    )

    assert set(np.unique(values)) == {0.0, 1.0}


def test_polar_curvilinear_grid_regrids_continuous_field(formal_fixture) -> None:
    frame, _, _ = _curvilinear_frame(formal_fixture)
    target_lat = np.array([74.90, 74.95])
    target_lon = np.array([-10.20, -10.00])

    values = _regrid_variable(
        frame,
        "wind_u10",
        target_lat,
        target_lon,
        categorical=False,
    )

    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
    target_x, target_y = _polar_stereographic_xy(
        longitude_degrees=target_lon_grid,
        latitude_degrees=target_lat_grid,
        radius_metres=6_378_273.0,
        central_meridian_degrees=-45.0,
    )
    expected = 2.0 * target_y + 3.0 * target_x
    np.testing.assert_allclose(values, expected, atol=1e-10)


def test_polar_curvilinear_grid_uses_nearest_for_categorical(formal_fixture) -> None:
    frame, _, _ = _curvilinear_frame(
        formal_fixture,
        field=np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
    )
    target_lat = np.array([74.90, 74.95])
    target_lon = np.array([-10.20, -10.00])

    values = _regrid_variable(
        frame,
        "wind_u10",
        target_lat,
        target_lon,
        categorical=True,
    )

    np.testing.assert_array_equal(values, np.array([[1.0, 12.0], [10.0, 21.0]]))


def test_polar_curvilinear_grid_rejects_unknown_projection(formal_fixture) -> None:
    frame, _, _ = _curvilinear_frame(
        formal_fixture,
        attrs={"query_projection": "+proj=merc +lon_0=-45 +R=6378273"},
    )
    with pytest.raises(GridCompatibilityError, match="unsupported curvilinear projection"):
        _regrid_variable(
            frame,
            "wind_u10",
            np.array([75.0]),
            np.array([-10.0]),
            categorical=False,
        )


def test_model_digest_is_policy_only_not_realized_corridor_grid() -> None:
    grid = TargetGridConfig(latitude_step_degrees=0.5, longitude_step_degrees=0.5)
    model = DemoRiskModelConfig()
    first = model_config_digest(grid=grid, model=model)
    grid.realize((10.0, 70.0, 12.0, 72.0))
    second = model_config_digest(grid=grid, model=model)
    grid.realize((30.0, 69.0, 80.0, 75.0))

    assert first == second == model_config_digest(grid=grid, model=model)


def test_versioned_default_configuration_is_the_runtime_policy() -> None:
    configuration = load_risk_build_configuration(
        PROJECT_ROOT / "configs/models/demo_unvalidated_v2.json"
    )

    assert configuration.grid_config == TargetGridConfig()
    assert configuration.model_config == DemoRiskModelConfig()
    assert configuration.model_config_digest == model_config_digest(
        grid=TargetGridConfig(),
        model=DemoRiskModelConfig(),
    )


def test_v1_configuration_is_audit_only_and_not_implicitly_migrated() -> None:
    with pytest.raises(RiskPipelineError, match="fields differ"):
        load_risk_build_configuration(
            PROJECT_ROOT / "configs/models/demo_unvalidated_v1.json"
        )


def test_v2_default_preserves_v1_baseline_arrays() -> None:
    values = {
        "ice_concentration": np.array([[0.4, np.nan], [1.2, 0.0]]),
        "ice_thickness": np.array([[1.0, 2.0], [4.0, 0.0]]),
        "ice_type": np.array([[2.0, 3.0], [4.0, 0.0]]),
        "ice_edge": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "ice_drift_u": np.array([[0.1, 0.2], [0.3, 0.0]]),
        "ice_drift_v": np.array([[0.05, 0.1], [0.2, 0.0]]),
        "significant_wave_height": np.array([[2.0, 4.0], [9.0, 0.0]]),
        "ocean_current_u": np.array([[0.4, 0.8], [2.2, 0.0]]),
        "ocean_current_v": np.array([[0.2, 0.4], [0.0, 0.0]]),
        "wind_u10": np.array([[5.0, 10.0], [31.0, 0.0]]),
        "wind_v10": np.array([[2.0, 4.0], [0.0, 0.0]]),
        "air_temperature_2m": np.array([[270.0, 260.0], [240.0, 280.0]]),
        "visibility": np.array([[8_000.0, 4_000.0], [0.0, 12_000.0]]),
        "sea_surface_height": np.array([[0.3, -1.0], [4.0, 0.0]]),
        "land_sea_mask": np.array([[1.0, 1.0], [0.0, np.nan]]),
    }
    components = (
        (0.24, _legacy_bounded(values["ice_concentration"], 0.0, 1.0)),
        (0.14, _legacy_bounded(values["ice_thickness"], 0.0, 3.0)),
        (0.05, _legacy_bounded(values["ice_type"], 0.0, 4.0)),
        (0.02, _legacy_bounded(values["ice_edge"], 0.0, 1.0)),
        (
            0.06,
            _legacy_bounded(np.hypot(values["ice_drift_u"], values["ice_drift_v"]), 0, 1.5),
        ),
        (0.13, _legacy_bounded(values["significant_wave_height"], 0.0, 8.0)),
        (
            0.07,
            _legacy_bounded(
                np.hypot(values["ocean_current_u"], values["ocean_current_v"]), 0, 2.0
            ),
        ),
        (
            0.10,
            _legacy_bounded(np.hypot(values["wind_u10"], values["wind_v10"]), 0, 30.0),
        ),
        (0.05, _legacy_bounded(273.15 - values["air_temperature_2m"], 0, 30.0)),
        (0.10, _legacy_bounded(10_000.0 - values["visibility"], 0, 10_000.0)),
        (0.04, _legacy_bounded(np.abs(values["sea_surface_height"]), 0, 3.0)),
    )
    expected_risk = sum(weight * component for weight, component in components)
    expected_valid = np.logical_and.reduce(
        [np.isfinite(component) for _, component in components]
    )
    expected_hard = ~np.isfinite(values["land_sea_mask"]) | (
        values["land_sea_mask"] < 0.5
    )
    expected_valid &= np.isfinite(values["land_sea_mask"])
    expected_risk = np.where(expected_valid, np.clip(expected_risk, 0.0, 1.0), np.nan)
    expected_confidence = np.where(expected_valid, 0.75, 0.0)
    expected_speed = np.where(
        expected_valid,
        np.clip(1.0 - 0.55 * expected_risk, 0.35, 1.0),
        0.35,
    )

    actual = _demo_unvalidated_risk(
        values,
        source_confidence=0.75,
        model_config=DemoRiskModelConfig(),
    )

    for expected, observed in zip(
        (expected_risk, expected_hard, expected_confidence, expected_speed),
        actual[:4],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(expected).astype(np.float32),
            np.asarray(observed).astype(np.float32),
        )
    expected_reason = np.where(
        expected_hard,
        "LAND",
        "NONE",
    ).astype("U32")
    np.testing.assert_array_equal(actual[4], expected_reason)


def test_v2_runtime_uses_configured_confidence_mask_and_speed_policies(
    formal_fixture,
) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    baseline = DemoRiskModelConfig()
    configured = replace(
        baseline,
        quality_confidence=replace(baseline.quality_confidence, good=0.8),
        temporal_method_confidence=replace(
            baseline.temporal_method_confidence,
            linear_interpolation=0.4,
        ),
        land_sea_mask_land_threshold=0.0,
        speed_risk_coefficient=0.2,
    )
    frames = RiskBuildService(utc_now=lambda: GENERATED).build_window(
        RiskBuildRequest(
            envelope=envelope,
            target_bbox=formal_fixture.bbox,
            model_config=configured,
        )
    )

    assert np.isclose(float(frames[1].payload["confidence"].max()), 0.32)
    assert not bool(frames[1].payload["hard_mask"].any())
    expected_speed = np.clip(
        1.0 - 0.2 * frames[1].payload["risk_score"].values,
        configured.minimum_speed_factor,
        1.0,
    )
    np.testing.assert_allclose(
        frames[1].payload["environment_speed_factor"].values,
        expected_speed,
    )


def test_each_configured_numeric_policy_changes_model_digest() -> None:
    grid = TargetGridConfig()
    baseline = DemoRiskModelConfig()
    baseline_digest = model_config_digest(grid=grid, model=baseline)
    variants: list[DemoRiskModelConfig] = []
    for index, component in enumerate(baseline.components):
        peer_index = 1 if index == 0 else 0
        adjusted = list(baseline.components)
        adjusted[index] = replace(component, weight=component.weight + 0.001)
        peer = adjusted[peer_index]
        adjusted[peer_index] = replace(peer, weight=peer.weight - 0.001)
        variants.append(replace(baseline, components=tuple(adjusted)))
        adjusted = list(baseline.components)
        adjusted[index] = replace(component, lower=component.lower - 0.01)
        variants.append(replace(baseline, components=tuple(adjusted)))
        adjusted = list(baseline.components)
        adjusted[index] = replace(component, upper=component.upper + 0.01)
        variants.append(replace(baseline, components=tuple(adjusted)))
    for field, value in (
        ("good", 0.99),
        ("suspect", 0.74),
        ("degraded", 0.49),
    ):
        variants.append(
            replace(
                baseline,
                quality_confidence=replace(
                    baseline.quality_confidence,
                    **{field: value},
                ),
            )
        )
    for field, value in (
        ("exact", 0.99),
        ("categorical_nearest", 0.84),
        ("linear_interpolation", 0.89),
        ("static", 0.99),
    ):
        variants.append(
            replace(
                baseline,
                temporal_method_confidence=replace(
                    baseline.temporal_method_confidence,
                    **{field: value},
                ),
            )
        )
    variants.extend(
        (
            replace(baseline, land_sea_mask_land_threshold=0.51),
            replace(baseline, speed_risk_coefficient=0.56),
            replace(baseline, minimum_speed_factor=0.36),
        )
    )

    assert all(
        model_config_digest(grid=grid, model=variant) != baseline_digest
        for variant in variants
    )
    assert len({model_config_digest(grid=grid, model=item) for item in variants}) == len(
        variants
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["model_config"].pop("speed_risk_coefficient"),
        lambda value: value["model_config"].update({"unexpected": 1}),
        lambda value: value["model_config"]["components"][0].update(
            {"transform": "unknown"}
        ),
        lambda value: value["model_config"]["components"][0].update(
            {"upper": 0.0}
        ),
        lambda value: value["model_config"]["quality_confidence"].update(
            {"suspect": 1.1}
        ),
    ),
)
def test_v2_configuration_rejects_missing_extra_and_illegal_values(
    tmp_path, mutation
) -> None:
    source = PROJECT_ROOT / "configs/models/demo_unvalidated_v2.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    mutation(document)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RiskPipelineError):
        load_risk_build_configuration(path)


def _legacy_bounded(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _curvilinear_frame(
    formal_fixture,
    *,
    field: np.ndarray | None = None,
    attrs: dict[str, str] | None = None,
):
    source = formal_fixture.prepared.frames["ocean_current"][0]
    x = np.array([9.6, 9.66, 9.72], dtype=np.float64)
    y = np.array([-13.9, -13.84, -13.78], dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    latitude, longitude = _inverse_polar_stereographic(xx, yy)
    if field is None:
        field = 2.0 * yy + 3.0 * xx
    projection_attrs = {
        "query_projection": "+proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273"
    }
    projection_attrs.update(attrs or {})
    dataset = xr.Dataset(
        {
            "wind_u10": (("time", "y", "x"), field[np.newaxis, ...]),
        },
        coords={
            "time": [datetime(2026, 7, 15, tzinfo=UTC)],
            "x": xr.DataArray(x, dims=("x",), attrs={"units": "100  km"}),
            "y": xr.DataArray(y, dims=("y",), attrs={"units": "100  km"}),
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
        },
        attrs=projection_attrs,
    )
    return replace(source, payload=dataset), x, y


def _inverse_polar_stereographic(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.hypot(x, y) * 100_000.0
    latitude = np.rad2deg(np.pi / 2.0 - 2.0 * np.arctan(rho / (2.0 * 6_378_273.0)))
    longitude = -45.0 + np.rad2deg(np.arctan2(x, -y))
    return latitude, longitude
