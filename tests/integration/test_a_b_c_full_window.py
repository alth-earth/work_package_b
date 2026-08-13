from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from arctic_route_contracts import (
    RunContext,
    ScenarioMode,
    create_run_context,
    verify_dataset_bundle,
)
from arctic_route_data import (
    DataCategory,
    ManifestRecord,
    PartitionedABCache,
    QualityFlag,
    SimulationClock,
    StandardDataFrame,
    WorkPackageA,
)
from arctic_route_planning import RiskSourcePlanningIngress
from arctic_route_planning.config import load_configuration
from arctic_route_planning.contracts import ProvenanceKind
from arctic_route_planning.service import ServicePlanningRequest

from arctic_route_risk import (
    BInputEnvelope,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
    TargetGridConfig,
)

CONFIG_ROOT = Path(__file__).parents[3] / "work_package_c/configs"
SCENARIO_ID = "tromso_isfjorden_july_2026_retrospective_v1"
AS_OF = datetime(2026, 8, 12, tzinfo=UTC)
TARGET_BBOX = (13.0, 69.75, 19.0, 78.15)

_VARIABLES = {
    "land_sea_mask": ("land_sea_mask",),
    "ocean_current": ("ocean_current_u", "ocean_current_v"),
    "sea_ice_concentration": ("ice_concentration",),
    "sea_ice_drift": ("ice_drift_u", "ice_drift_v"),
    "sea_ice_edge": ("ice_edge",),
    "sea_ice_thickness": ("ice_thickness",),
    "sea_ice_type": ("ice_type",),
    "temperature": ("air_temperature_2m",),
    "visibility": ("visibility",),
    "water_level": ("sea_surface_height",),
    "wave": ("significant_wave_height",),
    "wind_field": ("wind_u10", "wind_v10"),
}
_CADENCE = {
    "land_sea_mask": None,
    "ocean_current": 1.0,
    "sea_ice_concentration": 1.0,
    "sea_ice_drift": 1.0,
    "sea_ice_edge": 1.0,
    "sea_ice_thickness": 1.0,
    "sea_ice_type": 1.0,
    "temperature": 3.0,
    "visibility": 3.0,
    "water_level": 1.0,
    "wave": 3.0,
    "wind_field": 3.0,
}


class _FormalFixtureSource:
    """A DataSource fixture with verifiable native-style provenance metadata."""

    def __init__(
        self,
        *,
        start: datetime,
        end: datetime,
        corridor_id: str,
        source_bbox: tuple[float, float, float, float] = TARGET_BBOX,
        fixture_id: str = "e2e",
    ) -> None:
        self.records: dict[str, tuple[ManifestRecord, ...]] = {}
        self.payloads: dict[str, xr.Dataset] = {}
        horizon_hours = int((end - start).total_seconds() // 3600)
        for data_type in sorted(_VARIABLES):
            cadence = _CADENCE[data_type]
            hours = (0,) if cadence is None else range(0, horizon_hours + 1, int(cadence))
            records: list[ManifestRecord] = []
            for index, hour in enumerate(hours):
                data_id = f"{fixture_id}-{data_type}-{index:03d}"
                checksum = hashlib.sha256(data_id.encode()).hexdigest()
                snapshot_id = f"e2e-snapshot-{data_type}-{index:03d}"
                valid_time = start + timedelta(hours=hour)
                record = ManifestRecord(
                    data_id=data_id,
                    data_type=data_type,
                    category=_category(data_type),
                    route_id=corridor_id,
                    variables=_VARIABLES[data_type],
                    issue_time=AS_OF,
                    valid_time=valid_time,
                    ingest_time=AS_OF,
                    bbox=source_bbox,
                    crs="EPSG:4326",
                    resolution=(2.8, 3.0),
                    source="e2e-formal-fixture",
                    quality_flag=QualityFlag.GOOD,
                    version="1.0.0",
                    checksum=checksum,
                    relative_path=f"ready/{data_id}.nc",
                    size_bytes=512,
                    metadata={
                        "source_snapshot_id": snapshot_id,
                        "source_file_checksum": checksum,
                        "source_file": f"{data_id}.nc",
                        "nominal_interval_hours": cadence,
                    },
                )
                records.append(record)
                self.payloads[data_id] = _payload(record, index, source_bbox)
            self.records[data_type] = tuple(records)

    def list_available(
        self, data_type, start_time, end_time, *, route_id, as_of
    ) -> tuple[ManifestRecord, ...]:
        return tuple(
            record
            for record in self.records[data_type]
            if record.route_id == route_id
            and record.issue_time <= as_of
            and start_time <= record.valid_time <= end_time
        )

    def get_latest_before(
        self, data_type, target_time, *, route_id, as_of
    ) -> ManifestRecord | None:
        candidates = [
            record
            for record in self.records[data_type]
            if record.route_id == route_id
            and record.issue_time <= as_of
            and record.valid_time <= target_time
        ]
        return candidates[-1] if candidates else None

    def get_bracketing(
        self, data_type, target_time, *, route_id, as_of
    ) -> tuple[ManifestRecord | None, ManifestRecord | None]:
        candidates = tuple(
            record
            for record in self.records[data_type]
            if record.route_id == route_id and record.issue_time <= as_of
        )
        lower = [record for record in candidates if record.valid_time <= target_time]
        upper = [record for record in candidates if record.valid_time >= target_time]
        return (lower[-1] if lower else None, upper[0] if upper else None)

    def load_frame(self, record, *, generation_id, as_of) -> StandardDataFrame:
        if record.issue_time > as_of:
            raise ValueError("future fixture record")
        return StandardDataFrame(
            record,
            self.payloads[record.data_id].copy(deep=True),
            generation_id,
        )

    def verified_provenance_id(self, record: ManifestRecord) -> str | None:
        # A independently compares this with record_provenance_id(record).
        return str(record.metadata["source_snapshot_id"])


@pytest.mark.integration
def test_public_a_to_b_to_c_full_96_hour_formal_window(tmp_path) -> None:
    configuration = load_configuration(CONFIG_ROOT, SCENARIO_ID)
    scenario = configuration.scenario
    assert scenario.simulation_start is not None and scenario.simulation_end is not None
    source = _FormalFixtureSource(
        start=scenario.simulation_start,
        end=scenario.simulation_end,
        corridor_id=configuration.corridor.corridor_id,
    )
    a_service = WorkPackageA(
        source=source,
        clock=SimulationClock(scenario.simulation_start),
        cache=PartitionedABCache(max_memory_mb=32),
    )
    try:
        prepared = a_service.prepare_window_for_b(
            route_id=configuration.corridor.corridor_id,
            data_types=scenario.required_data_types,
            start_time=scenario.simulation_start,
            target_horizon_hours=scenario.horizon_hours,
            minimum_complete_horizon_hours=scenario.horizon_hours,
            expected_interval_hours=_CADENCE,
            knowledge_as_of=AS_OF,
        )
    finally:
        a_service.close()
    assert all(report.complete for report in prepared.coverage.values())

    run_context = create_run_context(
        scenario=scenario,
        corridor=configuration.corridor,
        vessel=configuration.vessel,
        dataset_bundle=verify_dataset_bundle(prepared.dataset_bundle.to_dict()),
        run_id="run-00000000-0000-4000-8000-000000000096",
        created_at=AS_OF,
    )
    envelope = BInputEnvelope.from_prepared_window(
        run_context=run_context,
        prepared_window=prepared,
        generation_id=prepared.generation_id,
        knowledge_as_of=prepared.as_of_time,
    )
    build_request = RiskBuildRequest(
        envelope=envelope,
        target_bbox=TARGET_BBOX,
        grid_config=TargetGridConfig(
            latitude_step_degrees=3.0,
            longitude_step_degrees=3.0,
        ),
    )
    risk_frames = RiskBuildService(utc_now=lambda: AS_OF).build_window(build_request)
    assert len(risk_frames) == 97

    store = PersistentRiskStore(tmp_path / "bc-store")
    store.activate_generation(run_context.run_id, prepared.generation_id)
    committed = store.publish_window(risk_frames)
    assert committed.count == 97

    request = ServicePlanningRequest(
        run_context=run_context,
        scenario=scenario,
        corridor=configuration.corridor,
        vessel=configuration.vessel,
        vessel_model=configuration.vessel_model,
        model_config_digest=build_request.model_config_digest,
        planner_config_digest=configuration.planner_config_digest,
        risk_provenance=ProvenanceKind.FORMAL,
        generation_id=prepared.generation_id,
        input_revision=0,
        as_of_time=AS_OF,
        start_time=scenario.simulation_start,
        start=(0, 2),
        goal=(3, 0),
        maximum_elapsed=scenario.simulation_end - scenario.simulation_start,
    )
    ingress = RiskSourcePlanningIngress(
        store,
        configuration=configuration,
    )
    prepared_planning = ingress.prepare(request)
    assert prepared_planning.window.commit_id == committed.commit_id

    batch = prepared_planning.execute()

    assert batch.published
    assert batch.selected.provenance is ProvenanceKind.FORMAL
    assert batch.selected.run_id == run_context.run_id
    assert batch.selected.config_digest == run_context.config_digest
    assert batch.selected.model_config_digest == build_request.model_config_digest
    assert batch.selected.destination_reached

    # Migration engineering fixture: the second public corridor uses the same
    # B policy identity, while its realized grid and frame IDs remain distinct.
    second = load_configuration(
        CONFIG_ROOT,
        "murmansk_dikson_july_2026_retrospective_v1",
    )
    second_scenario = second.scenario
    assert second_scenario.simulation_start is not None
    assert second_scenario.simulation_end is not None
    source_bbox = _bbox_tuple(second.corridor.data_bbox)
    second_source = _FormalFixtureSource(
        start=second_scenario.simulation_start,
        end=second_scenario.simulation_end,
        corridor_id=second.corridor.corridor_id,
        source_bbox=source_bbox,
        fixture_id="migration-168h",
    )
    second_a = WorkPackageA(
        source=second_source,
        clock=SimulationClock(second_scenario.simulation_start),
        cache=PartitionedABCache(max_memory_mb=32),
    )
    try:
        second_prepared = second_a.prepare_window_for_b(
            route_id=second.corridor.corridor_id,
            data_types=second_scenario.required_data_types,
            start_time=second_scenario.simulation_start,
            target_horizon_hours=second_scenario.horizon_hours,
            minimum_complete_horizon_hours=second_scenario.horizon_hours,
            expected_interval_hours=_CADENCE,
            knowledge_as_of=AS_OF,
        )
    finally:
        second_a.close()
    second_context = create_run_context(
        scenario=second_scenario,
        corridor=second.corridor,
        vessel=second.vessel,
        dataset_bundle=verify_dataset_bundle(second_prepared.dataset_bundle.to_dict()),
        run_id="run-00000000-0000-4000-8000-000000000168",
        created_at=AS_OF,
    )
    second_envelope = BInputEnvelope.from_prepared_window(
        run_context=second_context,
        prepared_window=second_prepared,
        generation_id=second_prepared.generation_id,
        knowledge_as_of=second_prepared.as_of_time,
    )
    second_request = RiskBuildRequest(
        envelope=second_envelope,
        target_bbox=(33.6, 69.15, 80.4, 73.55),
        grid_config=build_request.grid_config,
        model_config=build_request.model_config,
    )
    second_frames = RiskBuildService(utc_now=lambda: AS_OF).build_window(second_request)
    second_store = PersistentRiskStore(tmp_path / "bc-store-migration-168h")
    second_store.activate_generation(second_context.run_id, second_prepared.generation_id)
    second_commit = second_store.publish_window(second_frames)

    assert second_commit.count == 169
    assert second_request.model_config_digest == build_request.model_config_digest
    assert second_frames[0].payload.attrs["grid_id"] != risk_frames[0].payload.attrs["grid_id"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("horizon_hours", "expected_count"),
    ((96, 97), (168, 169), (216, 217)),
)
def test_b_dynamic_full_window_frame_counts(
    tmp_path, horizon_hours: int, expected_count: int
) -> None:
    """Engineering evidence for all currently supported migration horizons."""

    start = datetime(2026, 7, 15, tzinfo=UTC)
    end = start + timedelta(hours=horizon_hours)
    corridor_id = f"fixture_corridor_h{horizon_hours}"
    source = _FormalFixtureSource(
        start=start,
        end=end,
        corridor_id=corridor_id,
        fixture_id=f"horizon-{horizon_hours}h",
    )
    a_service = WorkPackageA(
        source=source,
        clock=SimulationClock(start),
        cache=PartitionedABCache(max_memory_mb=32),
    )
    try:
        prepared = a_service.prepare_window_for_b(
            route_id=corridor_id,
            data_types=tuple(sorted(_VARIABLES)),
            start_time=start,
            target_horizon_hours=horizon_hours,
            minimum_complete_horizon_hours=horizon_hours,
            expected_interval_hours=_CADENCE,
            knowledge_as_of=AS_OF,
        )
    finally:
        a_service.close()
    context = RunContext(
        schema_version="run-context.v2",
        run_id=f"run-00000000-0000-4000-8000-{horizon_hours:012d}",
        created_at=AS_OF,
        scenario_id=f"fixture_scenario_h{horizon_hours}",
        scenario_version="1.0.0",
        scenario_mode=ScenarioMode.RETROSPECTIVE_BEST_ESTIMATE,
        simulation_start=start,
        simulation_end=end,
        scenario_digest="1" * 64,
        corridor_id=corridor_id,
        corridor_version="1.0.0",
        corridor_digest="2" * 64,
        vessel_profile_id="fixture_vessel",
        vessel_profile_version="1.0.0",
        vessel_profile_digest="3" * 64,
        dataset_bundle_id=prepared.dataset_bundle.bundle_id,
        dataset_bundle_digest=prepared.dataset_bundle.bundle_digest,
        config_digest="4" * 64,
    )
    envelope = BInputEnvelope.from_prepared_window(
        run_context=context,
        prepared_window=prepared,
        generation_id=prepared.generation_id,
        knowledge_as_of=prepared.as_of_time,
    )
    request = RiskBuildRequest(
        envelope=envelope,
        target_bbox=TARGET_BBOX,
        grid_config=TargetGridConfig(
            latitude_step_degrees=3.0,
            longitude_step_degrees=3.0,
        ),
    )
    frames = RiskBuildService(utc_now=lambda: AS_OF).build_window(request)
    store = PersistentRiskStore(tmp_path / f"horizon-{horizon_hours}h")
    store.activate_generation(context.run_id, prepared.generation_id)
    committed = store.publish_window(frames)

    assert len(frames) == expected_count
    assert committed.count == expected_count


def _category(data_type: str) -> DataCategory:
    if data_type == "land_sea_mask":
        return DataCategory.STATIC
    if data_type in {
        "ocean_current",
        "sea_ice_concentration",
        "sea_ice_drift",
        "sea_ice_edge",
        "sea_ice_thickness",
        "sea_ice_type",
        "water_level",
    }:
        return DataCategory.SLOW
    return DataCategory.DYNAMIC


def _payload(
    record: ManifestRecord,
    index: int,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    latitude = np.linspace(bbox[1], bbox[3], 4, dtype=np.float64)
    longitude = np.linspace(bbox[0], bbox[2], 3, dtype=np.float64)
    shape = (latitude.size, longitude.size)
    base = {
        "land_sea_mask": 1.0,
        "ocean_current_u": 0.2,
        "ocean_current_v": 0.1,
        "ice_concentration": 0.15,
        "ice_drift_u": 0.05,
        "ice_drift_v": 0.02,
        "ice_edge": 0.0,
        "ice_thickness": 0.4,
        "ice_type": 1.0,
        "air_temperature_2m": 273.0,
        "visibility": 15_000.0,
        "sea_surface_height": 0.1,
        "significant_wave_height": 0.8,
        "wind_u10": 3.0,
        "wind_v10": 1.0,
    }
    data_vars = {
        variable: (
            ("latitude", "longitude"),
            np.full(shape, base[variable] + index * 1e-5, dtype=np.float64),
        )
        for variable in record.variables
    }
    return xr.Dataset(
        data_vars,
        coords={"latitude": latitude, "longitude": longitude},
        attrs={
            "data_type": record.data_type,
            "route_id": record.route_id,
            "issue_time": _iso_z(record.issue_time),
            "valid_time": _iso_z(record.valid_time),
        },
    )


def _iso_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _bbox_tuple(value) -> tuple[float, float, float, float]:
    return (value.west, value.south, value.east, value.north)
