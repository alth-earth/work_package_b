from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import numpy as np
import pytest
import xarray as xr
from arctic_route_contracts import RunContext, ScenarioMode, verify_dataset_bundle
from arctic_route_data import (
    AcquisitionPublisher,
    PartitionedABCache,
    SimulationClock,
    WorkPackageA,
)
from arctic_route_data.issue_time import IssueTimeEvidence, IssueTimeMethod
from arctic_route_data.sources import LocalArchiveSource

from arctic_route_risk import (
    BInputEnvelope,
    InputIdentityError,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
    StaleGenerationError,
    TargetGridConfig,
)

START = datetime(2026, 7, 15, tzinfo=UTC)
END = START + timedelta(hours=2)
AS_OF = datetime(2026, 8, 12, tzinfo=UTC)
BBOX = (10.0, 70.0, 12.0, 72.0)

VARIABLES = {
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
    "wave": (
        "significant_wave_height",
        "mean_wave_direction",
        "peak_wave_period",
    ),
    "wind_field": ("wind_u10", "wind_v10"),
}
CADENCE = {
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
UNITS = {
    "land_sea_mask": "1",
    "ocean_current_u": "m s-1",
    "ocean_current_v": "m s-1",
    "ice_concentration": "1",
    "ice_drift_u": "m s-1",
    "ice_drift_v": "m s-1",
    "ice_edge": "1",
    "ice_thickness": "m",
    "ice_type": "1",
    "air_temperature_2m": "K",
    "visibility": "m",
    "sea_surface_height": "m",
    "significant_wave_height": "m",
    "mean_wave_direction": "degree",
    "peak_wave_period": "s",
    "wind_u10": "m s-1",
    "wind_v10": "m s-1",
}
STANDARD_NAMES = {
    "ocean_current_u": "eastward_sea_water_velocity",
    "ocean_current_v": "northward_sea_water_velocity",
    "ice_concentration": "sea_ice_area_fraction",
    "ice_drift_u": "eastward_sea_ice_velocity",
    "ice_drift_v": "northward_sea_ice_velocity",
    "mean_wave_direction": "sea_surface_wave_from_direction",
    "wind_u10": "eastward_wind",
    "wind_v10": "northward_wind",
}
BASE = {
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
    "mean_wave_direction": 90.0,
    "peak_wave_period": 8.0,
    "wind_u10": 3.0,
    "wind_v10": 1.0,
}


def _issue_evidence() -> IssueTimeEvidence:
    return IssueTimeEvidence(
        issue_time=AS_OF,
        method=IssueTimeMethod.EXPLICIT_CATALOG,
        authority="cross-package archive fixture",
        reference="persisted exact-bundle restart test",
        observed_at=AS_OF,
        raw_value=AS_OF.isoformat(),
    )


def _dataset(data_type: str) -> xr.Dataset:
    cadence = CADENCE[data_type]
    hours = (0,) if cadence is None else tuple(range(0, 4, int(cadence)))
    times = [np.datetime64((START + timedelta(hours=hour)).replace(tzinfo=None)) for hour in hours]
    latitude = np.array([BBOX[1], BBOX[3]], dtype=np.float64)
    longitude = np.array([BBOX[0], BBOX[2]], dtype=np.float64)
    data_vars = {
        variable: (
            ("time", "latitude", "longitude"),
            np.stack(
                [
                    np.full((2, 2), BASE[variable] + index * 1e-4, dtype=np.float64)
                    for index in range(len(times))
                ]
            ),
        )
        for variable in VARIABLES[data_type]
    }
    dataset = xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": latitude, "longitude": longitude},
    )
    for variable in VARIABLES[data_type]:
        dataset[variable].attrs["units"] = UNITS[variable]
        if variable in STANDARD_NAMES:
            dataset[variable].attrs["standard_name"] = STANDARD_NAMES[variable]
    return dataset


def _persist_formal_bundle(data_root):
    snapshot = (
        data_root
        / "source_snapshots"
        / "archive-restart"
        / "snapshot-a"
        / "source.bin"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot_bytes = b"immutable archive-restart fixture source snapshot"
    snapshot.write_bytes(snapshot_bytes)
    snapshot_metadata = {
        "source_snapshot_id": "snapshot-a",
        "source_file": snapshot.name,
        "source_file_checksum": sha256(snapshot_bytes).hexdigest(),
        "source_snapshot_relative_path": snapshot.relative_to(data_root).as_posix(),
    }
    publisher = AcquisitionPublisher(data_root)
    for data_type in sorted(VARIABLES):
        publisher.publish_dataset(
            _dataset(data_type),
            data_type=data_type,
            route_id="archive_restart_corridor",
            source="archive-restart-fixture",
            version="1.0.0",
            issue_evidence=_issue_evidence(),
            metadata={
                **snapshot_metadata,
                "nominal_interval_hours": CADENCE[data_type],
            },
        )
    service = WorkPackageA(
        source=LocalArchiveSource(data_root),
        clock=SimulationClock(START),
        cache=PartitionedABCache(max_memory_mb=16),
    )
    try:
        return service.prepare_window_for_b(
            route_id="archive_restart_corridor",
            data_types=tuple(sorted(VARIABLES)),
            start_time=START,
            target_horizon_hours=2,
            minimum_complete_horizon_hours=2,
            expected_interval_hours=CADENCE,
            knowledge_as_of=AS_OF,
        )
    finally:
        service.close()


@pytest.mark.integration
def test_persisted_a_bundle_restart_resolves_into_b_and_rejects_tamper(tmp_path) -> None:
    data_root = tmp_path / "a-archive"
    initial = _persist_formal_bundle(data_root)
    bundle_path = tmp_path / "dataset-bundle-v2.json"
    bundle_path.write_text(
        json.dumps(initial.dataset_bundle.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    restarted = WorkPackageA(
        source=LocalArchiveSource(data_root),
        clock=SimulationClock(START),
        cache=PartitionedABCache(max_memory_mb=16),
    )
    try:
        resolved = restarted.resolve_dataset_bundle_for_b(
            json.loads(bundle_path.read_text(encoding="utf-8")),
            generation_id=0,
            knowledge_as_of=AS_OF,
        )
    finally:
        restarted.close()

    verified = verify_dataset_bundle(resolved.dataset_bundle.to_dict())
    context = RunContext(
        schema_version="run-context.v2",
        run_id="run-00000000-0000-4000-8000-000000000333",
        created_at=AS_OF,
        scenario_id="archive_restart_scenario",
        scenario_version="1.0.0",
        scenario_mode=ScenarioMode.RETROSPECTIVE_BEST_ESTIMATE,
        simulation_start=START,
        simulation_end=END,
        scenario_digest="1" * 64,
        corridor_id="archive_restart_corridor",
        corridor_version="1.0.0",
        corridor_digest="2" * 64,
        vessel_profile_id="archive_restart_vessel",
        vessel_profile_version="1.0.0",
        vessel_profile_digest="3" * 64,
        dataset_bundle_id=verified.bundle_id,
        dataset_bundle_digest=verified.bundle_digest,
        config_digest="4" * 64,
    )
    envelope = BInputEnvelope.from_prepared_window(
        run_context=context,
        prepared_window=resolved,
        generation_id=0,
        knowledge_as_of=AS_OF,
    )
    frames = RiskBuildService(utc_now=lambda: AS_OF).build_window(
        RiskBuildRequest(
            envelope=envelope,
            target_bbox=BBOX,
            grid_config=TargetGridConfig(
                latitude_step_degrees=2.0,
                longitude_step_degrees=2.0,
            ),
        )
    )
    store_root = tmp_path / "b-store"
    clock = SimulationClock(START)
    store = PersistentRiskStore(store_root)
    unbind = store.bind_generation_authority(context.run_id, clock)
    try:
        committed = store.publish_window(frames)
    finally:
        unbind()

    restarted_store = PersistentRiskStore(store_root)
    restarted_unbind = restarted_store.bind_generation_authority(context.run_id, clock)
    try:
        restored = restarted_store.get_committed_window(committed.query)
        replayed = restarted_store.publish_window(frames)

        assert restored == committed
        assert replayed == committed

        clock.seek(clock.now + timedelta(hours=1))
        with pytest.raises(StaleGenerationError, match="stale_generation"):
            restarted_store.publish_window(frames)
        with pytest.raises(StaleGenerationError, match="stale_generation"):
            restarted_store.get_committed_window(committed.query)
    finally:
        restarted_unbind()

    assert committed.count == 3
    assert len(resolved.payload_attestations) == len(resolved.dataset_bundle.records)

    payload = resolved.frames["wind_field"][0].payload
    payload["wind_u10"] = payload["wind_u10"] + 100.0
    with pytest.raises(InputIdentityError, match="payload attestation mismatch"):
        BInputEnvelope.from_prepared_window(
            run_context=context,
            prepared_window=resolved,
            generation_id=0,
            knowledge_as_of=AS_OF,
        )
