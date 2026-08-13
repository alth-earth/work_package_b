from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from arctic_route_contracts import RunContext, ScenarioMode
from arctic_route_data import (
    CoverageReport,
    DataCategory,
    DatasetBundle,
    ManifestRecord,
    PreparedWindow,
    QualityFlag,
    StandardDataFrame,
    semantic_payload_digest,
)

START = datetime(2026, 7, 15, tzinfo=UTC)
END = START + timedelta(hours=2)
AS_OF = datetime(2026, 8, 1, tzinfo=UTC)
GENERATION = 3
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
    "wave": ("significant_wave_height",),
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


@pytest.fixture
def formal_fixture():
    records: list[ManifestRecord] = []
    live_frames: dict[str, list[StandardDataFrame]] = defaultdict(list)
    verified: dict[str, str] = {}
    for data_type in sorted(VARIABLES):
        cadence = CADENCE[data_type]
        hours = (0,) if cadence is None else tuple(range(0, 4, int(cadence)))
        for index, hour in enumerate(hours):
            data_id = f"{data_type}-{index:02d}"
            checksum = hashlib.sha256(data_id.encode()).hexdigest()
            snapshot = f"snapshot-{data_type}-{index:02d}"
            category = (
                DataCategory.STATIC
                if cadence is None
                else DataCategory.SLOW
                if data_type.startswith("sea_ice")
                or data_type in {"ocean_current", "water_level"}
                else DataCategory.DYNAMIC
            )
            record = ManifestRecord(
                data_id=data_id,
                data_type=data_type,
                category=category,
                route_id="test_corridor",
                variables=VARIABLES[data_type],
                issue_time=AS_OF,
                valid_time=START + timedelta(hours=hour),
                ingest_time=AS_OF,
                bbox=BBOX,
                crs="EPSG:4326",
                resolution=(2.0, 2.0),
                source="formal-fixture",
                quality_flag=QualityFlag.GOOD,
                version="1.0.0",
                checksum=checksum,
                relative_path=f"ready/{data_id}.nc",
                size_bytes=128,
                metadata={
                    "source_snapshot_id": snapshot,
                    "source_file_checksum": checksum,
                    "source_file": f"{data_id}.nc",
                    "nominal_interval_hours": cadence,
                },
            )
            frame = StandardDataFrame(
                record=record,
                payload=_payload(data_type, index),
                generation_id=GENERATION,
            )
            records.append(record)
            live_frames[data_type].append(frame)
            verified[data_id] = snapshot

    bundle = DatasetBundle.create(
        corridor_id="test_corridor",
        as_of_time=AS_OF,
        requested_start=START,
        requested_end=END,
        minimum_required_end=END,
        requested_data_types=tuple(sorted(VARIABLES)),
        records=tuple(records),
        verified_provenance_ids=verified,
        expected_interval_hours=CADENCE,
    )
    reports = {
        proof.data_type: CoverageReport(
            data_type=proof.data_type,
            requested_start=START,
            requested_end=END,
            minimum_required_end=END,
            available_start=proof.available_start,
            available_end=proof.available_end,
            expected_interval_hours=proof.expected_interval_hours,
            missing_intervals=proof.missing_intervals,
            source_snapshot_ids=proof.source_snapshot_ids,
            has_start_support=proof.has_start_support,
            meets_minimum_horizon=proof.meets_minimum_horizon,
            covers_requested_window=proof.covers_requested_window,
            provenance_complete=proof.provenance_complete,
            complete=proof.complete,
        )
        for proof in bundle.coverage
    }
    prepared = PreparedWindow(
        route_id="test_corridor",
        generation_id=GENERATION,
        as_of_time=AS_OF,
        frames={name: tuple(items) for name, items in live_frames.items()},
        payload_attestations={
            frame.record.data_id: semantic_payload_digest(frame.record, frame.payload)
            for frames in live_frames.values()
            for frame in frames
        },
        coverage=reports,
        dataset_bundle=bundle,
    )
    context = RunContext(
        schema_version="run-context.v2",
        run_id="run-00000000-0000-4000-8000-000000000001",
        created_at=AS_OF,
        scenario_id="test_scenario",
        scenario_version="1.0.0",
        scenario_mode=ScenarioMode.RETROSPECTIVE_BEST_ESTIMATE,
        simulation_start=START,
        simulation_end=END,
        scenario_digest="1" * 64,
        corridor_id="test_corridor",
        corridor_version="1.0.0",
        corridor_digest="2" * 64,
        vessel_profile_id="test_vessel",
        vessel_profile_version="1.0.0",
        vessel_profile_digest="3" * 64,
        dataset_bundle_id=bundle.bundle_id,
        dataset_bundle_digest=bundle.bundle_digest,
        config_digest="4" * 64,
    )
    return SimpleNamespace(context=context, prepared=prepared, bbox=BBOX)


def _payload(data_type: str, index: int) -> xr.Dataset:
    latitude = np.array([70.0, 72.0], dtype=np.float64)
    longitude = np.array([10.0, 12.0], dtype=np.float64)
    base = {
        "land_sea_mask": np.array([[1.0, 0.0], [1.0, 1.0]]),
        "ocean_current_u": np.full((2, 2), 0.4),
        "ocean_current_v": np.full((2, 2), 0.2),
        "ice_concentration": np.full((2, 2), 0.4),
        "ice_drift_u": np.full((2, 2), 0.1),
        "ice_drift_v": np.full((2, 2), 0.05),
        "ice_edge": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "ice_thickness": np.full((2, 2), 1.0),
        "ice_type": np.full((2, 2), 2.0),
        "air_temperature_2m": np.full((2, 2), 270.0),
        "visibility": np.full((2, 2), 8_000.0),
        "sea_surface_height": np.full((2, 2), 0.3),
        "significant_wave_height": np.full((2, 2), 2.0),
        "wind_u10": np.full((2, 2), 5.0),
        "wind_v10": np.full((2, 2), 2.0),
    }
    data_vars = {}
    for variable in VARIABLES[data_type]:
        values = base[variable].astype(np.float64, copy=True)
        if data_type not in {"land_sea_mask", "sea_ice_type", "sea_ice_edge"}:
            values += index * 0.01
        data_vars[variable] = (("latitude", "longitude"), values)
    return xr.Dataset(
        data_vars,
        coords={"latitude": latitude, "longitude": longitude},
        attrs={"crs": "EPSG:4326"},
    )
