"""Strict A/shared input envelope for one full formal B build."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Protocol

from arctic_route_contracts import (
    DatasetBundleIdentity,
    RunContext,
    ScenarioMode,
    verify_dataset_bundle,
)
from arctic_route_data import (
    DatasetBundle,
    DatasetBundleRecord,
    StandardDataFrame,
    semantic_payload_digest,
)

from arctic_route_risk.errors import CoverageError, InputIdentityError

REQUIRED_FORMAL_DATA_TYPES = frozenset(
    {
        "land_sea_mask",
        "ocean_current",
        "sea_ice_concentration",
        "sea_ice_drift",
        "sea_ice_edge",
        "sea_ice_thickness",
        "sea_ice_type",
        "temperature",
        "visibility",
        "water_level",
        "wave",
        "wind_field",
    }
)


class CoverageLike(Protocol):
    complete: bool
    requested_start: datetime
    requested_end: datetime
    minimum_required_end: datetime


class PreparedWindowLike(Protocol):
    route_id: str
    generation_id: int
    as_of_time: datetime
    frames: Mapping[str, tuple[StandardDataFrame, ...]]
    payload_attestations: Mapping[str, str]
    coverage: Mapping[str, CoverageLike]
    dataset_bundle: DatasetBundle


@dataclass(frozen=True, slots=True)
class BInputEnvelope:
    """Verified identity plus exact live A payloads for one RunContext window.

    The shared verifier proves the bundle document. This class separately proves
    that every live payload is the exact record named by that document.
    """

    run_context: RunContext
    dataset_bundle: DatasetBundle
    verified_bundle: DatasetBundleIdentity
    frames: Mapping[str, tuple[StandardDataFrame, ...]]
    payload_attestations: Mapping[str, str]
    generation_id: int
    knowledge_as_of: datetime
    requested_start: datetime
    requested_end: datetime

    @classmethod
    def from_prepared_window(
        cls,
        *,
        run_context: RunContext,
        prepared_window: PreparedWindowLike,
        generation_id: int,
        knowledge_as_of: datetime,
        requested_start: datetime | None = None,
        requested_end: datetime | None = None,
    ) -> BInputEnvelope:
        start = requested_start or run_context.simulation_start
        end = requested_end or run_context.simulation_end
        for name, value in (
            ("run_context.simulation_start", run_context.simulation_start),
            ("run_context.simulation_end", run_context.simulation_end),
            ("prepared_window.as_of_time", prepared_window.as_of_time),
            ("requested_start", start),
            ("requested_end", end),
        ):
            _require_utc(value, field=name)
        if start != run_context.simulation_start or end != run_context.simulation_end:
            raise InputIdentityError(
                "input_identity_mismatch: formal B build must use the complete RunContext window"
            )
        active_generation = generation_id
        if (
            not isinstance(active_generation, int)
            or isinstance(active_generation, bool)
            or active_generation < 0
        ):
            raise InputIdentityError(
                "input_identity_mismatch: generation_id must be non-negative int"
            )
        if active_generation != prepared_window.generation_id:
            raise InputIdentityError(
                "input_identity_mismatch: orchestration generation differs from PreparedWindow"
            )
        as_of = knowledge_as_of
        _require_utc(as_of, field="knowledge_as_of")
        if as_of != prepared_window.as_of_time:
            raise InputIdentityError(
                "input_identity_mismatch: knowledge_as_of differs from PreparedWindow"
            )
        bundle = prepared_window.dataset_bundle
        verified = verify_dataset_bundle(bundle.to_dict())
        _validate_top_level_identity(run_context, prepared_window, verified, start, end, as_of)
        _validate_coverage(prepared_window, bundle, start, end)
        frozen_frames, payload_attestations = _validate_and_freeze_frames(
            prepared_window, bundle, active_generation, as_of
        )
        return cls(
            run_context=run_context,
            dataset_bundle=bundle,
            verified_bundle=verified,
            frames=frozen_frames,
            payload_attestations=payload_attestations,
            generation_id=active_generation,
            knowledge_as_of=as_of,
            requested_start=start,
            requested_end=end,
        )

    def verified_build_snapshot(self) -> BInputEnvelope:
        """Return a fresh private payload snapshot after rechecking A attestations.

        The envelope is intentionally inspectable, so a caller can replace an
        xarray variable even though its current NumPy buffers are read-only.
        Every build therefore re-verifies the semantic digest and then deep
        copies again before the risk algorithm reads any values.
        """

        snapshots = _snapshot_attested_frames(
            self.frames,
            self.payload_attestations,
        )
        return replace(self, frames=snapshots)


def _validate_top_level_identity(
    context: RunContext,
    prepared: PreparedWindowLike,
    verified: DatasetBundleIdentity,
    start: datetime,
    end: datetime,
    as_of: datetime,
) -> None:
    if not verified.formal_run_eligible:
        raise CoverageError(
            "forecast_coverage_insufficient: DatasetBundle is not formal-run eligible"
        )
    comparisons = {
        "dataset_bundle_id": (context.dataset_bundle_id, verified.bundle_id),
        "dataset_bundle_digest": (context.dataset_bundle_digest, verified.bundle_digest),
        "corridor_id": (context.corridor_id, verified.corridor_id),
        "prepared.route_id": (context.corridor_id, prepared.route_id),
        "prepared.as_of_time": (verified.as_of_time, as_of),
        "requested_start": (verified.requested_start, start),
    }
    mismatches = [name for name, (left, right) in comparisons.items() if left != right]
    if mismatches:
        raise InputIdentityError(
            "input_identity_mismatch: " + ", ".join(sorted(mismatches))
        )
    if verified.requested_end < end or verified.minimum_required_end < end:
        raise CoverageError(
            "forecast_coverage_insufficient: DatasetBundle does not cover RunContext end"
        )
    if context.scenario_mode is ScenarioMode.FROZEN_FORECAST and as_of > start:
        raise InputIdentityError(
            "future_information_leakage: frozen forecast knowledge is later than simulation start"
        )
    bundle_types = set(verified.requested_data_types)
    missing = sorted(REQUIRED_FORMAL_DATA_TYPES - bundle_types)
    if missing:
        raise CoverageError(
            "forecast_coverage_insufficient: missing formal data types: " + ", ".join(missing)
        )


def _validate_coverage(
    prepared: PreparedWindowLike,
    bundle: DatasetBundle,
    start: datetime,
    end: datetime,
) -> None:
    expected_types = set(bundle.requested_data_types)
    if set(prepared.coverage) != expected_types or set(prepared.frames) != expected_types:
        raise InputIdentityError(
            "input_identity_mismatch: PreparedWindow type keys differ from DatasetBundle"
        )
    incomplete = sorted(name for name, report in prepared.coverage.items() if not report.complete)
    if incomplete:
        raise CoverageError(
            "forecast_coverage_insufficient: incomplete PreparedWindow types: "
            + ", ".join(incomplete)
        )
    for data_type, report in prepared.coverage.items():
        if (
            report.requested_start != start
            or report.requested_end < end
            or report.minimum_required_end < end
        ):
            raise CoverageError(
                f"forecast_coverage_insufficient: {data_type} report does not cover full window"
            )


def _validate_and_freeze_frames(
    prepared: PreparedWindowLike,
    bundle: DatasetBundle,
    generation_id: int,
    as_of: datetime,
) -> tuple[
    Mapping[str, tuple[StandardDataFrame, ...]],
    Mapping[str, str],
]:
    by_id: dict[str, StandardDataFrame] = {}
    for data_type, frames in prepared.frames.items():
        for frame in frames:
            record = frame.record
            if record.data_id in by_id:
                raise InputIdentityError(
                    f"input_identity_mismatch: duplicate live data_id {record.data_id}"
                )
            if frame.generation_id != generation_id:
                raise InputIdentityError(
                    f"input_identity_mismatch: stale generation for {record.data_id}"
                )
            if record.route_id != prepared.route_id or record.data_type != data_type:
                raise InputIdentityError(
                    f"input_identity_mismatch: route/type mismatch for {record.data_id}"
                )
            if record.issue_time > as_of:
                raise InputIdentityError(
                    f"future_information_leakage: {record.data_id} issued after knowledge_as_of"
                )
            by_id[record.data_id] = frame

    expected_ids = {record.data_id for record in bundle.records}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise InputIdentityError(
            "input_identity_mismatch: exact live frames differ from bundle "
            f"missing={missing}, extra={extra}"
        )
    attestations = prepared.payload_attestations
    if not isinstance(attestations, Mapping) or set(attestations) != expected_ids:
        raise InputIdentityError(
            "input_identity_mismatch: PreparedWindow payload attestations differ "
            "from DatasetBundle records"
        )
    for bundled in bundle.records:
        live = by_id[bundled.data_id]
        if DatasetBundleRecord.from_manifest(live.record) != bundled:
            raise InputIdentityError(
                f"input_identity_mismatch: {bundled.data_id} record/provenance differs "
                "from DatasetBundle"
            )
        expected_attestation = attestations[bundled.data_id]
        _validate_attestation_shape(expected_attestation, data_id=bundled.data_id)
        actual_attestation = semantic_payload_digest(live.record, live.payload)
        if actual_attestation != expected_attestation:
            raise InputIdentityError(
                f"input_identity_mismatch: {bundled.data_id} payload attestation mismatch"
            )
    snapshots = _snapshot_attested_frames(prepared.frames, attestations)
    return snapshots, MappingProxyType(dict(attestations))


def _snapshot_attested_frames(
    frames_by_type: Mapping[str, tuple[StandardDataFrame, ...]],
    attestations: Mapping[str, str],
) -> Mapping[str, tuple[StandardDataFrame, ...]]:
    snapshots: dict[str, tuple[StandardDataFrame, ...]] = {}
    for data_type, frames in frames_by_type.items():
        copied: list[StandardDataFrame] = []
        for frame in frames:
            data_id = frame.record.data_id
            expected = attestations.get(data_id)
            _validate_attestation_shape(expected, data_id=data_id)
            if semantic_payload_digest(frame.record, frame.payload) != expected:
                raise InputIdentityError(
                    f"input_identity_mismatch: {data_id} payload changed after attestation"
                )
            snapshot = frame.consumer_view()
            if semantic_payload_digest(snapshot.record, snapshot.payload) != expected:
                raise InputIdentityError(
                    f"input_identity_mismatch: {data_id} payload changed while snapshotting"
                )
            copied.append(snapshot)
        snapshots[data_type] = tuple(copied)
    return MappingProxyType(snapshots)


def _validate_attestation_shape(value: object, *, data_id: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputIdentityError(
            f"input_identity_mismatch: {data_id} has invalid payload attestation"
        )


def _require_utc(value: datetime, *, field: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise InputIdentityError(f"input_identity_mismatch: {field} must use UTC")
