from __future__ import annotations

from dataclasses import replace
from datetime import timedelta, timezone

import numpy as np
import pytest
from arctic_route_data import StandardDataFrame

from arctic_route_risk import BInputEnvelope, InputIdentityError


def test_envelope_verifies_exact_bundle_and_live_frames(formal_fixture) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )

    assert envelope.verified_bundle.formal_run_eligible
    assert envelope.generation_id == formal_fixture.prepared.generation_id
    assert sum(map(len, envelope.frames.values())) == len(envelope.dataset_bundle.records)


def test_envelope_rejects_run_context_bundle_identity_mismatch(formal_fixture) -> None:
    context = replace(formal_fixture.context, dataset_bundle_digest="f" * 64)

    with pytest.raises(InputIdentityError, match="dataset_bundle_digest"):
        BInputEnvelope.from_prepared_window(
            run_context=context,
            prepared_window=formal_fixture.prepared,
            generation_id=formal_fixture.prepared.generation_id,
            knowledge_as_of=formal_fixture.prepared.as_of_time,
        )


def test_envelope_rejects_future_live_source_even_if_id_matches(formal_fixture) -> None:
    prepared = formal_fixture.prepared
    original = prepared.frames["wind_field"][0]
    future_record = replace(
        original.record,
        issue_time=prepared.as_of_time + timedelta(microseconds=1),
    )
    future = StandardDataFrame(
        record=future_record,
        payload=original.payload,
        generation_id=original.generation_id,
    )
    frames = dict(prepared.frames)
    frames["wind_field"] = (future, *prepared.frames["wind_field"][1:])

    with pytest.raises(InputIdentityError, match="future_information_leakage"):
        BInputEnvelope.from_prepared_window(
            run_context=formal_fixture.context,
            prepared_window=replace(prepared, frames=frames),
            generation_id=prepared.generation_id,
            knowledge_as_of=prepared.as_of_time,
        )


def test_envelope_rejects_equivalent_non_utc_cutoff(formal_fixture) -> None:
    offset = timezone(timedelta(hours=8))
    non_utc = formal_fixture.prepared.as_of_time.astimezone(offset)
    assert non_utc == formal_fixture.prepared.as_of_time

    with pytest.raises(InputIdentityError, match="must use UTC"):
        BInputEnvelope.from_prepared_window(
            run_context=formal_fixture.context,
            prepared_window=replace(formal_fixture.prepared, as_of_time=non_utc),
            generation_id=formal_fixture.prepared.generation_id,
            knowledge_as_of=non_utc,
        )


def test_envelope_rejects_formal_window_subset(formal_fixture) -> None:
    with pytest.raises(InputIdentityError, match="complete RunContext window"):
        BInputEnvelope.from_prepared_window(
            run_context=formal_fixture.context,
            prepared_window=formal_fixture.prepared,
            generation_id=formal_fixture.prepared.generation_id,
            knowledge_as_of=formal_fixture.prepared.as_of_time,
            requested_end=formal_fixture.context.simulation_end - timedelta(hours=1),
        )


def test_envelope_rejects_payload_changed_after_a_attestation(formal_fixture) -> None:
    prepared = formal_fixture.prepared
    original = prepared.frames["wind_field"][0]
    tampered = original.consumer_copy()
    tampered.payload["wind_u10"] = tampered.payload["wind_u10"] + 100.0
    frames = dict(prepared.frames)
    frames["wind_field"] = (tampered, *prepared.frames["wind_field"][1:])

    with pytest.raises(InputIdentityError, match="payload attestation mismatch"):
        BInputEnvelope.from_prepared_window(
            run_context=formal_fixture.context,
            prepared_window=replace(prepared, frames=frames),
            generation_id=prepared.generation_id,
            knowledge_as_of=prepared.as_of_time,
        )


def test_envelope_deep_snapshot_isolated_from_prepared_alias(formal_fixture) -> None:
    prepared = formal_fixture.prepared
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=prepared,
        generation_id=prepared.generation_id,
        knowledge_as_of=prepared.as_of_time,
    )
    before = envelope.frames["wind_field"][0].payload["wind_u10"].values.copy()

    source_payload = prepared.frames["wind_field"][0].payload
    source_payload["wind_u10"] = source_payload["wind_u10"] + 100.0

    np.testing.assert_array_equal(
        envelope.frames["wind_field"][0].payload["wind_u10"].values,
        before,
    )


def test_build_snapshot_rejects_envelope_payload_replacement(formal_fixture) -> None:
    prepared = formal_fixture.prepared
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=prepared,
        generation_id=prepared.generation_id,
        knowledge_as_of=prepared.as_of_time,
    )
    payload = envelope.frames["wave"][0].payload
    payload["significant_wave_height"] = payload["significant_wave_height"] + 1.0

    with pytest.raises(InputIdentityError, match="payload changed after attestation"):
        envelope.verified_build_snapshot()


def test_envelope_requires_exact_attestation_key_set(formal_fixture) -> None:
    prepared = formal_fixture.prepared
    attestations = dict(prepared.payload_attestations)
    attestations.pop(next(iter(attestations)))

    with pytest.raises(InputIdentityError, match="payload attestations differ"):
        BInputEnvelope.from_prepared_window(
            run_context=formal_fixture.context,
            prepared_window=replace(prepared, payload_attestations=attestations),
            generation_id=prepared.generation_id,
            knowledge_as_of=prepared.as_of_time,
        )
