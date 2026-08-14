from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from arctic_route_data import SimulationClock
from arctic_route_planning.contracts import (
    RiskWindowQuery,
    canonical_risk_frame_bytes,
    canonical_risk_id,
)
from arctic_route_planning.errors import ContextMismatchError, RiskCoverageError

from arctic_route_risk import (
    BInputEnvelope,
    PersistentRiskStore,
    PublicationConflictError,
    RiskBuildRequest,
    RiskBuildService,
    RiskPipelineError,
    StaleGenerationError,
)


def _built(formal_fixture):
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    return RiskBuildService(
        utc_now=lambda: datetime(2026, 8, 2, tzinfo=UTC)
    ).build_window(RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox))


def _for_run(frames, run_id: str):
    result = []
    for frame in frames:
        draft = replace(frame, run_id=run_id, risk_id="draft")
        result.append(replace(draft, risk_id=canonical_risk_id(draft)))
    return tuple(result)


def test_persistent_store_idempotently_commits_exact_window(tmp_path, formal_fixture) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)

    first = store.publish_window(frames)
    second = store.publish_window(frames)
    restored = store.get_committed_window(first.query)

    assert first.commit_id == second.commit_id == restored.commit_id
    assert [frame.risk_id for frame in restored.frames] == [frame.risk_id for frame in frames]
    assert restored.count == 3

    newer_query = RiskWindowQuery(
        start=first.start,
        end=first.end,
        interval=first.interval,
        run_id=first.run_id,
        scenario_id=first.scenario_id,
        corridor_id=first.corridor_id,
        generation_id=first.generation_id,
        vessel_profile_id=first.vessel_profile_id,
        config_digest=first.config_digest,
        model_config_digest=first.model_config_digest,
        as_of=first.as_of + timedelta(hours=1),
    )
    with pytest.raises(ContextMismatchError):
        store.get_committed_window(newer_query)


def test_publish_suffix_window_preserves_canonical_frame_identity(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)

    committed = store.publish_suffix_window(
        frames,
        start=frames[0].valid_time + timedelta(hours=1),
    )

    assert committed.start == frames[1].valid_time
    assert committed.end == frames[-1].valid_time
    assert committed.count == 2
    assert tuple(frame.risk_id for frame in committed.frames) == tuple(
        frame.risk_id for frame in frames[1:]
    )
    assert {frame.generation_id for frame in committed.frames} == {
        frames[0].generation_id
    }
    assert {frame.as_of_time for frame in committed.frames} == {frames[0].as_of_time}


def test_publish_suffix_window_requires_exact_hourly_boundary(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)

    with pytest.raises(RiskPipelineError, match="exactly match"):
        store.publish_suffix_window(
            frames,
            start=frames[0].valid_time + timedelta(minutes=30),
        )


def test_publish_suffix_window_validates_complete_input_before_slicing(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)

    with pytest.raises(RiskCoverageError, match="frames 数量"):
        store.publish_suffix_window(
            (frames[0], frames[2]),
            start=frames[2].valid_time,
        )


def test_publish_suffix_window_rejects_mixed_as_of(tmp_path, formal_fixture) -> None:
    frames = list(_built(formal_fixture))
    changed = replace(
        frames[1],
        as_of_time=frames[1].as_of_time + timedelta(hours=1),
        risk_id="draft",
    )
    frames[1] = replace(changed, risk_id=canonical_risk_id(changed))
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)

    with pytest.raises(PublicationConflictError, match="one exact as_of_time"):
        store.publish_suffix_window(frames, start=frames[1].valid_time)


def test_generation_fence_rejects_late_old_task_and_hides_old_commit(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    committed = store.publish_window(frames)

    store.activate_generation(frames[0].run_id, frames[0].generation_id + 1)

    with pytest.raises(StaleGenerationError, match="stale_generation"):
        store.publish_window(frames)
    with pytest.raises(StaleGenerationError, match="stale_generation"):
        store.get_committed_window(committed.query)


def test_immutable_frame_conflict_is_rejected(tmp_path, formal_fixture) -> None:
    frame = _built(formal_fixture)[0]
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frame.run_id, frame.generation_id)
    store.publish(frame)
    frame_path = tmp_path / "frames" / f"{frame.risk_id}.json"
    frame_path.write_text("{}", encoding="utf-8")

    with pytest.raises(PublicationConflictError, match="different content"):
        store.publish(frame)


def test_publish_uses_private_canonical_snapshot_before_store_lock(
    tmp_path, formal_fixture, monkeypatch
) -> None:
    frame = _built(formal_fixture)[0]
    expected_bytes = canonical_risk_frame_bytes(frame)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frame.run_id, frame.generation_id)
    write_frame = store._write_frame

    def mutate_caller_then_write(snapshot) -> None:
        frame.payload.attrs["caller_mutated_after_snapshot"] = True
        write_frame(snapshot)

    monkeypatch.setattr(store, "_write_frame", mutate_caller_then_write)

    store.publish(frame)

    persisted = tmp_path / "frames" / f"{frame.risk_id}.json"
    assert persisted.read_bytes() == expected_bytes


def test_publish_window_caller_mutation_cannot_split_commit_artifacts(
    tmp_path, formal_fixture, monkeypatch
) -> None:
    frames = _built(formal_fixture)
    expected_first_bytes = canonical_risk_frame_bytes(frames[0])
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    write_frame = store._write_frame
    mutated = False

    def write_snapshot_then_mutate_caller(snapshot) -> None:
        nonlocal mutated
        write_frame(snapshot)
        if not mutated:
            frames[0].payload.attrs["caller_mutated_during_publication"] = True
            mutated = True

    monkeypatch.setattr(store, "_write_frame", write_snapshot_then_mutate_caller)

    committed = store.publish_window(frames)
    restored = store.get_committed_window(committed.query)

    assert mutated
    assert restored.commit_id == committed.commit_id
    assert restored.content_digest == committed.content_digest
    assert canonical_risk_frame_bytes(restored.frames[0]) == expected_first_bytes


def test_tampered_commit_id_cannot_escape_commit_directory(tmp_path, formal_fixture) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    committed = store.publish_window(frames)
    pointer_path = next((tmp_path / "queries").glob("*.json"))
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["commit_id"] = "../../outside"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(PublicationConflictError, match="invalid commit_id"):
        store.get_committed_window(committed.query)


@pytest.mark.parametrize("unsafe_id", ("../outside", "/tmp/outside", "risk-not-canonical"))
def test_store_rejects_unsafe_nonformal_id_before_forming_path(
    tmp_path, formal_fixture, unsafe_id
) -> None:
    frame = _built(formal_fixture)[0]
    unsafe = replace(frame, risk_id=unsafe_id, provenance="synthetic")
    store = PersistentRiskStore(tmp_path / "store")
    store.activate_generation(unsafe.run_id, unsafe.generation_id)
    outside = tmp_path / "outside.json"

    with pytest.raises(PublicationConflictError, match="only accepts formal"):
        store.publish(unsafe)

    assert not outside.exists()
    assert not list((tmp_path / "store" / "frames").iterdir())


def test_generation_change_waits_for_committed_execution_lease(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    committed = store.publish_window(frames)
    started = threading.Event()
    finished = threading.Event()

    def change_generation() -> None:
        started.set()
        store.activate_generation(frames[0].run_id, frames[0].generation_id + 1)
        finished.set()

    with store.lease_committed_window(committed.query) as leased:
        assert leased.commit_id == committed.commit_id
        worker = threading.Thread(target=change_generation)
        worker.start()
        assert started.wait(timeout=1)
        time.sleep(0.05)
        assert not finished.is_set()

    worker.join(timeout=1)
    assert finished.is_set()
    with pytest.raises(StaleGenerationError):
        store.get_committed_window(committed.query)


def test_same_run_execution_leases_are_shared(tmp_path, formal_fixture) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    committed = store.publish_window(frames)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_lease(entered: threading.Event) -> None:
        try:
            with store.lease_committed_window(committed.query):
                entered.set()
                if not release.wait(timeout=5):
                    raise AssertionError("test did not release execution lease")
        except BaseException as exc:  # thread failures must reach the test
            errors.append(exc)

    first = threading.Thread(target=hold_lease, args=(first_entered,))
    second = threading.Thread(target=hold_lease, args=(second_entered,))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    concurrent = second_entered.wait(timeout=2)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert concurrent
    assert not first.is_alive() and not second.is_alive()
    assert not errors


def test_execution_leases_for_different_runs_are_independent(
    tmp_path, formal_fixture
) -> None:
    first_frames = _built(formal_fixture)
    second_frames = _for_run(
        first_frames,
        "run-00000000-0000-4000-8000-000000000777",
    )
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(first_frames[0].run_id, first_frames[0].generation_id)
    store.activate_generation(second_frames[0].run_id, second_frames[0].generation_id)
    first_commit = store.publish_window(first_frames)
    second_commit = store.publish_window(second_frames)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_lease(query, entered: threading.Event) -> None:
        try:
            with store.lease_committed_window(query):
                entered.set()
                if not release.wait(timeout=5):
                    raise AssertionError("test did not release execution lease")
        except BaseException as exc:  # thread failures must reach the test
            errors.append(exc)

    first = threading.Thread(target=hold_lease, args=(first_commit.query, first_entered))
    second = threading.Thread(target=hold_lease, args=(second_commit.query, second_entered))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    concurrent = second_entered.wait(timeout=2)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert concurrent
    assert not first.is_alive() and not second.is_alive()
    assert not errors


def test_public_clock_binding_fences_old_prepared_generation(
    tmp_path, formal_fixture
) -> None:
    frames = _built(formal_fixture)
    store = PersistentRiskStore(tmp_path)
    clock = SimulationClock(frames[0].valid_time)
    for _ in range(frames[0].generation_id):
        clock.seek(clock.now)
    unsubscribe = store.bind_generation_authority(frames[0].run_id, clock)
    try:
        committed = store.publish_window(frames)
        clock.seek(clock.now + timedelta(hours=1))

        with pytest.raises(StaleGenerationError, match="stale_generation"):
            store.publish_window(frames)
        with pytest.raises(StaleGenerationError, match="stale_generation"):
            store.get_committed_window(committed.query)
    finally:
        unsubscribe()
