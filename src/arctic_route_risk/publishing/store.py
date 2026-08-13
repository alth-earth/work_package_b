"""Persistent immutable RiskSource with generation fencing and window commits."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from arctic_route_planning.contracts import (
    CommittedRiskWindow,
    ProvenanceKind,
    RiskFrame,
    RiskWindowQuery,
    canonical_risk_frame_bytes,
    is_canonical_risk_id,
    risk_frame_content_digest,
    risk_frame_from_document,
    validate_canonical_risk_id,
)
from arctic_route_planning.errors import ContextMismatchError

from arctic_route_risk.errors import (
    PublicationConflictError,
    RiskPipelineError,
    StaleGenerationError,
)


class _GenerationSnapshot(Protocol):
    generation_id: int


class _GenerationAuthority(Protocol):
    def snapshot(self) -> _GenerationSnapshot: ...

    def subscribe_seek(
        self,
        listener: Callable[[_GenerationSnapshot], None],
    ) -> Callable[[], None]: ...


class PersistentRiskStore:
    """Filesystem-backed C ``RiskSource`` and ``CommittedRiskSource``.

    Frame documents and commits are content addressed and immutable. Each run
    has a shared/exclusive generation fence: executions and publications hold a
    shared lease, while generation activation takes the exclusive side. A
    separate store-wide write lock is held only while atomically updating the
    generation map or publishing immutable artifacts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._frames = self.root / "frames"
        self._commits = self.root / "commits"
        self._queries = self.root / "queries"
        self._state = self.root / "state"
        self._run_locks = self._state / "run-locks"
        for directory in (
            self._frames,
            self._commits,
            self._queries,
            self._state,
            self._run_locks,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._state / "store.lock"
        self._generation_path = self._state / "active-generations.json"

    def activate_generation(self, run_id: str, generation_id: int) -> None:
        _validate_generation(run_id, generation_id)
        # Lock order is always run fence first, then the short store write lock.
        # The exclusive run fence waits for every execution lease of this run,
        # without waiting for leases belonging to unrelated runs.
        with self._run_fence(run_id, exclusive=True), self._store_write_locked():
            state = self._read_generation_state()
            current = state.get(run_id)
            if current is not None and generation_id < current:
                raise StaleGenerationError(
                    f"stale_generation: run {run_id} is already at generation {current}"
                )
            if current == generation_id:
                return
            state[run_id] = generation_id
            _atomic_replace(self._generation_path, _canonical_json_bytes(state))

    def reset_to_generation(self, generation_id: int, *, run_id: str) -> None:
        """RiskSource-compatible fence operation without deleting history."""

        self.activate_generation(run_id, generation_id)

    def bind_generation_authority(
        self,
        run_id: str,
        authority: _GenerationAuthority,
    ) -> Callable[[], None]:
        """Bind a run fence to an orchestrator-owned public simulation clock.

        Formal integration should use this boundary instead of deriving the
        active generation from a previously returned PreparedWindow. The
        subscription is installed before the initial snapshot is activated;
        a small setup lock serializes a concurrent seek with that initial read.
        The returned callback removes the listener.
        """

        if not callable(getattr(authority, "snapshot", None)) or not callable(
            getattr(authority, "subscribe_seek", None)
        ):
            raise TypeError("generation authority must expose snapshot/subscribe_seek")
        setup_lock = RLock()

        def activate(snapshot: _GenerationSnapshot) -> None:
            with setup_lock:
                self.activate_generation(run_id, snapshot.generation_id)

        with setup_lock:
            unsubscribe = authority.subscribe_seek(activate)
            try:
                activate(authority.snapshot())
            except Exception:
                unsubscribe()
                raise
        return unsubscribe

    def publish(self, frame: RiskFrame) -> None:
        """Persist one immutable frame; formal C should still request a committed window."""

        snapshot = _private_frame_snapshot(frame)
        with self._run_fence(
            snapshot.run_id, exclusive=False
        ), self._store_write_locked():
            self._assert_active(snapshot.run_id, snapshot.generation_id)
            self._write_frame(snapshot)

    def publish_window(
        self,
        frames: Sequence[RiskFrame],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval_minutes: int = 60,
    ) -> CommittedRiskWindow:
        snapshots = tuple(_private_frame_snapshot(frame) for frame in frames)
        if not snapshots:
            raise RiskPipelineError("forecast_coverage_insufficient: cannot commit empty window")
        if interval_minutes != 60:
            raise RiskPipelineError("formal committed window interval must be exactly 60 minutes")
        ordered = tuple(sorted(snapshots, key=lambda frame: frame.valid_time))
        first = ordered[0]
        query = RiskWindowQuery(
            start=start or first.valid_time,
            end=end or ordered[-1].valid_time,
            interval=timedelta(minutes=interval_minutes),
            run_id=first.run_id,
            scenario_id=first.scenario_id,
            corridor_id=first.corridor_id,
            generation_id=first.generation_id,
            vessel_profile_id=first.vessel_profile_id,
            config_digest=first.config_digest,
            model_config_digest=first.model_config_digest,
            as_of=first.as_of_time,
        )
        committed = CommittedRiskWindow.create(query, ordered)
        # Everything below is derived exclusively from the private canonical
        # snapshots. Mutating the caller's inspectable xarray objects after this
        # point cannot split frame bytes, manifest digests, and the query pointer.
        manifest = _commit_document(committed)
        manifest_bytes = _canonical_json_bytes(manifest)
        pointer = {
            "schema_version": "b.risk-window-query-pointer.v1",
            "query_digest": _query_digest(query),
            "commit_id": committed.commit_id,
            "content_digest": committed.content_digest,
        }
        pointer_bytes = _canonical_json_bytes(pointer)
        with self._run_fence(
            query.run_id, exclusive=False
        ), self._store_write_locked():
            self._assert_active(query.run_id, query.generation_id)
            for frame in committed.frames:
                self._write_frame(frame)
            _write_once(self._commits / f"{committed.commit_id}.json", manifest_bytes)
            _write_once(
                self._queries / f"{pointer['query_digest']}.json",
                pointer_bytes,
            )
        return committed

    def get_committed_window(self, query: RiskWindowQuery) -> CommittedRiskWindow:
        with self._run_fence(query.run_id, exclusive=False):
            return self._get_committed_window_locked(query)

    @contextmanager
    def lease_committed_window(
        self, query: RiskWindowQuery
    ) -> Iterator[CommittedRiskWindow]:
        """Hold the generation fence from exact commit read through C execution.

        Leases for the same or different runs may coexist. Generation activation
        takes the exclusive side of this run's fence, so a seek for this run waits
        until every caller exits while unrelated runs continue independently.
        """

        with self._run_fence(query.run_id, exclusive=False):
            window = self._get_committed_window_locked(query)
            yield window

    def get_window(
        self,
        start: datetime,
        end: datetime,
        *,
        run_id: str,
        scenario_id: str,
        corridor_id: str,
        generation_id: int,
        vessel_profile_id: str,
        config_digest: str,
        model_config_digest: str,
        as_of: datetime,
    ) -> tuple[RiskFrame, ...]:
        start = _require_utc(start, field="start")
        end = _require_utc(end, field="end")
        as_of = _require_utc(as_of, field="as_of")
        if end < start:
            raise RiskPipelineError("risk window end cannot be earlier than start")
        candidates: list[RiskFrame] = []
        wrong_corridor = False
        with self._run_fence(run_id, exclusive=False):
            self._assert_active(run_id, generation_id)
            for path in self._frames.glob("risk-sha256-*.json"):
                frame = self._read_frame(path.stem)
                same_context = (
                    frame.run_id == run_id
                    and frame.scenario_id == scenario_id
                    and frame.generation_id == generation_id
                    and frame.vessel_profile_id == vessel_profile_id
                    and frame.config_digest == config_digest
                    and frame.model_config_digest == model_config_digest
                )
                wrong_corridor |= same_context and frame.corridor_id != corridor_id
                if (
                    same_context
                    and frame.corridor_id == corridor_id
                    and frame.as_of_time <= as_of
                    and start <= frame.valid_time <= end
                ):
                    candidates.append(frame)
        if wrong_corridor:
            raise ContextMismatchError("请求 corridor_id 与该运行已发布的 RiskFrame 不一致")
        by_time: dict[datetime, RiskFrame] = {}
        for frame in candidates:
            current = by_time.get(frame.valid_time)
            if current is None or (frame.as_of_time, frame.generated_at, frame.risk_id) > (
                current.as_of_time,
                current.generated_at,
                current.risk_id,
            ):
                by_time[frame.valid_time] = frame
        return tuple(by_time[key] for key in sorted(by_time))

    def latest_before(
        self,
        target: datetime,
        *,
        run_id: str,
        scenario_id: str,
        corridor_id: str,
        generation_id: int,
        vessel_profile_id: str,
        config_digest: str,
        model_config_digest: str,
        as_of: datetime,
    ) -> RiskFrame | None:
        target = _require_utc(target, field="target")
        frames = self.get_window(
            datetime.min.replace(tzinfo=UTC),
            target,
            run_id=run_id,
            scenario_id=scenario_id,
            corridor_id=corridor_id,
            generation_id=generation_id,
            vessel_profile_id=vessel_profile_id,
            config_digest=config_digest,
            model_config_digest=model_config_digest,
            as_of=as_of,
        )
        return frames[-1] if frames else None

    def _assert_active(self, run_id: str, generation_id: int) -> None:
        state = self._read_generation_state()
        current = state.get(run_id)
        if current is None:
            raise StaleGenerationError(
                f"stale_generation: run {run_id} has no activated generation"
            )
        if current != generation_id:
            raise StaleGenerationError(
                f"stale_generation: active={current}, attempted={generation_id}"
            )

    def _get_committed_window_locked(
        self, query: RiskWindowQuery
    ) -> CommittedRiskWindow:
        self._assert_active(query.run_id, query.generation_id)
        query_digest = _query_digest(query)
        pointer_path = self._queries / f"{query_digest}.json"
        if not pointer_path.exists():
            raise ContextMismatchError("BC 中没有完全匹配该查询的已提交风险窗口")
        pointer = _read_json(pointer_path)
        if set(pointer) != {
            "schema_version",
            "query_digest",
            "commit_id",
            "content_digest",
        }:
            raise PublicationConflictError("committed query pointer fields are invalid")
        if pointer.get("query_digest") != query_digest:
            raise PublicationConflictError("committed query pointer digest mismatch")
        commit_id = pointer.get("commit_id")
        if not isinstance(commit_id, str) or _COMMIT_ID.fullmatch(commit_id) is None:
            raise PublicationConflictError("committed pointer has invalid commit_id")
        manifest_path = self._commits / f"{commit_id}.json"
        manifest = _read_json(manifest_path)
        if manifest.get("content_digest") != pointer.get("content_digest"):
            raise PublicationConflictError("committed pointer and manifest disagree")
        raw_frames = manifest.get("frames")
        if not isinstance(raw_frames, list):
            raise PublicationConflictError("committed manifest frames are invalid")
        frame_ids: list[str] = []
        for item in raw_frames:
            if not isinstance(item, dict) or set(item) != {"risk_id", "content_digest"}:
                raise PublicationConflictError("committed manifest frame entry is invalid")
            risk_id = item.get("risk_id")
            if not isinstance(risk_id, str) or not is_canonical_risk_id(risk_id):
                raise PublicationConflictError("committed manifest has invalid risk_id")
            frame_ids.append(risk_id)
        frames = tuple(self._read_frame(risk_id) for risk_id in frame_ids)
        committed = CommittedRiskWindow.create(query, frames)
        expected = _commit_document(committed)
        if expected != manifest:
            raise PublicationConflictError("committed window manifest verification failed")
        committed.assert_matches(query)
        return committed

    def _write_frame(self, frame: RiskFrame) -> None:
        if frame.provenance is not ProvenanceKind.FORMAL:
            raise PublicationConflictError(
                "PersistentRiskStore only accepts formal RiskFrame publications"
            )
        if not is_canonical_risk_id(frame.risk_id):
            raise PublicationConflictError("formal publication has invalid canonical risk_id")
        validate_canonical_risk_id(frame)
        payload = canonical_risk_frame_bytes(frame)
        _write_once(self._frames / f"{frame.risk_id}.json", payload)

    def _read_frame(self, risk_id: str) -> RiskFrame:
        if not is_canonical_risk_id(risk_id):
            raise PublicationConflictError("invalid canonical risk_id")
        document = _read_json(self._frames / f"{risk_id}.json")
        frame = risk_frame_from_document(document)
        if frame.risk_id != risk_id:
            raise PublicationConflictError("frame path and canonical risk_id disagree")
        return frame

    def _read_generation_state(self) -> dict[str, int]:
        if not self._generation_path.exists():
            return {}
        value = _read_json(self._generation_path)
        if not isinstance(value, dict) or any(
            not isinstance(key, str)
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for key, item in value.items()
        ):
            raise PublicationConflictError("active generation state is invalid")
        return value

    @contextmanager
    def _run_fence(self, run_id: str, *, exclusive: bool) -> Iterator[None]:
        _validate_run_id(run_id)
        lock_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        lock_path = self._run_locks / f"{lock_digest}.lock"
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _store_write_locked(self) -> Iterator[None]:
        with self._lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _commit_document(window: CommittedRiskWindow) -> dict[str, Any]:
    return {
        "schema_version": window.schema_version,
        "commit_id": window.commit_id,
        "content_digest": window.content_digest,
        "start": _iso_z(window.start),
        "end": _iso_z(window.end),
        "interval_seconds": int(window.interval.total_seconds()),
        "count": window.count,
        "run_id": window.run_id,
        "scenario_id": window.scenario_id,
        "corridor_id": window.corridor_id,
        "generation_id": window.generation_id,
        "vessel_profile_id": window.vessel_profile_id,
        "config_digest": window.config_digest,
        "model_config_digest": window.model_config_digest,
        "as_of": _iso_z(window.as_of),
        "frames": [
            {
                "risk_id": frame.risk_id,
                "content_digest": risk_frame_content_digest(frame),
            }
            for frame in window.frames
        ],
    }


def _query_digest(query: RiskWindowQuery) -> str:
    document = {
        "schema_version": "b.risk-window-query.v1",
        "start": _iso_z(query.start),
        "end": _iso_z(query.end),
        "interval_seconds": int(query.interval.total_seconds()),
        "run_id": query.run_id,
        "scenario_id": query.scenario_id,
        "corridor_id": query.corridor_id,
        "generation_id": query.generation_id,
        "vessel_profile_id": query.vessel_profile_id,
        "config_digest": query.config_digest,
        "model_config_digest": query.model_config_digest,
        "as_of": _iso_z(query.as_of),
    }
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _private_frame_snapshot(frame: RiskFrame) -> RiskFrame:
    """Detach publication from every caller-owned xarray container alias."""

    encoded = canonical_risk_frame_bytes(frame)
    try:
        document = json.loads(encoded)
    except json.JSONDecodeError as exc:  # canonical bytes are internal, fail closed if corrupted
        raise PublicationConflictError("canonical RiskFrame snapshot is invalid JSON") from exc
    return risk_frame_from_document(document)


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise RiskPipelineError("run_id cannot be empty")


def _validate_generation(run_id: str, generation_id: int) -> None:
    _validate_run_id(run_id)
    if (
        isinstance(generation_id, bool)
        or not isinstance(generation_id, int)
        or generation_id < 0
    ):
        raise RiskPipelineError("generation_id must be a non-negative integer")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PublicationConflictError(f"missing or invalid immutable artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PublicationConflictError(f"immutable artifact is not an object: {path}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise PublicationConflictError(
                f"immutable ID already has different content: {path.name}"
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
            os.link(temporary_name, path)
        except FileExistsError as exc:
            if path.read_bytes() != payload:
                raise PublicationConflictError(
                    f"immutable ID raced with different content: {path.name}"
                ) from exc
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_replace(path: Path, payload: bytes) -> None:
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
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RiskPipelineError(f"{field} must use UTC")
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return _require_utc(value, field="time").isoformat().replace("+00:00", "Z")


_COMMIT_ID = re.compile(r"^risk-window-sha256-[0-9a-f]{64}$")
