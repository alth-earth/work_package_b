from __future__ import annotations

from datetime import UTC, datetime

import pytest
from arctic_route_planning.contracts import CommittedRiskSource, CommittedRiskWindow

from arctic_route_risk import (
    BInputEnvelope,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
)


@pytest.mark.integration
def test_b_store_satisfies_c_committed_source_protocol(tmp_path, formal_fixture) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    frames = RiskBuildService(
        utc_now=lambda: datetime(2026, 8, 2, tzinfo=UTC)
    ).build_window(RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox))
    store = PersistentRiskStore(tmp_path)
    store.activate_generation(frames[0].run_id, frames[0].generation_id)
    committed = store.publish_window(frames)

    assert isinstance(store, CommittedRiskSource)
    assert isinstance(store.get_committed_window(committed.query), CommittedRiskWindow)
