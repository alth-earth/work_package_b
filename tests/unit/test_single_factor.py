from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from arctic_route_risk import (
    BInputEnvelope,
    RiskBuildRequest,
    SingleFactorOutputPaths,
    build_realtime_single_factor_layers,
    write_realtime_single_factor_outputs,
)

GENERATED = datetime(2026, 8, 2, tzinfo=UTC)


def _request(formal_fixture) -> RiskBuildRequest:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    return RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)


def test_builds_realtime_single_factor_layers_from_latest_visible_data(
    formal_fixture,
) -> None:
    request = _request(formal_fixture)

    layers = build_realtime_single_factor_layers(request, utc_now=lambda: GENERATED)

    assert len(layers) == len(request.model_config.components)
    assert {layer.factor_id for layer in layers} == {
        component.component_id for component in request.model_config.components
    }
    for layer in layers:
        assert layer.valid_time <= request.envelope.knowledge_as_of
        assert layer.issue_time <= request.envelope.knowledge_as_of
        assert layer.collect_time == GENERATED
        assert layer.dataset.attrs["risk_role"] == "realtime_situation_awareness"
        assert layer.dataset.attrs["planning_contract"] == "non_authoritative_display_layer"
        assert layer.source_data_ids
        risk = layer.dataset["risk_score"].values
        finite = np.isfinite(risk)
        assert finite.any()
        assert float(np.nanmin(risk)) >= 0.0
        assert float(np.nanmax(risk)) <= 1.0
        assert set(np.unique(layer.dataset["risk_level"].values)) <= {0, 1, 2, 3, 4, 5}


def test_writes_single_factor_json_outputs_without_optional_plot_or_netcdf_dependencies(
    formal_fixture,
    tmp_path,
) -> None:
    request = _request(formal_fixture)

    output = write_realtime_single_factor_outputs(
        request,
        tmp_path,
        utc_now=lambda: GENERATED,
        write_netcdf=False,
        write_png=False,
    )

    assert isinstance(output, SingleFactorOutputPaths)
    assert output.summary_json.exists()
    assert not output.factor_netcdf
    assert not output.factor_png
    assert len(output.factor_json) == len(request.model_config.components)
    document = json.loads(output.summary_json.read_text(encoding="utf-8"))
    assert document["schema_version"] == "b.single-factor-risk-batch.v1"
    assert document["layer_count"] == len(request.model_config.components)
    assert all(item["valid_time"] for item in document["layers"])
