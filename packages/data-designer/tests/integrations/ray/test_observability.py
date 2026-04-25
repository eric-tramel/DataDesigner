# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from fake_ray_harness import FakeActorHandle

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.integrations.ray import (
    RayDatasetAnalysis,
    RayDatasetCreationResults,
    RayDatasetMetrics,
    RayMetricsError,
    RayThrottleSnapshot,
    RayTraceEvent,
    RayWorkerProfile,
)
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray.observability import normalize_ray_trace_event

pytestmark = pytest.mark.ray_fake


def test_ray_results_load_analysis_returns_profiles_traces_and_throttle(
    tmp_path: Path,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    collector = ray_backend_module._RayMetricsCollector(max_trace_events=1)
    collector.record({"total_rows": 2, "blocks": 1, "elapsed_seconds": 0.5, "block_id": "block-a"})
    collector.record_observability(
        {
            "worker_profile": RayWorkerProfile(
                block_id="block-a",
                total_rows=2,
                columns=["id", "label"],
                column_dtypes={"id": "int64", "label": "object"},
                non_null_counts={"id": 2, "label": 1},
                null_counts={"id": 0, "label": 1},
                memory_usage_bytes=128,
            ).to_dict(),
            "trace_events": [
                RayTraceEvent(
                    block_id="block-a",
                    event_type="block_started",
                    timestamp_seconds=1.0,
                    row_count=2,
                ).to_dict(),
                RayTraceEvent(
                    block_id="block-a",
                    event_type="block_completed",
                    timestamp_seconds=2.0,
                    elapsed_seconds=0.5,
                    row_count=2,
                ).to_dict(),
            ],
            "throttle_snapshots": [
                RayThrottleSnapshot(
                    provider_name="stub-provider",
                    model_id="stub-model",
                    domain="chat",
                    current_limit=2,
                    effective_max=4,
                    rate_limit_ceiling=3,
                    consecutive_rate_limits=1,
                ).to_dict()
            ],
        }
    )
    results = RayDatasetCreationResults(
        dataset=lazy.pd.DataFrame({"id": [1, 2]}),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=0, blocks=1),
        ray=types.SimpleNamespace(get=lambda ref: ref.value),
        metrics_collector=FakeActorHandle(collector),
    )

    analysis = results.load_analysis()

    assert analysis is not None
    assert analysis.total_rows == 2
    assert analysis.blocks == 1
    assert analysis.worker_profiles[0].block_id == "block-a"
    assert analysis.worker_profiles[0].null_counts == {"id": 0, "label": 1}
    assert analysis.trace_events[0].event_type == "block_started"
    assert analysis.trace_events_dropped == 1
    assert analysis.throttle_snapshots[0].rate_limit_ceiling == 3
    assert results.load_worker_metrics()[0].block_id == analysis.worker_profiles[0].block_id

    report_path = tmp_path / "ray-analysis.json"
    analysis.to_report(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["worker_profiles"][0]["block_id"] == "block-a"


def test_ray_results_load_analysis_returns_none_without_observability_payload(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    collector = ray_backend_module._RayMetricsCollector(max_trace_events=0)
    collector.record({"total_rows": 0, "blocks": 1, "elapsed_seconds": 0.25})
    results = RayDatasetCreationResults(
        dataset=lazy.pd.DataFrame({"id": []}),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=0, blocks=1),
        ray=types.SimpleNamespace(get=lambda ref: ref.value),
        metrics_collector=FakeActorHandle(collector),
    )

    assert results.load_analysis() is None


def test_ray_dataset_analysis_to_report_filters_json_sections(tmp_path: Path) -> None:
    analysis = RayDatasetAnalysis(
        total_rows=2,
        blocks=1,
        worker_profiles=[RayWorkerProfile(block_id="block-a", total_rows=2, columns=["label"])],
        trace_events=[RayTraceEvent(block_id="block-a", event_type="block_started", timestamp_seconds=1.0)],
        trace_events_dropped=3,
        throttle_snapshots=[
            RayThrottleSnapshot(provider_name="provider", model_id="model", domain="chat", current_limit=1)
        ],
    )
    report_path = tmp_path / "ray-analysis.json"

    analysis.to_report(report_path, include_sections=["summary", "trace_events"])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "summary": {
            "blocks": 1,
            "column_names": ["label"],
            "failed_blocks": 0,
            "successful_blocks": 1,
            "total_rows": 2,
            "trace_events_dropped": 3,
        },
        "trace_events": [
            {
                "block_id": "block-a",
                "details": None,
                "elapsed_seconds": 0.0,
                "event_type": "block_started",
                "ray_node_id": None,
                "ray_task_id": None,
                "row_count": 0,
                "timestamp_seconds": 1.0,
                "worker_hostname": None,
                "worker_pid": None,
            }
        ],
        "trace_events_dropped": 3,
    }


def test_ray_dataset_analysis_to_report_filters_html_sections(tmp_path: Path) -> None:
    analysis = RayDatasetAnalysis(
        total_rows=1,
        blocks=1,
        worker_profiles=[RayWorkerProfile(block_id="block-a", total_rows=1, columns=["label"])],
    )
    report_path = tmp_path / "ray-analysis.html"

    analysis.to_report(report_path, include_sections=["worker_profiles"])

    report = report_path.read_text(encoding="utf-8")
    assert "&quot;worker_profiles&quot;" in report
    assert "&quot;summary&quot;" not in report
    assert "&quot;trace_events&quot;" not in report


def test_ray_dataset_analysis_to_report_rejects_unknown_include_section(tmp_path: Path) -> None:
    analysis = RayDatasetAnalysis()

    with pytest.raises(RayMetricsError, match="Ray-native sections"):
        analysis.to_report(tmp_path / "ray-analysis.json", include_sections=["column_profilers"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_normalize_ray_trace_event_rejects_non_finite_timestamp_seconds(value: float) -> None:
    with pytest.raises(RayMetricsError, match="timestamp_seconds.*finite"):
        normalize_ray_trace_event(
            {
                "block_id": "block-a",
                "event_type": "block_started",
                "timestamp_seconds": value,
            }
        )


def test_ray_dataset_analysis_rejects_failed_blocks_greater_than_blocks() -> None:
    with pytest.raises(RayMetricsError, match="failed_blocks.*greater than.*blocks"):
        RayDatasetAnalysis(blocks=1, failed_blocks=2)
