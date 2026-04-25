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
    RayDatasetStats,
    RayMetricsError,
    RayThrottleSnapshot,
    RayTraceEvent,
    RayWorkerProfile,
)
from data_designer.integrations.ray import observability_collection as ray_observability_collection
from data_designer.integrations.ray.observability import normalize_ray_trace_event, normalize_ray_worker_profile
from data_designer.integrations.ray.results import RayResultArtifacts

pytestmark = pytest.mark.ray_fake


def test_ray_results_load_analysis_returns_profiles_traces_and_throttle(
    tmp_path: Path,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    collector = ray_observability_collection._RayMetricsCollector(max_trace_events=1)
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
    assert analysis.worker_profiles_dropped == 0
    assert analysis.trace_events[0].event_type == "block_started"
    assert analysis.trace_events_dropped == 1
    assert analysis.throttle_snapshots[0].rate_limit_ceiling == 3
    assert analysis.throttle_snapshots_dropped == 0
    assert results.load_worker_metrics()[0].block_id == analysis.worker_profiles[0].block_id

    report_path = tmp_path / "ray-analysis.json"
    analysis.to_report(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["worker_profiles"][0]["block_id"] == "block-a"
    assert report["worker_profiles_dropped"] == 0
    assert report["throttle_snapshots_dropped"] == 0


def test_ray_results_load_analysis_collects_dataset_stats_without_metrics_actor(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    class StatsDataset:
        def __init__(self) -> None:
            self.calls = 0

        def stats(self) -> str:
            self.calls += 1
            return "\n".join(
                [
                    "Operator 1 MapBatches(_RayBatchWorker): 2 tasks executed, 2 blocks produced",
                    "Backpressure: queued task wait time 0.25s",
                    "Object store memory: 10 MiB used, 0 bytes spilled",
                ]
            )

    dataset = StatsDataset()
    results = RayDatasetCreationResults(
        dataset=dataset,
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=4, blocks=2),
    )

    analysis = results.load_analysis()

    assert analysis is not None
    assert analysis.total_rows == 4
    assert analysis.blocks == 2
    assert analysis.ray_dataset_stats is not None
    assert analysis.ray_dataset_stats.stats_text_char_count > 0
    assert (
        "Operator 1 MapBatches(_RayBatchWorker): 2 tasks executed, 2 blocks produced"
        in analysis.ray_dataset_stats.operator_diagnostics
    )
    assert analysis.ray_dataset_stats.backpressure_diagnostics == ["Backpressure: queued task wait time 0.25s"]
    assert analysis.ray_dataset_stats.object_store_diagnostics == ["Object store memory: 10 MiB used, 0 bytes spilled"]
    assert results.load_analysis() is analysis
    assert dataset.calls == 1


def test_ray_results_load_analysis_preserves_dataset_stats_failure_as_warning(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    class FailingStatsDataset:
        def stats(self) -> str:
            raise RuntimeError("stats unavailable")

    results = RayDatasetCreationResults(
        dataset=FailingStatsDataset(),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=1, blocks=1),
    )

    analysis = results.load_analysis()

    assert analysis is not None
    assert analysis.ray_dataset_stats is not None
    assert analysis.ray_dataset_stats.stats_text is None
    assert analysis.ray_dataset_stats.warnings == ["Ray Dataset.stats() failed: RuntimeError: stats unavailable"]


def test_collect_ray_dataset_stats_bounds_raw_text_and_diagnostics() -> None:
    class LongStatsDataset:
        def stats(self) -> str:
            long_operator_line = f"Operator {'x' * 600}"
            return "\n".join([long_operator_line for _ in range(105)] + ["tail" * 1000])

    dataset_stats = ray_observability_collection.collect_ray_dataset_stats(LongStatsDataset())

    assert dataset_stats is not None
    assert dataset_stats.stats_text_truncated is True
    assert dataset_stats.stats_text is not None
    assert len(dataset_stats.stats_text) == 65_536
    assert len(dataset_stats.operator_diagnostics) == 100
    assert dataset_stats.operator_diagnostics_dropped == 5
    assert dataset_stats.operator_diagnostics[0].endswith("... [truncated]")


def test_assemble_ray_dataset_analysis_skips_malformed_diagnostics() -> None:
    analysis = ray_observability_collection.assemble_ray_dataset_analysis(
        RayDatasetMetrics(total_rows=1, blocks=1),
        {
            "worker_profiles": [{"block_id": "block-a", "total_rows": 1}, {"total_rows": 1}],
            "trace_events": "not-a-list",
            "throttle_snapshots": [
                {
                    "provider_name": "provider",
                    "model_id": "model",
                    "domain": "chat",
                    "current_limit": 1,
                },
                {"provider_name": 7},
            ],
            "trace_events_dropped": "bad-count",
        },
    )

    assert analysis is not None
    assert [profile.block_id for profile in analysis.worker_profiles] == ["block-a"]
    assert analysis.worker_profiles_dropped == 1
    assert analysis.trace_events == []
    assert analysis.trace_events_dropped == 1
    assert [snapshot.domain for snapshot in analysis.throttle_snapshots] == ["chat"]
    assert analysis.throttle_snapshots_dropped == 1
    assert any("Skipped Ray observability worker profile" in warning for warning in analysis.diagnostic_warnings)
    assert any("trace event payload must be a list" in warning for warning in analysis.diagnostic_warnings)
    assert any("trace_events_dropped" in warning for warning in analysis.diagnostic_warnings)


def test_ray_metrics_collector_bounds_profiles_and_throttle_snapshots() -> None:
    collector = ray_observability_collection._RayMetricsCollector(
        max_trace_events=1,
        max_worker_profiles=1,
        max_throttle_snapshots=1,
    )

    collector.record_observability(
        {
            "worker_profile": RayWorkerProfile(block_id="block-a", total_rows=1).to_dict(),
            "trace_events": [
                RayTraceEvent(block_id="block-a", event_type="block_started", timestamp_seconds=1.0).to_dict(),
                RayTraceEvent(block_id="block-a", event_type="block_completed", timestamp_seconds=2.0).to_dict(),
            ],
            "throttle_snapshots": [
                RayThrottleSnapshot(
                    provider_name="provider",
                    model_id="model",
                    domain="chat",
                    current_limit=1,
                ).to_dict(),
                RayThrottleSnapshot(
                    provider_name="provider",
                    model_id="model",
                    domain="embedding",
                    current_limit=2,
                ).to_dict(),
            ],
        }
    )
    collector.record_observability(
        {
            "worker_profile": RayWorkerProfile(block_id="block-b", total_rows=1).to_dict(),
            "trace_events": [
                RayTraceEvent(block_id="block-b", event_type="block_started", timestamp_seconds=3.0).to_dict()
            ],
            "throttle_snapshots": [
                RayThrottleSnapshot(
                    provider_name="provider",
                    model_id="model",
                    domain="chat",
                    current_limit=3,
                ).to_dict()
            ],
        }
    )

    snapshot = collector.observability_snapshot()
    assert [profile["block_id"] for profile in snapshot["worker_profiles"]] == ["block-a"]
    assert snapshot["worker_profiles_dropped"] == 1
    assert [event["block_id"] for event in snapshot["trace_events"]] == ["block-a"]
    assert snapshot["trace_events_dropped"] == 2
    assert [throttle_snapshot["domain"] for throttle_snapshot in snapshot["throttle_snapshots"]] == ["chat"]
    assert snapshot["throttle_snapshots_dropped"] == 2


def test_assemble_ray_dataset_analysis_reports_dropped_observability_counts() -> None:
    analysis = ray_observability_collection.assemble_ray_dataset_analysis(
        RayDatasetMetrics(total_rows=2, blocks=2),
        {
            "worker_profiles_dropped": 2,
            "trace_events_dropped": 3,
            "throttle_snapshots_dropped": 4,
        },
    )

    assert analysis is not None
    assert analysis.worker_profiles == []
    assert analysis.worker_profiles_dropped == 2
    assert analysis.trace_events == []
    assert analysis.trace_events_dropped == 3
    assert analysis.throttle_snapshots == []
    assert analysis.throttle_snapshots_dropped == 4


def test_ray_results_load_analysis_returns_none_without_observability_payload(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    collector = ray_observability_collection._RayMetricsCollector(max_trace_events=0)
    collector.record({"total_rows": 0, "blocks": 1, "elapsed_seconds": 0.25})
    results = RayDatasetCreationResults(
        dataset=lazy.pd.DataFrame({"id": []}),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=0, blocks=1),
        ray=types.SimpleNamespace(get=lambda ref: ref.value),
        metrics_collector=FakeActorHandle(collector),
    )

    assert results.load_analysis() is None


def test_assemble_ray_dataset_analysis_returns_none_without_observability_payload() -> None:
    assert ray_observability_collection.assemble_ray_dataset_analysis(RayDatasetMetrics(blocks=1), {}) is None


def test_ray_result_artifacts_load_observability_without_public_results() -> None:
    collector = ray_observability_collection._RayMetricsCollector(max_trace_events=2)
    collector.record({"total_rows": 1, "blocks": 1, "elapsed_seconds": 0.5, "block_id": "block-a"})
    collector.record_observability(
        {
            "worker_profile": RayWorkerProfile(block_id="block-a", total_rows=1, columns=["id"]).to_dict(),
            "trace_events": [
                RayTraceEvent(
                    block_id="block-a",
                    event_type="block_completed",
                    timestamp_seconds=2.0,
                    elapsed_seconds=0.5,
                    row_count=1,
                ).to_dict()
            ],
        }
    )
    artifacts = RayResultArtifacts(
        dataset=lazy.pd.DataFrame({"id": [1]}),
        metrics=RayDatasetMetrics(total_rows=0, blocks=1, elapsed_seconds=1.0),
        ray=types.SimpleNamespace(get=lambda ref: ref.value),
        metrics_collector=FakeActorHandle(collector),
    )

    analysis = artifacts.load_observability()

    assert analysis is not None
    assert analysis.total_rows == 1
    assert analysis.blocks == 1
    assert analysis.worker_profiles[0].block_id == "block-a"
    assert analysis.trace_events[0].event_type == "block_completed"


def test_ray_dataset_analysis_to_report_filters_json_sections(tmp_path: Path) -> None:
    analysis = RayDatasetAnalysis(
        total_rows=2,
        blocks=1,
        worker_profiles=[RayWorkerProfile(block_id="block-a", total_rows=2, columns=["label"])],
        worker_profiles_dropped=4,
        trace_events=[RayTraceEvent(block_id="block-a", event_type="block_started", timestamp_seconds=1.0)],
        trace_events_dropped=3,
        throttle_snapshots=[
            RayThrottleSnapshot(provider_name="provider", model_id="model", domain="chat", current_limit=1)
        ],
        throttle_snapshots_dropped=5,
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
            "throttle_snapshots_dropped": 5,
            "total_rows": 2,
            "trace_events_dropped": 3,
            "worker_profiles_dropped": 4,
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


def test_ray_dataset_analysis_to_report_filters_dataset_stats_section(tmp_path: Path) -> None:
    analysis = RayDatasetAnalysis(
        total_rows=1,
        blocks=1,
        ray_dataset_stats=RayDatasetStats(
            stats_text="Operator 1 MapBatches: 1 blocks",
            stats_text_char_count=31,
            operator_diagnostics=["Operator 1 MapBatches: 1 blocks"],
        ),
        diagnostic_warnings=["stats partially unavailable"],
    )
    report_path = tmp_path / "ray-analysis.json"

    analysis.to_report(report_path, include_sections=["ray_dataset_stats"])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == {
        "diagnostic_warnings": ["stats partially unavailable"],
        "ray_dataset_stats": {
            "backpressure_diagnostics": [],
            "backpressure_diagnostics_dropped": 0,
            "object_store_diagnostics": [],
            "object_store_diagnostics_dropped": 0,
            "operator_diagnostics": ["Operator 1 MapBatches: 1 blocks"],
            "operator_diagnostics_dropped": 0,
            "stats_text": "Operator 1 MapBatches: 1 blocks",
            "stats_text_char_count": 31,
            "stats_text_truncated": False,
            "warnings": [],
        },
    }


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


def test_normalize_ray_trace_event_rejects_invalid_optional_string() -> None:
    with pytest.raises(RayMetricsError, match="worker_hostname.*string"):
        normalize_ray_trace_event(
            {
                "block_id": "block-a",
                "event_type": "block_started",
                "timestamp_seconds": 1.0,
                "worker_hostname": 123,
            }
        )


def test_normalize_ray_worker_profile_preserves_engine_model_usage_optional_fields() -> None:
    payload = {
        "block_id": "block-a",
        "model_usage": {
            "model-a": {
                "token_usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                "request_usage": {"successful_requests": 1, "failed_requests": 0, "total_requests": 1},
                "tool_usage": {
                    "total_tool_calls": 2,
                    "total_tool_call_turns": 1,
                    "total_generations": 1,
                    "generations_with_tools": 1,
                },
                "image_usage": {"total_images": 4},
                "tokens_per_second": 3,
                "requests_per_minute": 60,
            }
        },
    }

    profile = normalize_ray_worker_profile(payload)

    assert profile.model_usage == payload["model_usage"]
    assert profile.model_usage is not payload["model_usage"]
    assert profile.model_usage["model-a"]["image_usage"] is not payload["model_usage"]["model-a"]["image_usage"]


@pytest.mark.parametrize("model_usage", [[], {"model-a": []}, {1: {}}])
def test_normalize_ray_worker_profile_rejects_invalid_model_usage(model_usage: object) -> None:
    with pytest.raises(RayMetricsError, match="model_usage|model usage"):
        normalize_ray_worker_profile({"block_id": "block-a", "model_usage": model_usage})


def test_ray_dataset_analysis_rejects_failed_blocks_greater_than_blocks() -> None:
    with pytest.raises(RayMetricsError, match="failed_blocks.*greater than.*blocks"):
        RayDatasetAnalysis(blocks=1, failed_blocks=2)
