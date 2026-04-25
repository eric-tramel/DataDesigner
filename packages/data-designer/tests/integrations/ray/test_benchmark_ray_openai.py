# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.integrations.ray import RayDatasetAnalysis, RayDatasetStats, RayTraceEvent, RayWorkerProfile

pytestmark = pytest.mark.ray_benchmark


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _benchmark_dir() -> Path:
    return _repo_root() / "scripts" / "benchmarks"


def _load_benchmark_module(module_name: str, script_path: Path) -> ModuleType:
    sys.path.insert(0, str(script_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load benchmark module from {script_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_path.parent))


def _load_common_module() -> ModuleType:
    return _load_benchmark_module("ray_benchmark_common", _benchmark_dir() / "ray_benchmark_common.py")


def test_ray_benchmark_scripts_import_with_shared_helpers() -> None:
    benchmark_dir = _benchmark_dir()

    modules = {
        path.stem: _load_benchmark_module(path.stem, path) for path in sorted(benchmark_dir.glob("benchmark_ray_*.py"))
    }

    assert modules["benchmark_ray_openai"].RESULT_PREFIX == "RAY_OPENAI_BENCHMARK_RESULT="
    assert modules["benchmark_ray_mock_provider"].RESULT_PREFIX == "RAY_MOCK_PROVIDER_BENCHMARK_RESULT="
    assert modules["benchmark_ray_streaming_out_of_core"].RESULT_PREFIX == "RAY_STREAMING_OUT_OF_CORE_RESULT="


def test_streaming_benchmark_summarizes_worker_memory_profiles() -> None:
    module = _load_benchmark_module(
        "benchmark_ray_streaming_out_of_core",
        _benchmark_dir() / "benchmark_ray_streaming_out_of_core.py",
    )
    analysis = RayDatasetAnalysis(
        worker_profiles=[
            RayWorkerProfile(
                block_id="block-a",
                total_rows=2,
                input_memory_usage_bytes=100,
                memory_usage_bytes=150,
                process_maxrss_bytes=1024,
            ),
            RayWorkerProfile(
                block_id="block-b",
                total_rows=2,
                input_memory_usage_bytes=200,
                memory_usage_bytes=250,
                process_maxrss_bytes=2048,
            ),
        ]
    )

    summary = module._worker_memory_summary(analysis)

    assert summary["profile_count"] == 2
    assert summary["input_memory_usage_bytes"]["max"] == 200.0
    assert summary["output_memory_usage_bytes"]["total"] == 400.0
    assert summary["process_maxrss_bytes"]["max"] == 2048.0
    assert summary["max_output_to_input_memory_ratio"] == 1.5


def test_parse_backends_supports_full_apples_to_apples_suite() -> None:
    benchmark_common = _load_common_module()

    assert benchmark_common.parse_backends("all") == list(benchmark_common.ALL_BACKENDS)
    assert benchmark_common.parse_backends("local-sync,local-async,ray-dataset,ray-arrow-refs") == [
        "local-sync",
        "local-async",
        "ray-dataset",
        "ray-arrow-refs",
    ]
    assert benchmark_common.parse_backends("local-sync,local-sync,ray-dataset") == ["local-sync", "ray-dataset"]


def test_parse_backends_rejects_unknown_backend() -> None:
    benchmark_common = _load_common_module()

    with pytest.raises(ValueError, match="Unsupported backend"):
        benchmark_common.parse_backends("local,ray")


def test_result_payload_reports_validity_and_provider_counters(tmp_path: Path) -> None:
    benchmark_common = _load_common_module()
    output = lazy.pd.DataFrame(
        {
            "instruction": ["write code", "debug code"],
            "code_implementation": ["print('ok')", "print('debug')"],
        }
    )
    metrics = {
        "total_rows": 2,
        "blocks": 1,
        "elapsed_seconds": 2.0,
        "model_usage": {
            "gpt-4.1": {
                "token_usage": {"input_tokens": 10, "output_tokens": 30, "total_tokens": 40},
                "request_usage": {"successful_requests": 2, "failed_requests": 1, "total_requests": 3},
                "retry_count": 4,
                "rate_limited_requests": 1,
            }
        },
    }

    payload = benchmark_common.result_payload(
        backend="local-sync",
        iteration=1,
        seed=11,
        output=output,
        elapsed_seconds=2.0,
        metrics=metrics,
        artifact_path=tmp_path,
        output_mode="pandas",
        batch_size=2,
        max_parallel_requests=2,
        expected_columns=["instruction", "code_implementation"],
        expected_rows=2,
        include_retry_throttle=True,
    )

    assert payload["validity"] == {
        "row_count_matches": True,
        "expected_columns_present": True,
        "expected_columns_non_null": True,
        "all_output_valid": True,
    }
    assert payload["throughput"]["requests_per_minute"] == 90
    assert payload["throughput"]["tokens_per_second"] == 20
    assert payload["throughput"]["retry_count"] == 4
    assert payload["throughput"]["throttle_count"] == 1


def test_ray_diagnostics_payload_summarizes_dataset_stats_without_raw_text() -> None:
    benchmark_common = _load_common_module()

    analysis = RayDatasetAnalysis(
        total_rows=2,
        blocks=1,
        trace_events=[RayTraceEvent(block_id="block-a", event_type="block_completed", timestamp_seconds=1.0)],
        trace_events_dropped=2,
        ray_dataset_stats=RayDatasetStats(
            stats_text="Operator 1 MapBatches: detailed stats text",
            stats_text_char_count=43,
            operator_diagnostics=["Operator 1 MapBatches: detailed stats text"],
            backpressure_diagnostics=["Backpressure: queued task wait time 0.1s"],
            object_store_diagnostics=["Object store memory: 1 MiB used"],
        ),
        diagnostic_warnings=["some Ray stats were unavailable"],
    )

    payload = benchmark_common.ray_diagnostics_payload(analysis)

    assert payload == {
        "diagnostic_warnings": ["some Ray stats were unavailable"],
        "ray_dataset_stats": {
            "backpressure_diagnostics": ["Backpressure: queued task wait time 0.1s"],
            "backpressure_diagnostics_dropped": 0,
            "object_store_diagnostics": ["Object store memory: 1 MiB used"],
            "object_store_diagnostics_dropped": 0,
            "operator_diagnostics": ["Operator 1 MapBatches: detailed stats text"],
            "operator_diagnostics_dropped": 0,
            "stats_text_available": True,
            "stats_text_char_count": 43,
            "stats_text_truncated": False,
            "warnings": [],
        },
        "throttle_snapshots": 0,
        "throttle_snapshots_dropped": 0,
        "trace_events": 1,
        "trace_events_dropped": 2,
        "worker_profiles": 0,
        "worker_profiles_dropped": 0,
    }


def test_failure_payload_preserves_result_shape() -> None:
    benchmark_common = _load_common_module()

    payload = benchmark_common.failure_payload(
        backend="ray-dataset",
        iteration=2,
        seed=12,
        elapsed_seconds=3.5,
        exc=RuntimeError("backend failed"),
        batch_size=8,
        max_parallel_requests=4,
        expected_columns=["a", "b"],
    )

    assert payload["status"] == "failed"
    assert payload["missing_columns"] == ["a", "b"]
    assert payload["validity"]["all_output_valid"] is False
    assert payload["throughput"]["total_requests"] == 0


def test_summary_groups_iterations_and_speedups() -> None:
    benchmark_common = _load_common_module()
    results = [
        _benchmark_result(backend="local-sync", iteration=1, elapsed_seconds=10.0, rows_per_second=10.0),
        _benchmark_result(backend="local-async", iteration=1, elapsed_seconds=5.0, rows_per_second=20.0),
        _benchmark_result(backend="ray-dataset", iteration=1, elapsed_seconds=4.0, rows_per_second=25.0),
        _benchmark_result(backend="local-sync", iteration=2, elapsed_seconds=8.0, rows_per_second=12.5),
        _benchmark_result(backend="local-async", iteration=2, elapsed_seconds=4.0, rows_per_second=25.0),
        _benchmark_result(backend="ray-dataset", iteration=2, elapsed_seconds=2.0, rows_per_second=50.0),
    ]

    summary = benchmark_common.build_summary(results, include_retry_throttle=True)

    assert summary["all_output_valid"] is True
    assert summary["failed_backends"] == []
    assert summary["per_backend"]["local-sync"]["iterations"] == 2
    assert summary["per_backend"]["local-async"]["rows_per_second"]["mean"] == 22.5
    assert summary["comparisons_vs_local_sync"]["local-async"]["mean"] == 2.0
    assert summary["comparisons_vs_local_sync"]["ray-dataset"]["mean"] == 3.25


def test_summary_flags_row_count_mismatch_as_failed_backend() -> None:
    benchmark_common = _load_common_module()
    result = _benchmark_result(
        backend="ray-dataset",
        iteration=1,
        elapsed_seconds=4.0,
        rows_per_second=25.0,
    )
    result["validity"]["row_count_matches"] = False
    result["validity"]["all_output_valid"] = False
    result["metrics"] = {"failed_blocks": 0}
    result["throughput"]["failed_requests"] = 0

    summary = benchmark_common.build_summary([result], include_retry_throttle=True)

    assert summary["all_output_valid"] is False
    assert summary["failed_backends"] == ["ray-dataset"]


def test_fail_on_invalid_summary_exits_nonzero_unless_allowed() -> None:
    benchmark_common = _load_common_module()
    summary = {"all_output_valid": False, "failed_backends": ["ray-dataset"]}

    with pytest.raises(SystemExit) as exc_info:
        benchmark_common.fail_on_invalid_summary(summary=summary, allow_failures=False)

    assert exc_info.value.code == 1
    assert benchmark_common.fail_on_invalid_summary(summary=summary, allow_failures=True) is None


def test_json_default_serializes_numpy_values() -> None:
    benchmark_common = _load_common_module()

    assert benchmark_common.json_default(lazy.np.array([1, 2])) == [1, 2]
    assert benchmark_common.json_default(lazy.np.int64(3)) == 3


def test_ray_benchmark_scripts_do_not_call_private_data_designer_builders_directly() -> None:
    private_methods = {"_create_resource_provider", "_create_dataset_builder"}
    offenders: list[str] = []
    for script_path in sorted(_benchmark_dir().glob("benchmark_ray_*.py")):
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in private_methods:
                offenders.append(f"{script_path.name}:{node.lineno}:{node.attr}")

    assert offenders == []


def _benchmark_result(
    *,
    backend: str,
    iteration: int,
    elapsed_seconds: float,
    rows_per_second: float,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "iteration": iteration,
        "elapsed_seconds": elapsed_seconds,
        "rows_per_second": rows_per_second,
        "validity": {"all_output_valid": True},
        "throughput": {
            "requests_per_minute": 60.0,
            "tokens_per_second": 120.0,
            "failed_requests": 0,
            "retry_count": 0,
            "throttle_count": 0,
        },
    }
