# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy

pytestmark = pytest.mark.ray_benchmark


def _load_benchmark_module() -> Any:
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "scripts" / "benchmarks" / "benchmark_ray_openai.py"
    spec = importlib.util.spec_from_file_location("benchmark_ray_openai", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark script from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_backends_supports_full_apples_to_apples_suite() -> None:
    benchmark = _load_benchmark_module()

    assert benchmark._parse_backends("all") == list(benchmark.ALL_BACKENDS)
    assert benchmark._parse_backends("local-sync,local-async,ray-dataset,ray-arrow-refs") == [
        "local-sync",
        "local-async",
        "ray-dataset",
        "ray-arrow-refs",
    ]
    assert benchmark._parse_backends("local-sync,local-sync,ray-dataset") == ["local-sync", "ray-dataset"]


def test_parse_backends_rejects_unknown_backend() -> None:
    benchmark = _load_benchmark_module()

    with pytest.raises(ValueError, match="Unsupported backend"):
        benchmark._parse_backends("local,ray")


def test_result_payload_reports_validity_and_provider_counters(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
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

    payload = benchmark._result_payload(
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


def test_summary_groups_iterations_and_speedups() -> None:
    benchmark = _load_benchmark_module()
    results = [
        _benchmark_result(backend="local-sync", iteration=1, elapsed_seconds=10.0, rows_per_second=10.0),
        _benchmark_result(backend="local-async", iteration=1, elapsed_seconds=5.0, rows_per_second=20.0),
        _benchmark_result(backend="ray-dataset", iteration=1, elapsed_seconds=4.0, rows_per_second=25.0),
        _benchmark_result(backend="local-sync", iteration=2, elapsed_seconds=8.0, rows_per_second=12.5),
        _benchmark_result(backend="local-async", iteration=2, elapsed_seconds=4.0, rows_per_second=25.0),
        _benchmark_result(backend="ray-dataset", iteration=2, elapsed_seconds=2.0, rows_per_second=50.0),
    ]

    summary = benchmark._build_summary(results)

    assert summary["all_output_valid"] is True
    assert summary["failed_backends"] == []
    assert summary["per_backend"]["local-sync"]["iterations"] == 2
    assert summary["per_backend"]["local-async"]["rows_per_second"]["mean"] == 22.5
    assert summary["comparisons_vs_local_sync"]["local-async"]["mean"] == 2.0
    assert summary["comparisons_vs_local_sync"]["ray-dataset"]["mean"] == 3.25


def test_summary_flags_row_count_mismatch_as_failed_backend() -> None:
    benchmark = _load_benchmark_module()
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

    summary = benchmark._build_summary([result])

    assert summary["all_output_valid"] is False
    assert summary["failed_backends"] == ["ray-dataset"]


def test_fail_on_invalid_summary_exits_nonzero_unless_allowed() -> None:
    benchmark = _load_benchmark_module()
    summary = {"all_output_valid": False, "failed_backends": ["ray-dataset"]}

    with pytest.raises(SystemExit) as exc_info:
        benchmark._fail_on_invalid_summary(summary=summary, allow_failures=False)

    assert exc_info.value.code == 1
    assert benchmark._fail_on_invalid_summary(summary=summary, allow_failures=True) is None


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
