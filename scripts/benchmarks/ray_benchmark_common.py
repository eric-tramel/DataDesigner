# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Ray benchmark scripts.

The local sync/async benchmark baselines intentionally use DataDesigner private
builder construction methods so they can time the same compiled config path
without the public facade's result packaging. Keep that benchmark-only coupling
centralized here; individual ``benchmark_ray_*.py`` scripts should call these
helpers instead of private ``DataDesigner`` methods directly.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

BackendName = Literal["local-sync", "local-async", "ray-dataset", "ray-arrow-refs"]
RayOutputMode = Literal["dataset", "arrow_refs"]
SummaryExtrasBuilder = Callable[[list[dict[str, Any]]], dict[str, Any]]

ALL_BACKENDS: tuple[BackendName, ...] = ("local-sync", "local-async", "ray-dataset", "ray-arrow-refs")


@dataclass(frozen=True)
class MetricStats:
    mean: float
    stdev: float
    minimum: float
    maximum: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
            "n": self.n,
        }


def parse_backends(value: str, *, all_backends: tuple[BackendName, ...] = ALL_BACKENDS) -> list[BackendName]:
    if value.strip() == "all":
        return list(all_backends)

    backends: list[BackendName] = []
    seen: set[str] = set()
    for raw_backend in value.split(","):
        backend = raw_backend.strip()
        if backend not in all_backends:
            raise ValueError(f"Unsupported backend {backend!r}. Expected one of {', '.join(all_backends)} or all.")
        if backend not in seen:
            backends.append(backend)  # type: ignore[arg-type]
            seen.add(backend)
    if not backends:
        raise ValueError("At least one backend must be selected.")
    return backends


def run_local_preview_benchmark(
    *,
    backend: BackendName,
    dataset_name: str,
    use_async: bool,
    iteration: int,
    seed: int,
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    batch_size: int,
    max_parallel_requests: int,
    artifact_path: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    expected_columns: list[str],
    manage_async_engine_env: bool = False,
    seed_python_random: bool = False,
    include_retry_throttle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed, seed_python_random=seed_python_random)
    context = async_engine_env(enabled=use_async) if manage_async_engine_env else contextlib.nullcontext()
    with context:
        designer = DataDesigner(
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
        )
        designer.set_run_config(run_config(batch_size))

        resource_provider = designer._create_resource_provider(dataset_name, config_builder)
        builder = designer._create_dataset_builder(config_builder.build(), resource_provider, use_async=use_async)

        start = time.perf_counter()
        raw_output = builder.build_preview(num_records=num_records)
        output = builder.process_preview(raw_output)
        elapsed_seconds = time.perf_counter() - start
        metrics = {
            "total_rows": len(output),
            "blocks": math.ceil(num_records / batch_size),
            "failed_blocks": 0,
            "elapsed_seconds": elapsed_seconds,
            "model_usage": resource_provider.model_registry.get_model_usage_stats(elapsed_seconds) or None,
        }
    return result_payload(
        backend=backend,
        iteration=iteration,
        seed=seed,
        output=output,
        elapsed_seconds=elapsed_seconds,
        metrics=metrics,
        artifact_path=artifact_path,
        output_mode="pandas",
        batch_size=batch_size,
        max_parallel_requests=max_parallel_requests,
        expected_columns=expected_columns,
        expected_rows=num_records,
        include_retry_throttle=include_retry_throttle,
    )


def run_ray_backend_benchmark(
    *,
    backend: BackendName,
    ray_output: RayOutputMode,
    iteration: int,
    seed: int,
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    batch_size: int,
    max_parallel_requests: int,
    ray_cpus: int,
    artifact_path: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    expected_columns: list[str],
    sandbox_safe_ray_init: bool = False,
    ray_address: str | None = None,
    set_uv_runtime_env: bool = False,
    seed_python_random: bool = False,
    include_retry_throttle: bool = False,
) -> dict[str, Any]:
    seed_everything(seed, seed_python_random=seed_python_random)
    with ray_session(
        ray_cpus=ray_cpus,
        sandbox_safe_ray_init=sandbox_safe_ray_init,
        ray_address=ray_address,
        set_uv_runtime_env=set_uv_runtime_env,
    ):
        designer = DataDesigner(
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
            backend=RayBackend(batch_size=batch_size, output=ray_output),
        )
        designer.set_run_config(run_config(batch_size))
        start = time.perf_counter()
        results = designer.create(config_builder, num_records=num_records)
        output = results.load_dataset().to_pandas()
        metrics = results.load_metrics().to_dict()
        elapsed_seconds = time.perf_counter() - start
        payload = result_payload(
            backend=backend,
            iteration=iteration,
            seed=seed,
            output=output,
            elapsed_seconds=elapsed_seconds,
            metrics=metrics,
            artifact_path=artifact_path,
            output_mode=ray_output,
            batch_size=batch_size,
            max_parallel_requests=max_parallel_requests,
            expected_columns=expected_columns,
            expected_rows=num_records,
            include_retry_throttle=include_retry_throttle,
        )
        if ray_output == "arrow_refs":
            payload["arrow_ref_count"] = len(results.output)
        return payload


@contextlib.contextmanager
def ray_session(
    *,
    ray_cpus: int,
    sandbox_safe_ray_init: bool = False,
    ray_address: str | None = None,
    set_uv_runtime_env: bool = False,
) -> Iterator[Any]:
    if set_uv_runtime_env:
        os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray = __import__("ray")
    if sandbox_safe_ray_init:
        patch_ray_sandbox_process_discovery()
    init_kwargs: dict[str, Any] = {
        "num_cpus": ray_cpus,
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if ray_address is not None:
        init_kwargs["address"] = ray_address
    ray.init(**init_kwargs)
    try:
        yield ray
    finally:
        ray.shutdown()


def run_config(batch_size: int) -> RunConfig:
    return RunConfig(
        buffer_size=batch_size,
        disable_early_shutdown=True,
        max_conversation_restarts=0,
        max_conversation_correction_steps=0,
    )


def seed_everything(seed: int, *, seed_python_random: bool = False) -> None:
    lazy.np.random.seed(seed)
    if seed_python_random:
        random.seed(seed)


def result_payload(
    *,
    backend: BackendName,
    iteration: int,
    seed: int,
    output: Any,
    elapsed_seconds: float,
    metrics: dict[str, Any] | None,
    artifact_path: Path,
    output_mode: str,
    batch_size: int,
    max_parallel_requests: int,
    expected_columns: list[str],
    expected_rows: int,
    include_retry_throttle: bool = False,
) -> dict[str, Any]:
    rows = len(output)
    null_counts = {str(key): int(value) for key, value in output.isna().sum().to_dict().items()}
    empty_string_counts = empty_string_counts_for_output(output)
    missing_columns = [column for column in expected_columns if column not in output.columns]
    expected_null_counts = {column: null_counts.get(column, rows) for column in expected_columns}
    validity = {
        "row_count_matches": rows == expected_rows,
        "expected_columns_present": not missing_columns,
        "expected_columns_non_null": all(count == 0 for count in expected_null_counts.values()),
        "all_output_valid": rows == expected_rows
        and not missing_columns
        and all(count == 0 for count in expected_null_counts.values()),
    }
    throughput = throughput_payload(
        metrics=metrics,
        elapsed_seconds=elapsed_seconds,
        rows=rows,
        include_retry_throttle=include_retry_throttle,
    )
    return {
        "backend": backend,
        "iteration": iteration,
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "rows": rows,
        "rows_per_second": rows / elapsed_seconds if elapsed_seconds > 0 else 0,
        "columns": list(output.columns),
        "missing_columns": missing_columns,
        "null_counts": null_counts,
        "empty_string_counts": empty_string_counts,
        "validity": validity,
        "throughput": throughput,
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "output_mode": output_mode,
        "batch_size": batch_size,
        "max_parallel_requests": max_parallel_requests,
    }


def failure_payload(
    *,
    backend: BackendName,
    iteration: int,
    seed: int,
    elapsed_seconds: float,
    exc: Exception,
    batch_size: int,
    max_parallel_requests: int,
    expected_columns: list[str],
    include_retry_throttle: bool = False,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "iteration": iteration,
        "seed": seed,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "elapsed_seconds": elapsed_seconds,
        "rows": 0,
        "rows_per_second": 0,
        "columns": [],
        "missing_columns": expected_columns,
        "null_counts": {},
        "empty_string_counts": {},
        "validity": {
            "row_count_matches": False,
            "expected_columns_present": False,
            "expected_columns_non_null": False,
            "all_output_valid": False,
        },
        "throughput": throughput_payload(
            metrics=None,
            elapsed_seconds=elapsed_seconds,
            rows=0,
            include_retry_throttle=include_retry_throttle,
        ),
        "metrics": None,
        "artifact_path": "",
        "output_mode": "",
        "batch_size": batch_size,
        "max_parallel_requests": max_parallel_requests,
    }


def empty_string_counts_for_output(output: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in output.columns:
        series = output[column]
        if getattr(series.dtype, "kind", None) in {"O", "U", "S"}:
            counts[str(column)] = int(series.fillna("").astype(str).eq("").sum())
    return counts


def throughput_payload(
    *,
    metrics: dict[str, Any] | None,
    elapsed_seconds: float,
    rows: int,
    include_retry_throttle: bool = False,
) -> dict[str, Any]:
    model_usage = (metrics or {}).get("model_usage") or {}
    request_counts = sum_nested_usage(model_usage, "request_usage")
    token_counts = sum_nested_usage(model_usage, "token_usage")
    total_requests = int(request_counts.get("total_requests", 0))
    total_tokens = int(token_counts.get("total_tokens", 0))
    payload = {
        "rows_per_second": rows / elapsed_seconds if elapsed_seconds > 0 else 0,
        "requests_per_minute": total_requests / elapsed_seconds * 60 if elapsed_seconds > 0 else 0,
        "tokens_per_second": total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0,
        "successful_requests": int(request_counts.get("successful_requests", 0)),
        "failed_requests": int(request_counts.get("failed_requests", 0)),
        "total_requests": total_requests,
        "input_tokens": int(token_counts.get("input_tokens", 0)),
        "output_tokens": int(token_counts.get("output_tokens", 0)),
        "total_tokens": total_tokens,
    }
    if include_retry_throttle:
        retry_count = sum_first_available_counter(model_usage, ("retry_count", "retries", "total_retries"))
        throttle_count = sum_first_available_counter(
            model_usage,
            ("throttle_count", "rate_limited_requests", "throttled_requests"),
        )
        payload.update(
            {
                "retry_count": retry_count if retry_count is not None else 0,
                "retry_count_available": retry_count is not None,
                "throttle_count": throttle_count if throttle_count is not None else 0,
                "throttle_count_available": throttle_count is not None,
            }
        )
    return payload


def sum_nested_usage(model_usage: dict[str, Any], usage_key: str) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for stats in model_usage.values():
        usage = stats.get(usage_key) if isinstance(stats, dict) else None
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals


def sum_first_available_counter(model_usage: dict[str, Any], counter_names: tuple[str, ...]) -> int | None:
    total = 0
    found = False
    for stats in model_usage.values():
        if not isinstance(stats, dict):
            continue
        for counter_name in counter_names:
            value = stats.get(counter_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            total += int(value)
            found = True
            break
    return total if found else None


def build_summary(
    results: list[dict[str, Any]],
    *,
    all_backends: tuple[BackendName, ...] = ALL_BACKENDS,
    baseline_backend: BackendName = "local-sync",
    include_requests_per_minute: bool = True,
    include_retry_throttle: bool = False,
    extra_backend_stats: SummaryExtrasBuilder | None = None,
) -> dict[str, Any]:
    per_backend: dict[str, dict[str, Any]] = {}
    for backend in all_backends:
        backend_results = [result for result in results if result["backend"] == backend]
        if not backend_results:
            continue
        backend_summary = {
            "iterations": len(backend_results),
            "elapsed_seconds": compute_stats(
                [float(result["elapsed_seconds"]) for result in backend_results]
            ).to_dict(),
            "rows_per_second": compute_stats(
                [float(result["rows_per_second"]) for result in backend_results]
            ).to_dict(),
            "tokens_per_second": compute_stats(
                [float(result["throughput"]["tokens_per_second"]) for result in backend_results]
            ).to_dict(),
            "failed_requests": sum(int(result["throughput"]["failed_requests"]) for result in backend_results),
            "all_output_valid": all(bool(result["validity"]["all_output_valid"]) for result in backend_results),
        }
        if include_requests_per_minute:
            backend_summary["requests_per_minute"] = compute_stats(
                [float(result["throughput"]["requests_per_minute"]) for result in backend_results]
            ).to_dict()
        if include_retry_throttle:
            backend_summary["retry_count"] = sum(int(result["throughput"]["retry_count"]) for result in backend_results)
            backend_summary["throttle_count"] = sum(
                int(result["throughput"]["throttle_count"]) for result in backend_results
            )
        if extra_backend_stats is not None:
            backend_summary.update(extra_backend_stats(backend_results))
        per_backend[backend] = backend_summary

    comparisons = speedup_comparisons(results, all_backends=all_backends, baseline_backend=baseline_backend)
    return {
        "per_backend": per_backend,
        "comparisons_vs_local_sync": comparisons,
        "all_output_valid": all(bool(result["validity"]["all_output_valid"]) for result in results),
        "failed_backends": failed_backends(results),
    }


def failed_backends(results: list[dict[str, Any]]) -> list[str]:
    return [
        str(result["backend"])
        for result in results
        if result.get("status") == "failed" or not bool(result["validity"]["all_output_valid"])
    ]


def fail_on_invalid_summary(*, summary: dict[str, Any], allow_failures: bool) -> None:
    if allow_failures:
        return
    if summary["failed_backends"] or not summary["all_output_valid"]:
        raise SystemExit(1)


def speedup_comparisons(
    results: list[dict[str, Any]],
    *,
    all_backends: tuple[BackendName, ...] = ALL_BACKENDS,
    baseline_backend: BackendName,
) -> dict[str, Any]:
    by_iteration_backend = {(result["iteration"], result["backend"]): result for result in results}
    comparisons: dict[str, Any] = {}
    for backend in all_backends:
        if backend == baseline_backend:
            continue
        speedups = []
        for result in results:
            if result["backend"] != backend:
                continue
            baseline = by_iteration_backend.get((result["iteration"], baseline_backend))
            if baseline is None:
                continue
            elapsed = float(result["elapsed_seconds"])
            if elapsed > 0:
                speedups.append(float(baseline["elapsed_seconds"]) / elapsed)
        if speedups:
            comparisons[backend] = compute_stats(speedups).to_dict()
    return comparisons


def compute_stats(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(mean=0.0, stdev=0.0, minimum=0.0, maximum=0.0, n=0)
    if len(values) == 1:
        return MetricStats(mean=values[0], stdev=0.0, minimum=values[0], maximum=values[0], n=1)
    return MetricStats(
        mean=statistics.mean(values),
        stdev=statistics.stdev(values),
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )


def emit_result(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix}{json.dumps(payload, sort_keys=True, default=json_default)}", flush=True)


def write_json_report(
    path: Path, payload: dict[str, Any], *, sort_keys: bool = False, trailing_newline: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=sort_keys, default=json_default)
    if trailing_newline:
        content += "\n"
    path.write_text(content)


def json_default(value: Any) -> Any:
    if isinstance(value, lazy.np.generic):
        return value.item()
    if isinstance(value, lazy.np.ndarray):
        return value.tolist()
    return str(value)


@contextlib.contextmanager
def async_engine_env(*, enabled: bool) -> Iterator[None]:
    previous_value = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE")
    os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("DATA_DESIGNER_ASYNC_ENGINE", None)
        else:
            os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = previous_value


def patch_ray_sandbox_process_discovery() -> None:
    import ray._private.node

    def _sandbox_safe_system_processes(self: object) -> str:
        all_processes = getattr(self, "all_processes", {})
        pids: list[str] = []
        for processes in all_processes.values():
            if processes:
                pids.append(str(processes[0].process.pid))
        return ",".join(pids)

    ray._private.node.Node._get_system_processes_for_resource_isolation = _sandbox_safe_system_processes
