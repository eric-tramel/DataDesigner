# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark local and Ray execution with the configured OpenAI provider.

Live provider calls are intentionally opt-in:

    DATA_DESIGNER_RUN_LIVE_OPENAI_BENCHMARK=1 \
        python scripts/benchmarks/benchmark_ray_openai.py --run-live --iterations 3

The suite compares local sync, local async, Ray Dataset, and Ray Arrow-ref paths
against the same docs/assets recipe, batch size, and model concurrency settings.
It emits line-delimited JSON for logs and can write a machine-readable report.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import (
    get_default_providers,
    resolve_seed_default_model_settings,
)
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

BackendName = Literal["local-sync", "local-async", "ray-dataset", "ray-arrow-refs"]
RayOutputMode = Literal["dataset", "arrow_refs"]

ALL_BACKENDS: tuple[BackendName, ...] = ("local-sync", "local-async", "ray-dataset", "ray-arrow-refs")
RESULT_PREFIX = "RAY_OPENAI_BENCHMARK_RESULT="
LIVE_RUN_ENV = "DATA_DESIGNER_RUN_LIVE_OPENAI_BENCHMARK"
DEFAULT_RECIPE_PATH = Path("docs/assets/recipes/code_generation/text_to_python.py")
DEFAULT_NUM_RECORDS = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_ITERATIONS = 1
DEFAULT_RAY_CPUS = 4
DEFAULT_MODEL_ALIAS = "openai-text"
DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_SEED = 11


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE_PATH)
    parser.add_argument("--model-alias", type=str, default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--provider-name", type=str, default=DEFAULT_PROVIDER_NAME)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-parallel-requests", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ray-cpus", type=int, default=DEFAULT_RAY_CPUS)
    parser.add_argument(
        "--backends",
        type=str,
        default="all",
        help="Comma-separated subset of: local-sync,local-async,ray-dataset,ray-arrow-refs, or all.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/dd-ray-openai-benchmark-artifacts"))
    parser.add_argument("--managed-assets-path", type=Path, default=Path("/tmp/dd-ray-openai-benchmark-managed-assets"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--run-live",
        action="store_true",
        default=False,
        help=f"Allow live provider calls. Also accepted when {LIVE_RUN_ENV}=1.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        default=False,
        help="Exit 0 even if a backend produces invalid output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _require_live_opt_in(run_live=args.run_live)

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.managed_assets_path.mkdir(parents=True, exist_ok=True)

    resolve_seed_default_model_settings()
    selected_backends = _parse_backends(args.backends)
    recipe = _load_recipe(args.recipe_path)
    providers = _provider_configs(args.provider_name)
    _require_provider_credentials(providers)

    setup_config = _build_openai_config(
        recipe=recipe,
        model_alias=args.model_alias,
        provider_name=args.provider_name,
        max_parallel_requests=args.max_parallel_requests,
    )
    expected_columns = [column.name for column in setup_config.get_column_configs()]
    setup = {
        "recipe_path": str(args.recipe_path),
        "model_alias": args.model_alias,
        "provider_name": args.provider_name,
        "num_records": args.num_records,
        "batch_size": args.batch_size,
        "max_parallel_requests": args.max_parallel_requests,
        "iterations": args.iterations,
        "seed": args.seed,
        "ray_cpus": args.ray_cpus,
        "backends": selected_backends,
        "skip_health_checks": True,
        "model_configs": _model_config_summary(setup_config.model_configs),
        "providers": [provider.name for provider in providers],
        "expected_columns": expected_columns,
        "live_run_env": LIVE_RUN_ENV,
        "allow_failures": args.allow_failures,
    }
    _emit({"type": "setup", **setup})

    results: list[dict[str, Any]] = []
    for iteration in range(1, args.iterations + 1):
        iteration_seed = args.seed + iteration - 1
        for backend in selected_backends:
            result = _run_backend(
                backend=backend,
                iteration=iteration,
                seed=iteration_seed,
                recipe=recipe,
                model_alias=args.model_alias,
                provider_name=args.provider_name,
                num_records=args.num_records,
                batch_size=args.batch_size,
                max_parallel_requests=args.max_parallel_requests,
                ray_cpus=args.ray_cpus,
                artifact_root=args.artifact_root,
                managed_assets_path=args.managed_assets_path,
                model_providers=providers,
                expected_columns=expected_columns,
            )
            results.append(result)
            _emit({"type": "backend_result", **result})

    summary = _build_summary(results)
    _emit({"type": "summary", **summary})
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"setup": setup, "results": results, "summary": summary}, indent=2, default=_json_default)
        )
    _fail_on_invalid_summary(summary=summary, allow_failures=args.allow_failures)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_records < 1:
        raise ValueError("--num-records must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.max_parallel_requests < 1:
        raise ValueError("--max-parallel-requests must be >= 1.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.ray_cpus < 1:
        raise ValueError("--ray-cpus must be >= 1.")


def _require_live_opt_in(*, run_live: bool) -> None:
    if run_live or os.environ.get(LIVE_RUN_ENV) == "1":
        return
    raise SystemExit(
        f"This benchmark performs live provider calls. Re-run with --run-live or set {LIVE_RUN_ENV}=1 to opt in."
    )


def _parse_backends(value: str) -> list[BackendName]:
    if value.strip() == "all":
        return list(ALL_BACKENDS)

    backends: list[BackendName] = []
    seen: set[str] = set()
    for raw_backend in value.split(","):
        backend = raw_backend.strip()
        if backend not in ALL_BACKENDS:
            raise ValueError(f"Unsupported backend {backend!r}. Expected one of {', '.join(ALL_BACKENDS)} or all.")
        if backend not in seen:
            backends.append(backend)  # type: ignore[arg-type]
            seen.add(backend)
    if not backends:
        raise ValueError("At least one backend must be selected.")
    return backends


def _load_recipe(recipe_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dd_ray_openai_benchmark_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe from {recipe_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider_configs(provider_name: str) -> list[ModelProvider]:
    providers = [provider for provider in get_default_providers() if provider.name == provider_name]
    if not providers:
        raise RuntimeError(f"No configured {provider_name!r} model provider found.")
    return providers


def _require_provider_credentials(providers: list[ModelProvider]) -> None:
    missing_keys = [provider.api_key for provider in providers if not _provider_has_api_key(provider)]
    if missing_keys:
        keys = ", ".join(str(key) for key in missing_keys)
        raise SystemExit(f"Provider credentials are missing for: {keys}.")


def _provider_has_api_key(provider: ModelProvider) -> bool:
    api_key = provider.api_key
    if api_key is None:
        return False
    if api_key.isupper() and "_" in api_key:
        return bool(os.environ.get(api_key))
    return True


def _build_openai_config(
    *,
    recipe: Any,
    model_alias: str,
    provider_name: str,
    max_parallel_requests: int,
) -> DataDesignerConfigBuilder:
    source_builder = recipe.build_config(model_alias=model_alias)
    openai_model_configs = _provider_model_configs(
        source_builder.model_configs,
        provider_name=provider_name,
        max_parallel_requests=max_parallel_requests,
    )
    config_builder = DataDesignerConfigBuilder(
        model_configs=openai_model_configs,
        tool_configs=copy.deepcopy(source_builder.tool_configs),
    )
    for column_config in source_builder.get_column_configs():
        config_builder.add_column(copy.deepcopy(column_config))
    for processor_config in source_builder.get_processor_configs():
        config_builder.add_processor(copy.deepcopy(processor_config))
    return config_builder


def _provider_model_configs(
    model_configs: list[ModelConfig],
    *,
    provider_name: str,
    max_parallel_requests: int,
) -> list[ModelConfig]:
    filtered = []
    for model_config in model_configs:
        if model_config.provider != provider_name and not model_config.alias.startswith(f"{provider_name}-"):
            continue
        inference_parameters = model_config.inference_parameters.model_copy(
            update={"max_parallel_requests": max_parallel_requests}
        )
        filtered.append(
            model_config.model_copy(
                deep=True,
                update={"inference_parameters": inference_parameters, "skip_health_check": True},
            )
        )
    if not filtered:
        raise RuntimeError(f"No {provider_name!r} model configs found in recipe configuration.")
    return filtered


def _model_config_summary(model_configs: list[ModelConfig]) -> list[dict[str, Any]]:
    return [
        {
            "alias": model_config.alias,
            "model": model_config.model,
            "provider": model_config.provider,
            "max_tokens": getattr(model_config.inference_parameters, "max_tokens", None),
            "max_parallel_requests": model_config.inference_parameters.max_parallel_requests,
            "skip_health_check": model_config.skip_health_check,
        }
        for model_config in model_configs
    ]


def _run_backend(
    *,
    backend: BackendName,
    iteration: int,
    seed: int,
    recipe: Any,
    model_alias: str,
    provider_name: str,
    num_records: int,
    batch_size: int,
    max_parallel_requests: int,
    ray_cpus: int,
    artifact_root: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    expected_columns: list[str],
) -> dict[str, Any]:
    config_builder = _build_openai_config(
        recipe=recipe,
        model_alias=model_alias,
        provider_name=provider_name,
        max_parallel_requests=max_parallel_requests,
    )
    artifact_path = artifact_root / f"{backend}-iter-{iteration:02d}"
    if backend in {"local-sync", "local-async"}:
        return _run_local_backend(
            backend=backend,
            use_async=backend == "local-async",
            iteration=iteration,
            seed=seed,
            config_builder=config_builder,
            num_records=num_records,
            batch_size=batch_size,
            max_parallel_requests=max_parallel_requests,
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
            expected_columns=expected_columns,
        )

    return _run_ray_backend(
        backend=backend,
        ray_output="arrow_refs" if backend == "ray-arrow-refs" else "dataset",
        iteration=iteration,
        seed=seed,
        config_builder=config_builder,
        num_records=num_records,
        batch_size=batch_size,
        max_parallel_requests=max_parallel_requests,
        ray_cpus=ray_cpus,
        artifact_path=artifact_path,
        managed_assets_path=managed_assets_path,
        model_providers=model_providers,
        expected_columns=expected_columns,
    )


def _run_local_backend(
    *,
    backend: BackendName,
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
) -> dict[str, Any]:
    _seed_everything(seed)
    designer = DataDesigner(
        artifact_path=artifact_path,
        managed_assets_path=managed_assets_path,
        model_providers=model_providers,
    )
    designer.set_run_config(_run_config(batch_size))
    resource_provider = designer._create_resource_provider("benchmark", config_builder)
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
    return _result_payload(
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
    )


def _run_ray_backend(
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
) -> dict[str, Any]:
    _seed_everything(seed)
    ray = __import__("ray")
    ray.init(num_cpus=ray_cpus, ignore_reinit_error=True, include_dashboard=False)
    try:
        designer = DataDesigner(
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
            backend=RayBackend(batch_size=batch_size, output=ray_output),
        )
        designer.set_run_config(_run_config(batch_size))
        start = time.perf_counter()
        results = designer.create(config_builder, num_records=num_records)
        output = results.load_dataset().to_pandas()
        metrics = results.load_metrics().to_dict()
        elapsed_seconds = time.perf_counter() - start
        payload = _result_payload(
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
        )
        if ray_output == "arrow_refs":
            payload["arrow_ref_count"] = len(results.output)
        return payload
    finally:
        ray.shutdown()


def _run_config(batch_size: int) -> RunConfig:
    return RunConfig(
        buffer_size=batch_size,
        disable_early_shutdown=True,
        max_conversation_restarts=0,
        max_conversation_correction_steps=0,
    )


def _seed_everything(seed: int) -> None:
    lazy.np.random.seed(seed)


def _result_payload(
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
) -> dict[str, Any]:
    rows = len(output)
    null_counts = {str(key): int(value) for key, value in output.isna().sum().to_dict().items()}
    empty_string_counts = _empty_string_counts(output)
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
    throughput = _throughput_payload(metrics=metrics, elapsed_seconds=elapsed_seconds, rows=rows)
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


def _empty_string_counts(output: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in output.columns:
        series = output[column]
        if getattr(series.dtype, "kind", None) in {"O", "U", "S"}:
            counts[str(column)] = int(series.fillna("").astype(str).eq("").sum())
    return counts


def _throughput_payload(*, metrics: dict[str, Any] | None, elapsed_seconds: float, rows: int) -> dict[str, Any]:
    model_usage = (metrics or {}).get("model_usage") or {}
    request_counts = _sum_nested_usage(model_usage, "request_usage")
    token_counts = _sum_nested_usage(model_usage, "token_usage")
    total_requests = int(request_counts.get("total_requests", 0))
    total_tokens = int(token_counts.get("total_tokens", 0))
    retry_count = _sum_first_available_counter(model_usage, ("retry_count", "retries", "total_retries"))
    throttle_count = _sum_first_available_counter(
        model_usage,
        ("throttle_count", "rate_limited_requests", "throttled_requests"),
    )
    return {
        "rows_per_second": rows / elapsed_seconds if elapsed_seconds > 0 else 0,
        "requests_per_minute": total_requests / elapsed_seconds * 60 if elapsed_seconds > 0 else 0,
        "tokens_per_second": total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0,
        "successful_requests": int(request_counts.get("successful_requests", 0)),
        "failed_requests": int(request_counts.get("failed_requests", 0)),
        "total_requests": total_requests,
        "input_tokens": int(token_counts.get("input_tokens", 0)),
        "output_tokens": int(token_counts.get("output_tokens", 0)),
        "total_tokens": total_tokens,
        "retry_count": retry_count if retry_count is not None else 0,
        "retry_count_available": retry_count is not None,
        "throttle_count": throttle_count if throttle_count is not None else 0,
        "throttle_count_available": throttle_count is not None,
    }


def _sum_nested_usage(model_usage: dict[str, Any], usage_key: str) -> dict[str, int | float]:
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


def _sum_first_available_counter(model_usage: dict[str, Any], counter_names: tuple[str, ...]) -> int | None:
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


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_backend: dict[str, dict[str, Any]] = {}
    for backend in ALL_BACKENDS:
        backend_results = [result for result in results if result["backend"] == backend]
        if not backend_results:
            continue
        per_backend[backend] = {
            "iterations": len(backend_results),
            "elapsed_seconds": _compute_stats(
                [float(result["elapsed_seconds"]) for result in backend_results]
            ).to_dict(),
            "rows_per_second": _compute_stats(
                [float(result["rows_per_second"]) for result in backend_results]
            ).to_dict(),
            "requests_per_minute": _compute_stats(
                [float(result["throughput"]["requests_per_minute"]) for result in backend_results]
            ).to_dict(),
            "tokens_per_second": _compute_stats(
                [float(result["throughput"]["tokens_per_second"]) for result in backend_results]
            ).to_dict(),
            "failed_requests": sum(int(result["throughput"]["failed_requests"]) for result in backend_results),
            "retry_count": sum(int(result["throughput"]["retry_count"]) for result in backend_results),
            "throttle_count": sum(int(result["throughput"]["throttle_count"]) for result in backend_results),
            "all_output_valid": all(bool(result["validity"]["all_output_valid"]) for result in backend_results),
        }

    comparisons = _speedup_comparisons(results, baseline_backend="local-sync")
    return {
        "per_backend": per_backend,
        "comparisons_vs_local_sync": comparisons,
        "all_output_valid": all(bool(result["validity"]["all_output_valid"]) for result in results),
        "failed_backends": _failed_backends(results),
    }


def _failed_backends(results: list[dict[str, Any]]) -> list[str]:
    return [
        str(result["backend"])
        for result in results
        if result.get("status") == "failed" or not bool(result["validity"]["all_output_valid"])
    ]


def _fail_on_invalid_summary(*, summary: dict[str, Any], allow_failures: bool) -> None:
    if allow_failures:
        return
    if summary["failed_backends"] or not summary["all_output_valid"]:
        raise SystemExit(1)


def _speedup_comparisons(results: list[dict[str, Any]], *, baseline_backend: BackendName) -> dict[str, Any]:
    by_iteration_backend = {(result["iteration"], result["backend"]): result for result in results}
    comparisons: dict[str, Any] = {}
    for backend in ALL_BACKENDS:
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
            comparisons[backend] = _compute_stats(speedups).to_dict()
    return comparisons


def _compute_stats(values: list[float]) -> MetricStats:
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


def _emit(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True, default=_json_default)}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, lazy.np.generic):
        return value.item()
    if isinstance(value, lazy.np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
