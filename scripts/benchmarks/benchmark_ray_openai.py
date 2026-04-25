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
import os
from pathlib import Path
from typing import Any

import ray_benchmark_common as benchmark_common
from ray_benchmark_common import ALL_BACKENDS, BackendName, RayOutputMode

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import (
    get_default_providers,
    resolve_seed_default_model_settings,
)
from data_designer.config.models import ModelConfig, ModelProvider

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
    selected_backends = benchmark_common.parse_backends(args.backends)
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
        benchmark_common.write_json_report(args.output_json, {"setup": setup, "results": results, "summary": summary})
    benchmark_common.fail_on_invalid_summary(summary=summary, allow_failures=args.allow_failures)


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
    return benchmark_common.run_local_preview_benchmark(
        backend=backend,
        dataset_name="benchmark",
        use_async=use_async,
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
        include_retry_throttle=True,
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
    return benchmark_common.run_ray_backend_benchmark(
        backend=backend,
        ray_output=ray_output,
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
        include_retry_throttle=True,
    )


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return benchmark_common.build_summary(
        results,
        all_backends=ALL_BACKENDS,
        include_requests_per_minute=True,
        include_retry_throttle=True,
    )


def _emit(payload: dict[str, Any]) -> None:
    benchmark_common.emit_result(RESULT_PREFIX, payload)


if __name__ == "__main__":
    main()
