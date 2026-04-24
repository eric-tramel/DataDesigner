# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark local and Ray execution with the configured OpenAI provider.

This benchmark intentionally uses only OpenAI model configs/providers while
exercising a real docs/assets recipe. It is meant for the experimental Ray
backend report, not as a stable public benchmark suite.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Literal

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import get_default_providers
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

BackendName = Literal["local", "ray"]

RESULT_PREFIX = "RAY_OPENAI_BENCHMARK_RESULT="
DEFAULT_RECIPE_PATH = Path("docs/assets/recipes/code_generation/text_to_python.py")
DEFAULT_NUM_RECORDS = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_RAY_CPUS = 4
DEFAULT_MODEL_ALIAS = "openai-text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE_PATH)
    parser.add_argument("--model-alias", type=str, default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ray-cpus", type=int, default=DEFAULT_RAY_CPUS)
    parser.add_argument("--backends", type=str, default="local,ray", help="Comma-separated subset of: local,ray")
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/dd-ray-openai-benchmark-artifacts"))
    parser.add_argument("--managed-assets-path", type=Path, default=Path("/tmp/dd-ray-openai-benchmark-managed-assets"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.managed_assets_path.mkdir(parents=True, exist_ok=True)

    selected_backends = _parse_backends(args.backends)
    recipe = _load_recipe(args.recipe_path)
    openai_providers = _openai_providers()
    setup = {
        "recipe_path": str(args.recipe_path),
        "model_alias": args.model_alias,
        "num_records": args.num_records,
        "batch_size": args.batch_size,
        "ray_cpus": args.ray_cpus,
        "backends": selected_backends,
        "skip_health_checks": True,
        "model_configs": _model_config_summary(_build_openai_config(recipe, args.model_alias).model_configs),
        "providers": [provider.name for provider in openai_providers],
        "local_execution": "current local default backend",
        "ray_execution": "RayBackend async worker path",
    }
    _emit({"type": "setup", **setup})

    results: list[dict[str, Any]] = []
    for backend in selected_backends:
        result = _run_backend(
            backend=backend,
            recipe=recipe,
            model_alias=args.model_alias,
            num_records=args.num_records,
            batch_size=args.batch_size,
            ray_cpus=args.ray_cpus,
            artifact_root=args.artifact_root,
            managed_assets_path=args.managed_assets_path,
            model_providers=openai_providers,
        )
        results.append(result)
        _emit({"type": "backend_result", **result})

    summary = _build_summary(results)
    if summary is not None:
        _emit({"type": "summary", **summary})
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"setup": setup, "results": results, "summary": summary}, indent=2))


def _parse_backends(value: str) -> list[BackendName]:
    backends: list[BackendName] = []
    for raw_backend in value.split(","):
        backend = raw_backend.strip()
        if backend not in {"local", "ray"}:
            raise ValueError(f"Unsupported backend {backend!r}. Expected 'local', 'ray', or both.")
        backends.append(backend)  # type: ignore[arg-type]
    return backends


def _load_recipe(recipe_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dd_ray_openai_benchmark_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe from {recipe_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _openai_providers() -> list[ModelProvider]:
    providers = [provider for provider in get_default_providers() if provider.name == "openai"]
    if not providers:
        raise RuntimeError("No configured OpenAI model provider found.")
    return providers


def _build_openai_config(recipe: Any, model_alias: str) -> DataDesignerConfigBuilder:
    source_builder = recipe.build_config(model_alias=model_alias)
    openai_model_configs = _openai_model_configs(source_builder.model_configs)
    config_builder = DataDesignerConfigBuilder(
        model_configs=openai_model_configs,
        tool_configs=copy.deepcopy(source_builder.tool_configs),
    )
    for column_config in source_builder.get_column_configs():
        config_builder.add_column(copy.deepcopy(column_config))
    for processor_config in source_builder.get_processor_configs():
        config_builder.add_processor(copy.deepcopy(processor_config))
    return config_builder


def _openai_model_configs(model_configs: list[ModelConfig]) -> list[ModelConfig]:
    filtered = [
        model_config.model_copy(deep=True, update={"skip_health_check": True})
        for model_config in model_configs
        if model_config.provider == "openai" or model_config.alias.startswith("openai-")
    ]
    if not filtered:
        raise RuntimeError("No OpenAI model configs found in recipe configuration.")
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
    recipe: Any,
    model_alias: str,
    num_records: int,
    batch_size: int,
    ray_cpus: int,
    artifact_root: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
) -> dict[str, Any]:
    config_builder = _build_openai_config(recipe, model_alias)
    artifact_path = artifact_root / f"{backend}-{num_records}-{int(time.time())}"
    start = time.perf_counter()
    if backend == "ray":
        return _run_ray_backend(
            config_builder=config_builder,
            num_records=num_records,
            batch_size=batch_size,
            ray_cpus=ray_cpus,
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
            start=start,
        )
    return _run_local_backend(
        config_builder=config_builder,
        num_records=num_records,
        artifact_path=artifact_path,
        managed_assets_path=managed_assets_path,
        model_providers=model_providers,
        start=start,
    )


def _run_local_backend(
    *,
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    start: float,
) -> dict[str, Any]:
    designer = DataDesigner(
        artifact_path=artifact_path,
        managed_assets_path=managed_assets_path,
        model_providers=model_providers,
    )
    results = designer.create(config_builder, num_records=num_records)
    output = results.load_dataset()
    elapsed_seconds = time.perf_counter() - start
    return _result_payload(
        backend="local",
        output=output,
        elapsed_seconds=elapsed_seconds,
        metrics=None,
        artifact_path=artifact_path,
    )


def _run_ray_backend(
    *,
    config_builder: DataDesignerConfigBuilder,
    num_records: int,
    batch_size: int,
    ray_cpus: int,
    artifact_path: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    start: float,
) -> dict[str, Any]:
    ray = __import__("ray")
    ray.init(num_cpus=ray_cpus, ignore_reinit_error=True)
    try:
        designer = DataDesigner(
            artifact_path=artifact_path,
            managed_assets_path=managed_assets_path,
            model_providers=model_providers,
            backend=RayBackend(batch_size=batch_size),
        )
        results = designer.create(config_builder, num_records=num_records)
        output = results.load_dataset().to_pandas()
        elapsed_seconds = time.perf_counter() - start
        return _result_payload(
            backend="ray",
            output=output,
            elapsed_seconds=elapsed_seconds,
            metrics=results.load_metrics().to_dict(),
            artifact_path=artifact_path,
        )
    finally:
        ray.shutdown()


def _result_payload(
    *,
    backend: BackendName,
    output: Any,
    elapsed_seconds: float,
    metrics: dict[str, Any] | None,
    artifact_path: Path,
) -> dict[str, Any]:
    rows = len(output)
    return {
        "backend": backend,
        "elapsed_seconds": elapsed_seconds,
        "rows": rows,
        "rows_per_second": rows / elapsed_seconds if elapsed_seconds > 0 else 0,
        "columns": list(output.columns),
        "null_counts": output.isna().sum().to_dict(),
        "metrics": metrics,
        "artifact_path": str(artifact_path),
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_backend = {result["backend"]: result for result in results}
    local = by_backend.get("local")
    ray = by_backend.get("ray")
    if local is None or ray is None:
        return None
    local_elapsed = float(local["elapsed_seconds"])
    ray_elapsed = float(ray["elapsed_seconds"])
    return {
        "local_elapsed_seconds": local_elapsed,
        "ray_elapsed_seconds": ray_elapsed,
        "ray_speedup_vs_local": local_elapsed / ray_elapsed if ray_elapsed > 0 else None,
        "local_rows_per_second": local["rows_per_second"],
        "ray_rows_per_second": ray["rows_per_second"],
    }


def _emit(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
