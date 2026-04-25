# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Ray Data LLM/vLLM execution with the RayBackend OpenAI-compatible path.

Live vLLM execution is intentionally opt-in:

    DATA_DESIGNER_RUN_RAY_DATA_LLM_VLLM_BENCHMARK=1 \
        uv run --all-packages --extra ray python scripts/benchmarks/benchmark_ray_data_llm_vllm.py \
        --model-source meta-llama/Llama-3.1-8B-Instruct \
        --provider-endpoint http://127.0.0.1:8000/v1 \
        --provider-model meta-llama/Llama-3.1-8B-Instruct \
        --require-gpu

The benchmark uses one independent plain LLM text column so the
``RayDataLLMStageOptions(execute=True)`` prototype and the existing RayBackend
+ OpenAI-compatible vLLM server/provider path can be compared directly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import ray_benchmark_common as benchmark_common

from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.integrations.ray import RayBackend, RayDataLLMStageOptions
from data_designer.integrations.ray.llm import (
    RayDataLLMCapabilities,
    RayDataLLMStagePlan,
    probe_ray_data_llm_capabilities,
)
from data_designer.interface.data_designer import DataDesigner

RESULT_PREFIX = "RAY_DATA_LLM_VLLM_BENCHMARK_RESULT="
LIVE_RUN_ENV = "DATA_DESIGNER_RUN_RAY_DATA_LLM_VLLM_BENCHMARK"
MODEL_SOURCE_ENV = "DATA_DESIGNER_RAY_DATA_LLM_MODEL_SOURCE"
VLLM_MODEL_ENV = "DATA_DESIGNER_VLLM_MODEL"
VLLM_ENDPOINT_ENV = "DATA_DESIGNER_VLLM_ENDPOINT"
DEFAULT_PROVIDER_NAME = "local-vllm"
DEFAULT_PROVIDER_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_ALIAS = "vllm-text"
DEFAULT_OUTPUT_COLUMN = "completion"
DEFAULT_PROMPT = "Write one concise sentence about synthetic dataset generation."
DEFAULT_NUM_RECORDS = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_ITERATIONS = 1
DEFAULT_RAY_CPUS = 4
DEFAULT_MAX_TOKENS = 32
DEFAULT_SEED = 118

BenchmarkBackendName = Literal["ray-openai-vllm", "ray-data-llm"]
ALL_BACKENDS: tuple[BenchmarkBackendName, ...] = ("ray-openai-vllm", "ray-data-llm")
BASELINE_BACKEND: BenchmarkBackendName = "ray-openai-vllm"


@dataclass(frozen=True)
class PreflightResult:
    """Prerequisite check result for a live Ray Data LLM/vLLM benchmark run."""

    ok: bool
    failures: tuple[str, ...]
    capabilities: RayDataLLMCapabilities
    gpu_count: int | None
    provider_checked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": list(self.failures),
            "capabilities": _capabilities_payload(self.capabilities),
            "gpu_count": self.gpu_count,
            "provider_checked": self.provider_checked,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", default=os.environ.get(MODEL_SOURCE_ENV) or os.environ.get(VLLM_MODEL_ENV))
    parser.add_argument("--provider-model", default=os.environ.get(VLLM_MODEL_ENV))
    parser.add_argument("--provider-endpoint", default=os.environ.get(VLLM_ENDPOINT_ENV, DEFAULT_PROVIDER_ENDPOINT))
    parser.add_argument("--provider-name", default=DEFAULT_PROVIDER_NAME)
    parser.add_argument("--provider-api-key-env", default=None)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--output-column", default=DEFAULT_OUTPUT_COLUMN)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-parallel-requests", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ray-cpus", type=int, default=DEFAULT_RAY_CPUS)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-data-llm-batch-size", type=int, default=None)
    parser.add_argument("--ray-data-llm-concurrency", default=None)
    parser.add_argument("--engine-kwargs-json", default=None)
    parser.add_argument(
        "--backends",
        default="all",
        help="Comma-separated subset of: ray-openai-vllm,ray-data-llm, or all.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/dd-ray-data-llm-vllm-benchmark-artifacts"))
    parser.add_argument(
        "--managed-assets-path",
        type=Path,
        default=Path("/tmp/dd-ray-data-llm-vllm-benchmark-managed-assets"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--run-live",
        action="store_true",
        default=False,
        help=f"Allow live vLLM execution. Also accepted when {LIVE_RUN_ENV}=1.",
    )
    parser.add_argument(
        "--skip-missing-prereqs",
        action="store_true",
        default=False,
        help="Exit 0 with a skipped JSON payload when live prerequisites are missing.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        default=False,
        help="Exit 0 even if one compared backend fails or produces invalid output.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=False,
        help="Require a visible NVIDIA GPU before running the benchmark.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        default=False,
        help="Do not verify GPU visibility during preflight.",
    )
    parser.add_argument(
        "--skip-provider-health-check",
        action="store_true",
        default=False,
        help="Do not probe the OpenAI-compatible /models endpoint before running.",
    )
    parser.add_argument(
        "--sandbox-safe-ray-init",
        action="store_true",
        default=False,
        help="Patch Ray process discovery for local macOS sandbox benchmark runs.",
    )
    parser.add_argument(
        "--set-uv-runtime-env",
        action="store_true",
        default=False,
        help="Set RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 before Ray initialization.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    _require_live_opt_in(run_live=args.run_live)

    selected_backends = _parse_backends(args.backends)
    provider_model = _provider_model(args)
    expected_columns = [args.output_column]
    preflight = _preflight(args=args, provider_model=provider_model)
    if not preflight.ok:
        _handle_preflight_failure(args=args, preflight=preflight)
        return

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.managed_assets_path.mkdir(parents=True, exist_ok=True)
    setup_config = _build_config(args=args, provider_model=provider_model)

    setup = {
        "model_source": args.model_source,
        "provider_model": provider_model,
        "provider_endpoint": args.provider_endpoint,
        "provider_name": args.provider_name,
        "model_alias": args.model_alias,
        "output_column": args.output_column,
        "num_records": args.num_records,
        "batch_size": args.batch_size,
        "max_parallel_requests": args.max_parallel_requests,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "iterations": args.iterations,
        "seed": args.seed,
        "ray_cpus": args.ray_cpus,
        "ray_address": args.ray_address,
        "backends": selected_backends,
        "preflight": preflight.to_dict(),
        "model_configs": _model_config_summary(setup_config.model_configs),
        "expected_columns": expected_columns,
        "live_run_env": LIVE_RUN_ENV,
        "allow_failures": args.allow_failures,
    }
    _emit({"type": "setup", **setup})

    model_providers = [_provider_config(args)]
    results: list[dict[str, Any]] = []
    for iteration in range(1, args.iterations + 1):
        iteration_seed = args.seed + iteration - 1
        for backend in selected_backends:
            start = time.perf_counter()
            try:
                result = _run_backend(
                    backend=backend,
                    iteration=iteration,
                    seed=iteration_seed,
                    args=args,
                    provider_model=provider_model,
                    model_providers=model_providers,
                    expected_columns=expected_columns,
                )
            except Exception as exc:
                result = benchmark_common.failure_payload(
                    backend=cast(benchmark_common.BackendName, backend),
                    iteration=iteration,
                    seed=iteration_seed,
                    elapsed_seconds=time.perf_counter() - start,
                    exc=exc,
                    batch_size=args.batch_size,
                    max_parallel_requests=args.max_parallel_requests,
                    expected_columns=expected_columns,
                    include_retry_throttle=True,
                )
                result["backend"] = backend
                result["error_context"] = "benchmark backend execution failed"
            results.append(result)
            _emit({"type": "backend_result", **result})

    summary = _build_summary(results)
    _emit({"type": "summary", **summary})
    report = {"setup": setup, "results": results, "summary": summary}
    if args.output_json is not None:
        benchmark_common.write_json_report(args.output_json, report, sort_keys=True, trailing_newline=True)
    benchmark_common.fail_on_invalid_summary(summary=summary, allow_failures=args.allow_failures)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_records < 1:
        raise ValueError("--num-records must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.max_parallel_requests < 1:
        raise ValueError("--max-parallel-requests must be >= 1.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1.")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1.")
    if args.ray_cpus < 1:
        raise ValueError("--ray-cpus must be >= 1.")
    if args.ray_data_llm_batch_size is not None and args.ray_data_llm_batch_size < 1:
        raise ValueError("--ray-data-llm-batch-size must be >= 1 when provided.")
    _parse_concurrency(args.ray_data_llm_concurrency)
    _parse_engine_kwargs(args.engine_kwargs_json)
    _parse_backends(args.backends)


def _require_live_opt_in(*, run_live: bool) -> None:
    if run_live or os.environ.get(LIVE_RUN_ENV) == "1":
        return
    raise SystemExit(
        f"This benchmark performs live vLLM execution. Re-run with --run-live or set {LIVE_RUN_ENV}=1 to opt in."
    )


def _parse_backends(value: str) -> list[BenchmarkBackendName]:
    if value.strip() == "all":
        return list(ALL_BACKENDS)
    backends: list[BenchmarkBackendName] = []
    seen: set[str] = set()
    for raw_backend in value.split(","):
        backend = raw_backend.strip()
        if backend not in ALL_BACKENDS:
            raise ValueError(f"Unsupported backend {backend!r}. Expected one of {', '.join(ALL_BACKENDS)} or all.")
        if backend not in seen:
            backends.append(cast(BenchmarkBackendName, backend))
            seen.add(backend)
    if not backends:
        raise ValueError("At least one backend must be selected.")
    return backends


def _provider_model(args: argparse.Namespace) -> str:
    provider_model = args.provider_model or args.model_source
    return str(provider_model or "")


def _preflight(*, args: argparse.Namespace, provider_model: str) -> PreflightResult:
    failures: list[str] = []
    if not args.model_source:
        failures.append(
            f"--model-source is required for Ray Data LLM execution; set it or {MODEL_SOURCE_ENV}/{VLLM_MODEL_ENV}."
        )
    if not provider_model:
        failures.append("--provider-model is required for the OpenAI-compatible vLLM server path.")

    capabilities = probe_ray_data_llm_capabilities()
    if not capabilities.supports_local_vllm:
        detail = f" import failed with {capabilities.import_error}" if capabilities.import_error else ""
        failures.append(
            "ray.data.llm local vLLM processor APIs are unavailable"
            f" (ray_version={capabilities.ray_version!r}, missing={capabilities.missing_symbols!r}).{detail}"
        )

    if not _module_importable("vllm"):
        failures.append("vLLM is not importable on the driver. Install vLLM in the benchmark environment.")
    if not _module_importable("ray"):
        failures.append("Ray is not importable. Install DataDesigner with the ray extra.")

    gpu_count: int | None = None
    if args.require_gpu and not args.skip_gpu_check:
        gpu_count = _visible_nvidia_gpu_count()
        if gpu_count is None:
            failures.append("GPU availability could not be verified with CUDA_VISIBLE_DEVICES or nvidia-smi.")
        elif gpu_count < 1:
            failures.append("No visible NVIDIA GPU was found.")

    provider_checked = False
    if not args.skip_provider_health_check:
        provider_checked = True
        try:
            _check_provider_endpoint(args=args)
        except RuntimeError as exc:
            failures.append(str(exc))

    return PreflightResult(
        ok=not failures,
        failures=tuple(failures),
        capabilities=capabilities,
        gpu_count=gpu_count,
        provider_checked=provider_checked,
    )


def _handle_preflight_failure(*, args: argparse.Namespace, preflight: PreflightResult) -> None:
    payload = {
        "status": "skipped" if args.skip_missing_prereqs else "failed",
        "type": "preflight",
        "preflight": preflight.to_dict(),
    }
    _emit(payload)
    if args.output_json is not None:
        benchmark_common.write_json_report(
            args.output_json,
            {"setup": {"live_run_env": LIVE_RUN_ENV}, "results": [], "summary": payload},
            sort_keys=True,
            trailing_newline=True,
        )
    if args.skip_missing_prereqs:
        raise SystemExit(0)
    reasons = "; ".join(preflight.failures)
    raise SystemExit(f"Ray Data LLM/vLLM benchmark prerequisites are missing: {reasons}")


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _visible_nvidia_gpu_count() -> int | None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        normalized = cuda_visible_devices.strip()
        if normalized in {"", "-1", "none", "None"}:
            return 0
        return len([device for device in normalized.split(",") if device.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))


def _check_provider_endpoint(*, args: argparse.Namespace) -> None:
    endpoint = str(args.provider_endpoint).rstrip("/")
    url = f"{endpoint}/models"
    headers: dict[str, str] = {}
    api_key = _provider_api_key_value(args)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise RuntimeError(f"OpenAI-compatible vLLM endpoint health check failed with HTTP {status}: {url}")
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI-compatible vLLM endpoint health check failed with HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI-compatible vLLM endpoint is not reachable at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"OpenAI-compatible vLLM endpoint timed out at {url}.") from exc


def _provider_api_key_value(args: argparse.Namespace) -> str | None:
    if args.provider_api_key_env is None:
        return None
    value = os.environ.get(args.provider_api_key_env)
    if not value:
        raise RuntimeError(f"Provider API key environment variable {args.provider_api_key_env!r} is not set.")
    return value


def _provider_config(args: argparse.Namespace) -> ModelProvider:
    return ModelProvider(
        name=args.provider_name,
        endpoint=args.provider_endpoint,
        provider_type="openai",
        api_key=args.provider_api_key_env,
    )


def _build_config(*, args: argparse.Namespace, provider_model: str) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias=args.model_alias,
                model=provider_model,
                provider=args.provider_name,
                skip_health_check=True,
                inference_parameters=ChatCompletionInferenceParams(
                    max_parallel_requests=args.max_parallel_requests,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
            )
        ]
    )
    config_builder.add_column(
        LLMTextColumnConfig(
            name=args.output_column,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            model_alias=args.model_alias,
        )
    )
    return config_builder


def _run_backend(
    *,
    backend: BenchmarkBackendName,
    iteration: int,
    seed: int,
    args: argparse.Namespace,
    provider_model: str,
    model_providers: list[ModelProvider],
    expected_columns: list[str],
) -> dict[str, Any]:
    config_builder = _build_config(args=args, provider_model=provider_model)
    artifact_path = args.artifact_root / f"{backend}-iter-{iteration:02d}"
    if backend == "ray-openai-vllm":
        return _run_ray_openai_vllm_backend(
            backend=backend,
            iteration=iteration,
            seed=seed,
            args=args,
            config_builder=config_builder,
            model_providers=model_providers,
            artifact_path=artifact_path,
            expected_columns=expected_columns,
        )
    return _run_ray_data_llm_backend(
        backend=backend,
        iteration=iteration,
        seed=seed,
        args=args,
        config_builder=config_builder,
        model_providers=model_providers,
        artifact_path=artifact_path,
        expected_columns=expected_columns,
    )


def _run_ray_openai_vllm_backend(
    *,
    backend: BenchmarkBackendName,
    iteration: int,
    seed: int,
    args: argparse.Namespace,
    config_builder: DataDesignerConfigBuilder,
    model_providers: list[ModelProvider],
    artifact_path: Path,
    expected_columns: list[str],
) -> dict[str, Any]:
    result = benchmark_common.run_ray_backend_benchmark(
        backend=cast(benchmark_common.BackendName, backend),
        ray_output="dataset",
        iteration=iteration,
        seed=seed,
        config_builder=config_builder,
        num_records=args.num_records,
        batch_size=args.batch_size,
        max_parallel_requests=args.max_parallel_requests,
        ray_cpus=args.ray_cpus,
        artifact_path=artifact_path,
        managed_assets_path=args.managed_assets_path,
        model_providers=model_providers,
        expected_columns=expected_columns,
        sandbox_safe_ray_init=args.sandbox_safe_ray_init,
        ray_address=args.ray_address,
        set_uv_runtime_env=args.set_uv_runtime_env,
        include_retry_throttle=True,
    )
    result["backend"] = backend
    result["comparison_role"] = "ray-backend-openai-compatible-vllm-server"
    return result


def _run_ray_data_llm_backend(
    *,
    backend: BenchmarkBackendName,
    iteration: int,
    seed: int,
    args: argparse.Namespace,
    config_builder: DataDesignerConfigBuilder,
    model_providers: list[ModelProvider],
    artifact_path: Path,
    expected_columns: list[str],
) -> dict[str, Any]:
    benchmark_common.seed_everything(seed)
    with benchmark_common.ray_session(
        ray_cpus=args.ray_cpus,
        sandbox_safe_ray_init=args.sandbox_safe_ray_init,
        ray_address=args.ray_address,
        set_uv_runtime_env=args.set_uv_runtime_env,
    ):
        designer = DataDesigner(
            artifact_path=artifact_path,
            managed_assets_path=args.managed_assets_path,
            model_providers=model_providers,
            backend=RayBackend(
                batch_size=args.batch_size,
                output="dataset",
                preflight_model_health_check=False,
                llm_stage_options=RayDataLLMStageOptions(
                    enabled=True,
                    execute=True,
                    model_source=args.model_source,
                    column_names=(args.output_column,),
                    batch_size=args.ray_data_llm_batch_size or args.batch_size,
                    concurrency=_parse_concurrency(args.ray_data_llm_concurrency),
                    engine_kwargs=_parse_engine_kwargs(args.engine_kwargs_json),
                    allow_model_facade_fallback=False,
                ),
            ),
        )
        designer.set_run_config(benchmark_common.run_config(args.batch_size))
        start = time.perf_counter()
        results = designer.create(config_builder, num_records=args.num_records)
        output = results.load_dataset().to_pandas()
        metrics = results.load_metrics().to_dict()
        ray_diagnostics = benchmark_common.load_ray_diagnostics_payload(results)
        elapsed_seconds = time.perf_counter() - start
        payload = benchmark_common.result_payload(
            backend=cast(benchmark_common.BackendName, backend),
            iteration=iteration,
            seed=seed,
            output=output,
            elapsed_seconds=elapsed_seconds,
            metrics=metrics,
            artifact_path=artifact_path,
            output_mode="dataset",
            batch_size=args.batch_size,
            max_parallel_requests=args.max_parallel_requests,
            expected_columns=expected_columns,
            expected_rows=args.num_records,
            include_retry_throttle=True,
        )
        payload["backend"] = backend
        payload["comparison_role"] = "ray-data-llm-vllm-engine-processor"
        payload["llm_stage_plan"] = _llm_stage_plan_payload(results.llm_stage_plan)
        payload["ray_diagnostics"] = ray_diagnostics
        return payload


def _parse_concurrency(value: str | None) -> int | tuple[int, int] | None:
    if value is None or value == "":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 1:
        concurrency = int(parts[0])
        if concurrency < 1:
            raise ValueError("--ray-data-llm-concurrency must be >= 1.")
        return concurrency
    if len(parts) == 2:
        minimum, maximum = (int(part) for part in parts)
        if minimum < 1 or maximum < minimum:
            raise ValueError("--ray-data-llm-concurrency tuple must satisfy 1 <= min <= max.")
        return (minimum, maximum)
    raise ValueError("--ray-data-llm-concurrency must be an integer or min,max.")


def _parse_engine_kwargs(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--engine-kwargs-json must decode to a JSON object.")
    return dict(parsed)


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = benchmark_common.build_summary(
        results,
        all_backends=cast(tuple[benchmark_common.BackendName, ...], ALL_BACKENDS),
        baseline_backend=cast(benchmark_common.BackendName, BASELINE_BACKEND),
        include_requests_per_minute=True,
        include_retry_throttle=True,
    )
    summary["comparisons_vs_ray_openai_vllm"] = summary.pop("comparisons_vs_local_sync")
    summary["baseline_backend"] = BASELINE_BACKEND
    summary["ray_data_llm_usage_stats_note"] = (
        "Ray Data LLM execution currently does not report ModelFacade token/request usage stats."
    )
    return summary


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


def _capabilities_payload(capabilities: RayDataLLMCapabilities) -> dict[str, Any]:
    return {
        "available": capabilities.available,
        "ray_version": capabilities.ray_version,
        "missing_symbols": list(capabilities.missing_symbols),
        "has_build_processor": capabilities.has_build_processor,
        "has_vllm_engine_processor": capabilities.has_vllm_engine_processor,
        "has_http_request_processor": capabilities.has_http_request_processor,
        "has_serve_deployment_processor": capabilities.has_serve_deployment_processor,
        "supports_local_vllm": capabilities.supports_local_vllm,
        "supports_openai_compatible_endpoint": capabilities.supports_openai_compatible_endpoint,
        "import_error": capabilities.import_error,
    }


def _llm_stage_plan_payload(plan: RayDataLLMStagePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "status": plan.status,
        "enabled": plan.enabled,
        "should_execute": plan.should_execute,
        "disabled_reason": plan.disabled_reason,
        "selected_candidate": None
        if plan.selected_candidate is None
        else {
            "column_name": plan.selected_candidate.column_name,
            "model_alias": plan.selected_candidate.model_alias,
            "task_type": plan.selected_candidate.task_type,
            "model_source": plan.selected_candidate.model_source,
            "blocked_reasons": list(plan.selected_candidate.blocked_reasons),
        },
        "capabilities": None if plan.capabilities is None else _capabilities_payload(plan.capabilities),
    }


def _emit(payload: dict[str, Any]) -> None:
    benchmark_common.emit_result(RESULT_PREFIX, payload)


if __name__ == "__main__":
    main()
