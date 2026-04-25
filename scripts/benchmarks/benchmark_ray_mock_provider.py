# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Ray backend scaling against a local mock OpenAI-compatible provider.

The mock provider returns deterministic pseudo-random token counts and sleeps for
``base_latency_ms + output_tokens * delay_ms_per_output_token`` before responding.
This keeps benchmark cost local while exercising the same HTTP model client,
throttling, local async, Ray Dataset, and Ray Arrow-ref execution paths used by
live provider runs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import signal
import socket
import statistics
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import ray_benchmark_common as benchmark_common
from ray_benchmark_common import ALL_BACKENDS, BackendName, RayOutputMode

import data_designer.config as dd
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider

RESULT_PREFIX = "RAY_MOCK_PROVIDER_BENCHMARK_RESULT="
DEFAULT_MODEL_ALIAS = "mock-text"
DEFAULT_PROVIDER_NAME = "mock-provider"
DEFAULT_MODEL_NAME = "mock-token-latency"
DEFAULT_NUM_RECORDS = 1024
DEFAULT_BATCH_SIZE = 64
DEFAULT_ITERATIONS = 1
DEFAULT_RAY_CPUS = 4
DEFAULT_LLM_COLUMNS = 3
DEFAULT_SEED = 11


@dataclass(frozen=True)
class MockProviderConfig:
    seed: int
    min_output_tokens: int
    max_output_tokens: int
    base_latency_ms: float
    delay_ms_per_output_token: float
    delay_ms_per_input_token: float
    jitter_ms: float
    chars_per_input_token: int
    emit_token_text: bool


@dataclass
class MockProviderStats:
    request_count: int = 0
    active_requests: int = 0
    max_in_flight: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_delay_seconds: float = 0.0
    status_counts: dict[int, int] = field(default_factory=dict)
    delays_seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        delays = sorted(self.delays_seconds)
        return {
            "request_count": self.request_count,
            "max_in_flight": self.max_in_flight,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_delay_seconds": self.total_delay_seconds,
            "status_counts": dict(self.status_counts),
            "delay_seconds": {
                "mean": statistics.mean(delays) if delays else 0.0,
                "min": delays[0] if delays else 0.0,
                "p50": _percentile(delays, 0.50),
                "p95": _percentile(delays, 0.95),
                "max": delays[-1] if delays else 0.0,
            },
        }


class MockTokenProviderService:
    def __init__(self, config: MockProviderConfig) -> None:
        self.config = config
        self._stats = MockProviderStats()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        handler = _handler_for_service(self)
        server = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
        self._server = server
        self.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        self._thread = threading.Thread(target=server.serve_forever, name="mock-token-provider", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self.url = None

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = MockProviderStats()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._stats.to_dict()

    def handle_chat_completion(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        model = str(payload.get("model") or DEFAULT_MODEL_NAME)
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        request_hash = _stable_request_hash(self.config.seed, payload)
        rng = random.Random(request_hash)

        input_tokens = _estimate_input_tokens(messages, chars_per_token=self.config.chars_per_input_token)
        max_tokens = payload.get("max_tokens")
        output_ceiling = self.config.max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            output_ceiling = min(output_ceiling, max_tokens)
        output_floor = min(self.config.min_output_tokens, output_ceiling)
        output_tokens = rng.randint(output_floor, output_ceiling)
        delay_seconds = (
            self.config.base_latency_ms
            + output_tokens * self.config.delay_ms_per_output_token
            + input_tokens * self.config.delay_ms_per_input_token
            + rng.random() * self.config.jitter_ms
        ) / 1000.0

        self._record_request_started()
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            content = _mock_content(request_hash, output_tokens, emit_token_text=self.config.emit_token_text)
            response = {
                "id": f"chatcmpl-mock-{request_hash:x}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
            self._record_request_finished(
                status_code=200,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                delay_seconds=delay_seconds,
            )
            return 200, response
        except Exception:
            self._record_request_finished(
                status_code=500,
                input_tokens=input_tokens,
                output_tokens=0,
                delay_seconds=delay_seconds,
            )
            raise

    def _record_request_started(self) -> None:
        with self._lock:
            self._stats.request_count += 1
            self._stats.active_requests += 1
            self._stats.max_in_flight = max(self._stats.max_in_flight, self._stats.active_requests)

    def _record_request_finished(
        self,
        *,
        status_code: int,
        input_tokens: int,
        output_tokens: int,
        delay_seconds: float,
    ) -> None:
        with self._lock:
            self._stats.active_requests = max(0, self._stats.active_requests - 1)
            self._stats.total_input_tokens += input_tokens
            self._stats.total_output_tokens += output_tokens
            self._stats.total_tokens += input_tokens + output_tokens
            self._stats.total_delay_seconds += delay_seconds
            self._stats.status_counts[status_code] = self._stats.status_counts.get(status_code, 0) + 1
            self._stats.delays_seconds.append(delay_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-parallel-requests", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ray-cpus", type=int, default=DEFAULT_RAY_CPUS)
    parser.add_argument("--llm-columns", type=int, default=DEFAULT_LLM_COLUMNS)
    parser.add_argument(
        "--backends",
        type=str,
        default="ray-dataset,ray-arrow-refs",
        help="Comma-separated subset of: local-sync,local-async,ray-dataset,ray-arrow-refs, or all.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/dd-ray-mock-provider-benchmark-artifacts"))
    parser.add_argument(
        "--managed-assets-path",
        type=Path,
        default=Path("/tmp/dd-ray-mock-provider-benchmark-managed-assets"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-output-tokens", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--base-latency-ms", type=float, default=5.0)
    parser.add_argument("--delay-ms-per-output-token", type=float, default=0.35)
    parser.add_argument("--delay-ms-per-input-token", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--chars-per-input-token", type=int, default=4)
    parser.add_argument(
        "--compact-response-text",
        action="store_true",
        default=False,
        help="Return a short response string while still reporting simulated token counts.",
    )
    parser.add_argument(
        "--backend-timeout-sec",
        type=float,
        default=0.0,
        help="Abort an individual backend if it exceeds this many seconds. 0 disables the timeout.",
    )
    parser.add_argument(
        "--sandbox-safe-ray-init",
        action="store_true",
        default=False,
        help="Patch Ray startup process enumeration for restricted macOS sandboxes.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        default=False,
        help="Exit 0 even if a backend fails or produces invalid output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.managed_assets_path.mkdir(parents=True, exist_ok=True)

    service_config = MockProviderConfig(
        seed=args.seed,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        base_latency_ms=args.base_latency_ms,
        delay_ms_per_output_token=args.delay_ms_per_output_token,
        delay_ms_per_input_token=args.delay_ms_per_input_token,
        jitter_ms=args.jitter_ms,
        chars_per_input_token=args.chars_per_input_token,
        emit_token_text=not args.compact_response_text,
    )
    selected_backends = benchmark_common.parse_backends(args.backends)
    service = MockTokenProviderService(service_config)
    service.start()
    try:
        if service.url is None:
            raise RuntimeError("Mock provider service did not start.")
        providers = [
            ModelProvider(
                name=DEFAULT_PROVIDER_NAME,
                endpoint=service.url,
                provider_type="openai",
                api_key="mock-key",
            )
        ]
        setup_config = _build_mock_config(
            llm_columns=args.llm_columns,
            max_parallel_requests=args.max_parallel_requests,
        )
        expected_columns = [column.name for column in setup_config.get_column_configs()]
        setup = {
            "num_records": args.num_records,
            "batch_size": args.batch_size,
            "max_parallel_requests": args.max_parallel_requests,
            "iterations": args.iterations,
            "seed": args.seed,
            "ray_cpus": args.ray_cpus,
            "llm_columns": args.llm_columns,
            "expected_requests_per_iteration": args.num_records * args.llm_columns,
            "backends": selected_backends,
            "model_configs": _model_config_summary(setup_config.model_configs),
            "providers": [provider.name for provider in providers],
            "expected_columns": expected_columns,
            "mock_provider": {
                **service_config.__dict__,
                "endpoint": service.url,
            },
        }
        _emit({"type": "setup", **setup})

        results: list[dict[str, Any]] = []
        for iteration in range(1, args.iterations + 1):
            iteration_seed = args.seed + iteration - 1
            for backend in selected_backends:
                service.reset_stats()
                start = time.perf_counter()
                try:
                    with _timeout_after(args.backend_timeout_sec, f"{backend} iteration {iteration}"):
                        result = _run_backend(
                            backend=backend,
                            iteration=iteration,
                            seed=iteration_seed,
                            num_records=args.num_records,
                            batch_size=args.batch_size,
                            max_parallel_requests=args.max_parallel_requests,
                            ray_cpus=args.ray_cpus,
                            llm_columns=args.llm_columns,
                            artifact_root=args.artifact_root,
                            managed_assets_path=args.managed_assets_path,
                            model_providers=providers,
                            expected_columns=expected_columns,
                            sandbox_safe_ray_init=args.sandbox_safe_ray_init,
                        )
                    result["status"] = "ok"
                except Exception as exc:
                    result = benchmark_common.failure_payload(
                        backend=backend,
                        iteration=iteration,
                        seed=iteration_seed,
                        elapsed_seconds=time.perf_counter() - start,
                        exc=exc,
                        batch_size=args.batch_size,
                        max_parallel_requests=args.max_parallel_requests,
                        expected_columns=expected_columns,
                    )
                result["mock_service"] = _mock_service_payload(
                    service.snapshot(), elapsed_seconds=result["elapsed_seconds"]
                )
                results.append(result)
                _emit({"type": "backend_result", **result})

        summary = _build_summary(results)
        _emit({"type": "summary", **summary})
        if args.output_json is not None:
            benchmark_common.write_json_report(
                args.output_json, {"setup": setup, "results": results, "summary": summary}
            )
        benchmark_common.fail_on_invalid_summary(summary=summary, allow_failures=args.allow_failures)
    finally:
        service.stop()


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
    if args.llm_columns < 1:
        raise ValueError("--llm-columns must be >= 1.")
    if args.min_output_tokens < 1 or args.max_output_tokens < 1:
        raise ValueError("--min-output-tokens and --max-output-tokens must be >= 1.")
    if args.min_output_tokens > args.max_output_tokens:
        raise ValueError("--min-output-tokens must be <= --max-output-tokens.")
    if args.chars_per_input_token < 1:
        raise ValueError("--chars-per-input-token must be >= 1.")


def _build_mock_config(*, llm_columns: int, max_parallel_requests: int) -> DataDesignerConfigBuilder:
    model_config = ModelConfig(
        alias=DEFAULT_MODEL_ALIAS,
        model=DEFAULT_MODEL_NAME,
        provider=DEFAULT_PROVIDER_NAME,
        skip_health_check=True,
        inference_parameters=ChatCompletionInferenceParams(
            max_parallel_requests=max_parallel_requests,
            timeout=300,
        ),
    )
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    builder.add_column(
        dd.SamplerColumnConfig(
            name="segment",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["alpha", "beta", "gamma", "delta"]),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="entropy",
            sampler_type=dd.SamplerType.UNIFORM,
            params=dd.UniformSamplerParams(low=0, high=1_000_000, decimal_places=0),
        )
    )
    previous_column = "entropy"
    for index in range(1, llm_columns + 1):
        column_name = f"mock_llm_{index:02d}"
        builder.add_column(
            dd.LLMTextColumnConfig(
                name=column_name,
                model_alias=DEFAULT_MODEL_ALIAS,
                system_prompt="You are a deterministic benchmark model.",
                prompt=(
                    f"Generate benchmark text for stage {index}; "
                    "segment={{ segment }}; entropy={{ entropy }}; previous={{ " + previous_column + " }}."
                ),
            )
        )
        previous_column = column_name
    return builder


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
    num_records: int,
    batch_size: int,
    max_parallel_requests: int,
    ray_cpus: int,
    llm_columns: int,
    artifact_root: Path,
    managed_assets_path: Path,
    model_providers: list[ModelProvider],
    expected_columns: list[str],
    sandbox_safe_ray_init: bool,
) -> dict[str, Any]:
    config_builder = _build_mock_config(llm_columns=llm_columns, max_parallel_requests=max_parallel_requests)
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
        sandbox_safe_ray_init=sandbox_safe_ray_init,
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
        dataset_name="mock-benchmark",
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
        manage_async_engine_env=True,
        seed_python_random=True,
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
    sandbox_safe_ray_init: bool,
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
        sandbox_safe_ray_init=sandbox_safe_ray_init,
        ray_address="local",
        set_uv_runtime_env=True,
        seed_python_random=True,
    )


def _mock_service_payload(snapshot: dict[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    total_delay_seconds = float(snapshot.get("total_delay_seconds", 0.0) or 0.0)
    request_count = int(snapshot.get("request_count", 0) or 0)
    return {
        **snapshot,
        "effective_service_concurrency": total_delay_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "requests_per_second": request_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return benchmark_common.build_summary(
        results,
        all_backends=ALL_BACKENDS,
        include_requests_per_minute=False,
        extra_backend_stats=_mock_backend_stats,
    )


def _mock_backend_stats(backend_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "effective_service_concurrency": benchmark_common.compute_stats(
            [float(result["mock_service"]["effective_service_concurrency"]) for result in backend_results]
        ).to_dict(),
        "max_in_flight": max(int(result["mock_service"]["max_in_flight"]) for result in backend_results),
        "failed": any(result.get("status") == "failed" for result in backend_results),
    }


def _handler_for_service(service: MockTokenProviderService) -> type[BaseHTTPRequestHandler]:
    class MockProviderHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            if not self.path.endswith("/chat/completions"):
                self._write_json(404, {"error": {"message": f"Unsupported route {self.path!r}"}})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                status_code, response = service.handle_chat_completion(payload)
            except Exception as exc:
                status_code = 500
                response = {"error": {"message": str(exc), "type": type(exc).__name__}}
            self._write_json(status_code, response)

        def do_GET(self) -> None:
            if self.path.endswith("/health"):
                self._write_json(200, {"status": "ok"})
            elif self.path.endswith("/stats"):
                self._write_json(200, service.snapshot())
            else:
                self._write_json(404, {"error": {"message": f"Unsupported route {self.path!r}"}})

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, default=benchmark_common.json_default).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MockProviderHandler


def _stable_request_hash(seed: int, payload: dict[str, Any]) -> int:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=benchmark_common.json_default)
    digest = hashlib.sha256(f"{seed}:{serialized}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _estimate_input_tokens(messages: list[Any], *, chars_per_token: int) -> int:
    char_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            char_count += len(content)
        elif isinstance(content, list):
            char_count += sum(len(str(item)) for item in content)
        else:
            char_count += len(str(content or ""))
    return max(1, math.ceil(char_count / chars_per_token))


def _mock_content(request_hash: int, output_tokens: int, *, emit_token_text: bool) -> str:
    if not emit_token_text:
        return f"mock response {request_hash:x} output_tokens={output_tokens}"
    return " ".join(f"tok{(request_hash + index) % 100_000:05d}" for index in range(output_tokens))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * percentile))))
    return sorted_values[index]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _timeout_after(seconds: float, label: str) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"{label} exceeded timeout of {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _emit(payload: dict[str, Any]) -> None:
    benchmark_common.emit_result(RESULT_PREFIX, payload)


if __name__ == "__main__":
    main()
