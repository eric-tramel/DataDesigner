# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fake_ray_harness import FakeRayDataset, install_fake_ray

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from data_designer.integrations.ray import RayDatasetAnalysis, RayDatasetStats, RayTraceEvent, RayWorkerProfile
from data_designer.integrations.ray import llm as ray_llm_module

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
    assert modules["benchmark_ray_data_llm_vllm"].RESULT_PREFIX == "RAY_DATA_LLM_VLLM_BENCHMARK_RESULT="


def test_ray_data_llm_vllm_benchmark_requires_live_opt_in() -> None:
    module = _load_benchmark_module(
        "benchmark_ray_data_llm_vllm_opt_in",
        _benchmark_dir() / "benchmark_ray_data_llm_vllm.py",
    )

    with pytest.raises(SystemExit, match="live vLLM execution"):
        module.main(["--skip-provider-health-check"])


def test_ray_data_llm_vllm_benchmark_skips_missing_prereqs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module(
        "benchmark_ray_data_llm_vllm_skip",
        _benchmark_dir() / "benchmark_ray_data_llm_vllm.py",
    )
    output_json = tmp_path / "skip-report.json"
    unavailable_capabilities = module.RayDataLLMCapabilities(
        available=False,
        ray_version=None,
        missing_symbols=("build_processor", "vLLMEngineProcessorConfig"),
        has_build_processor=False,
        has_vllm_engine_processor=False,
        has_http_request_processor=False,
        has_serve_deployment_processor=False,
        import_error="ModuleNotFoundError: boto3",
    )

    monkeypatch.setattr(module, "probe_ray_data_llm_capabilities", lambda: unavailable_capabilities)
    monkeypatch.setattr(module, "_module_importable", lambda module_name: False)

    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                "--run-live",
                "--skip-missing-prereqs",
                "--skip-provider-health-check",
                "--model-source",
                "local-model",
                "--provider-model",
                "local-model",
                "--output-json",
                str(output_json),
            ]
        )

    assert exc_info.value.code == 0
    report = json.loads(output_json.read_text())
    assert report["summary"]["status"] == "skipped"
    assert any("ray.data.llm" in failure for failure in report["summary"]["preflight"]["failures"])
    assert any("vLLM is not importable" in failure for failure in report["summary"]["preflight"]["failures"])


def test_ray_data_llm_vllm_benchmark_runs_fake_provider_and_processor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module(
        "benchmark_ray_data_llm_vllm_fake",
        _benchmark_dir() / "benchmark_ray_data_llm_vllm.py",
    )
    fake_ray = install_fake_ray(monkeypatch)
    fake_ray.init = lambda **_: None
    fake_ray.shutdown = lambda: None
    processor_calls = _install_fake_ray_data_llm(monkeypatch)
    provider_requests: list[ChatCompletionRequest] = []

    def fake_completion(_: OpenAICompatibleClient, request: ChatCompletionRequest) -> ChatCompletionResponse:
        provider_requests.append(request)
        prompt = request.messages[-1]["content"]
        return ChatCompletionResponse(
            message=AssistantMessage(content=f"fake-provider:{prompt}"),
            usage=Usage(input_tokens=2, output_tokens=3, total_tokens=5),
        )

    async def fake_acompletion(_: OpenAICompatibleClient, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return fake_completion(_, request)

    monkeypatch.setattr(OpenAICompatibleClient, "completion", fake_completion)
    monkeypatch.setattr(OpenAICompatibleClient, "acompletion", fake_acompletion)
    output_json = tmp_path / "benchmark-report.json"

    module.main(
        [
            "--run-live",
            "--skip-provider-health-check",
            "--skip-gpu-check",
            "--model-source",
            "local-model",
            "--provider-model",
            "local-model",
            "--prompt",
            "Say hello.",
            "--num-records",
            "2",
            "--batch-size",
            "1",
            "--max-parallel-requests",
            "2",
            "--iterations",
            "1",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--managed-assets-path",
            str(tmp_path / "managed-assets"),
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text())
    backends = {result["backend"] for result in report["results"]}
    ray_data_llm_result = next(result for result in report["results"] if result["backend"] == "ray-data-llm")

    assert backends == {"ray-openai-vllm", "ray-data-llm"}
    assert report["summary"]["all_output_valid"] is True
    assert report["summary"]["baseline_backend"] == "ray-openai-vllm"
    assert "comparisons_vs_ray_openai_vllm" in report["summary"]
    assert ray_data_llm_result["llm_stage_plan"]["status"] == "eligible"
    assert ray_data_llm_result["llm_stage_plan"]["should_execute"] is True
    assert len(processor_calls) == 1
    assert len(provider_requests) == 2


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


def _install_fake_ray_data_llm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    processor_calls: list[dict[str, Any]] = []

    class FakeVLLMEngineProcessorConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeRayDataLLMProcessor:
        def __init__(self, *, preprocess: Any, postprocess: Any) -> None:
            self._preprocess = preprocess
            self._postprocess = postprocess

        def __call__(self, dataset: FakeRayDataset) -> FakeRayDataset:
            def preprocess_row(row: dict[str, Any]) -> dict[str, Any]:
                output = dict(row)
                output.update(self._preprocess(row))
                return output

            def run_vllm_batch(batch: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
                rows: list[dict[str, Any]] = []
                for row in batch.to_dict(orient="records"):
                    messages = row["messages"]
                    row["generated_text"] = f"fake-vllm:{messages[-1]['content']}"
                    rows.append(row)
                return lazy.pd.DataFrame(rows)

            return dataset.map(preprocess_row).map_batches(run_vllm_batch, batch_format="pandas").map(self._postprocess)

    def build_processor(
        config: FakeVLLMEngineProcessorConfig,
        preprocess: Any,
        postprocess: Any,
        **kwargs: Any,
    ) -> FakeRayDataLLMProcessor:
        processor_calls.append(
            {
                "config": config,
                "preprocess": preprocess,
                "postprocess": postprocess,
                "kwargs": kwargs,
            }
        )
        return FakeRayDataLLMProcessor(preprocess=preprocess, postprocess=postprocess)

    fake_ray_data_llm = types.SimpleNamespace(
        build_processor=build_processor,
        vLLMEngineProcessorConfig=FakeVLLMEngineProcessorConfig,
        HttpRequestProcessorConfig=object(),
        ServeDeploymentProcessorConfig=object(),
    )
    original_import_module = ray_llm_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray.data.llm":
            return fake_ray_data_llm
        if name == "vllm":
            return types.SimpleNamespace()
        return original_import_module(name)

    monkeypatch.setattr(ray_llm_module.importlib, "import_module", import_module)
    return processor_calls
