# Ray Data LLM / vLLM Integration Evaluation

Issue: [#52](https://github.com/eric-tramel/DataDesigner/issues/52)  
Date: 2026-04-25  
Status: planning hook implemented; narrow opt-in execution prototype implemented; live benchmark harness added

## Decision

Ray Data LLM/vLLM should not be added as a first-class `ModelProvider` yet. Keep external provider behavior on the existing `ModelFacade` and HTTP client adapters. Treat Ray Data LLM as a future RayBackend-only stage optimization for local GPU inference after the Ray planner can prove a column stage is eligible.

The safe slice is a capability and planning hook plus a tightly bounded execution prototype: `probe_ray_data_llm_capabilities()` detects the optional `ray.data.llm` surface at runtime, `assess_ray_data_llm_integration()` returns the current placement recommendation without importing Ray during package import, and `RayDataLLMStageOptions` lets `RayBackend` explicitly plan one opt-in local vLLM stage candidate. `execute=True` replaces the standard worker path only for one independent plain `LLMTextColumnConfig`; otherwise the backend fails fast or falls back to the existing `ModelFacade` path depending on the configured fallback policy.

## Sources Reviewed

- Ray Data LLM docs: `ray.data.llm` is a Ray Dataset batch inference pipeline that can run vLLM/SGLang engines directly or call hosted endpoints through processor configs. Current docs reviewed: Ray 2.55.1, while this repository's Ray extra requires `ray[data]>=2.54.0,<3`. See <https://docs.ray.io/en/latest/data/working-with-llms.html>.
- `vLLMEngineProcessorConfig` exposes stage-level knobs such as `model_source`, `batch_size`, `concurrency`, `engine_kwargs`, `task_type`, and `dynamic_lora_loading_path`. See <https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html>.
- vLLM engine arguments own GPU execution details such as tensor/pipeline parallelism and distributed executor backend selection. See <https://docs.vllm.ai/en/latest/configuration/engine_args/>.
- vLLM and Ray Serve can expose OpenAI-compatible HTTP APIs; those already fit DataDesigner through `ModelProvider(provider_type="openai")`. See <https://docs.vllm.ai/en/latest/serving/openai_compatible_server/>.

Local package introspection for the original execution prototype found `ray.data.llm` optional dependencies and `vllm` were not installed in the worker environment, so no live GPU benchmark was run in PR #132.

## Fit Analysis

### Provider Abstraction

DataDesigner generators obtain models through `ModelRegistry` and `ModelFacade`. That path owns request construction, retry behavior, adaptive throttling, health checks, parser correction, MCP/tool loops, and usage stats.

Ray Data LLM is not only a request adapter. It is a Ray Dataset processor stage with preprocess/postprocess functions and its own batching and resource controls. Modeling that as `ModelProvider(provider_type="ray-vllm")` would either bypass `ModelFacade` semantics or force a provider adapter to orchestrate Ray Dataset transforms from inside per-cell model calls, which would violate the current layering and execution model.

### Recommended Placement

The idiomatic path is a RayBackend optimization:

1. The interface still receives declarative model and column configs.
2. The Ray backend planner identifies eligible stages.
3. Eligible Ray Dataset blocks flow through a Ray Data LLM processor.
4. Ineligible columns continue through the existing engine block executor and `ModelFacade`.

This preserves the declarative config contract and avoids changing external provider behavior.

### Batching and Concurrency

DataDesigner currently controls model concurrency with `max_parallel_requests`, async task scheduling, worker batch size, and optional Ray-level provider throttling. Ray Data LLM controls throughput with processor `batch_size`, stage `concurrency`, vLLM engine kwargs, and GPU placement. A future implementation needs an explicit mapping rather than reusing provider fields implicitly.

### LoRA

Ray Data LLM exposes dynamic LoRA loading at the processor level. DataDesigner does not currently model per-row or per-column LoRA adapter selection in `ModelConfig`, so LoRA should remain a future opt-in Ray integration field rather than a hidden provider behavior.

### Embeddings, Classification, and Multimodal

Ray Data LLM supports generation, embedding, classification, and scoring task types. That is promising for chat/text and embedding columns, but multimodal support needs extra care because DataDesigner image context resolution, generated artifact paths, and response parsing currently live around the model facade.

## Initial Eligibility

A narrow future prototype should only consider columns that meet all of these conditions:

- RayBackend is active and `probe_ray_data_llm_capabilities().supports_local_vllm` is true.
- The model is local vLLM-capable and explicitly opted in through a Ray integration option.
- The column is an LLM text or embedding stage with no MCP tool config.
- The stage does not require parser correction loops, custom retry semantics, or per-cell provider extras that Ray Data LLM cannot preserve.
- The surrounding column dependency graph allows a whole Ray Dataset stage boundary.

All other model-generated columns should keep using the existing `ModelFacade`.

The current implementation records this eligibility through `plan_ray_data_llm_stage()` and exposes the resolved plan as `RayDatasetCreationResults.llm_stage_plan`. `RayDataLLMStageOptions(allow_model_facade_fallback=False)` turns the hook into a fail-fast verifier so a job does not silently run through the existing facade when the requested Ray Data LLM stage is unavailable or ineligible. `RayDataLLMStageOptions(execute=True)` is stricter: it requires the selected stage to be executable, imports optional Ray Data LLM and vLLM dependencies lazily, and bypasses the worker `ModelFacade` path only for the supported one-column prototype.

## Execution Prototype

The executable slice intentionally supports less than the planner can describe:

- exactly one selected `LLMTextColumnConfig`
- no prompt dependencies, tools, trace columns, reasoning-content side effects, `skip`, `drop`, `allow_resize`, or DataDesigner processor configs
- fixed chat-completion sampling parameters only; timeout, `extra_body`, and distribution-sampled parameters continue through the `ModelFacade` path
- no seed-window hydration and no `output_chunk_rows`

The Ray Data LLM processor preprocess function builds OpenAI-style chat messages and preserves the original Ray row in an internal field. The postprocess function writes the vLLM `generated_text` into the selected DataDesigner column and restores pre-existing non-selected columns. From-scratch range ids are not exposed unless they are needed internally for `preserve_order=True`.

## Non-Goals For This Slice

- No new `ModelProvider.provider_type`.
- No required `vllm` dependency.
- No changes to existing external OpenAI-compatible provider behavior.
- No planner rewrite or GPU benchmark in this PR.
- No usage-stat parity, trace parity, MCP/tool loop support, structured parsing, or processor support in the execution prototype.

## Benchmark Harness

`scripts/benchmarks/benchmark_ray_data_llm_vllm.py` now provides the explicit verification harness for [issue #118](https://github.com/eric-tramel/DataDesigner/issues/118). It compares:

- `ray-openai-vllm`: existing `RayBackend` worker execution through `ModelFacade` and an OpenAI-compatible local vLLM server/provider.
- `ray-data-llm`: `RayBackend` with `RayDataLLMStageOptions(enabled=True, execute=True)` using a Ray Data LLM `vLLMEngineProcessorConfig`.

The harness is live opt-in through `--run-live` or `DATA_DESIGNER_RUN_RAY_DATA_LLM_VLLM_BENCHMARK=1`, probes optional `ray.data.llm` and `vllm` imports before execution, can require visible NVIDIA GPUs with `--require-gpu`, and can either fail or emit a skipped JSON report when prerequisites are missing. CI covers the command behavior with fake Ray Data LLM and fake OpenAI-compatible provider wiring, so base installs still do not require vLLM.

Example live run:

```bash
DATA_DESIGNER_RUN_RAY_DATA_LLM_VLLM_BENCHMARK=1 \
uv run --all-packages --extra ray python scripts/benchmarks/benchmark_ray_data_llm_vllm.py \
  --model-source meta-llama/Llama-3.1-8B-Instruct \
  --provider-endpoint http://127.0.0.1:8000/v1 \
  --provider-model meta-llama/Llama-3.1-8B-Instruct \
  --num-records 128 \
  --batch-size 16 \
  --ray-data-llm-concurrency 1,2 \
  --engine-kwargs-json '{"tensor_parallel_size": 1}' \
  --require-gpu \
  --output-json /tmp/dd-ray-data-llm-vllm-benchmark.json
```

## Follow-Up Recommendation

[Issue #118](https://github.com/eric-tramel/DataDesigner/issues/118) should be closable after this harness is run in a real GPU/vLLM environment and records a successful comparison, or if any live-only parity gaps discovered by that run are split into concrete follow-up issues.

The remaining validation step needs an environment with `ray.data.llm` optional dependencies, vLLM, GPU resources, and a real local model source. Local package introspection on the current development environment found `ray.data.llm` imports fail because optional dependencies such as `boto3` are absent, so local tests use a fake Ray Data LLM processor and the real path fails fast when those optional dependencies are unavailable.
