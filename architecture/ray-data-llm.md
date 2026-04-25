# Ray Data LLM / vLLM Integration Evaluation

Issue: [#52](https://github.com/eric-tramel/DataDesigner/issues/52)  
Date: 2026-04-25  
Status: planning hook implemented; execution integration deferred

## Decision

Ray Data LLM/vLLM should not be added as a first-class `ModelProvider` yet. Keep external provider behavior on the existing `ModelFacade` and HTTP client adapters. Treat Ray Data LLM as a future RayBackend-only stage optimization for local GPU inference after the Ray planner can prove a column stage is eligible.

The safe slice is a capability and planning hook: `probe_ray_data_llm_capabilities()` detects the optional `ray.data.llm` surface at runtime, `assess_ray_data_llm_integration()` returns the current placement recommendation without importing Ray during package import, and `RayDataLLMStageOptions` lets `RayBackend` explicitly plan one opt-in local vLLM stage candidate while continuing to execute through the existing `ModelFacade` unless that fallback is disabled.

## Sources Reviewed

- Ray Data LLM docs: `ray.data.llm` is a Ray Dataset batch inference pipeline that can run vLLM/SGLang engines directly or call hosted endpoints through processor configs. Current docs reviewed: Ray 2.55.1, while this repository's Ray extra requires `ray[data]>=2.54.0,<3`. See <https://docs.ray.io/en/latest/data/working-with-llms.html>.
- `vLLMEngineProcessorConfig` exposes stage-level knobs such as `model_source`, `batch_size`, `concurrency`, `engine_kwargs`, `task_type`, and `dynamic_lora_loading_path`. See <https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html>.
- vLLM engine arguments own GPU execution details such as tensor/pipeline parallelism and distributed executor backend selection. See <https://docs.vllm.ai/en/latest/configuration/engine_args/>.
- vLLM and Ray Serve can expose OpenAI-compatible HTTP APIs; those already fit DataDesigner through `ModelProvider(provider_type="openai")`. See <https://docs.vllm.ai/en/latest/serving/openai_compatible_server/>.

Local package introspection for this evaluation found `ray` and `vllm` were not installed in the worker environment, so no live GPU benchmark was run.

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

The current implementation records this eligibility through `plan_ray_data_llm_stage()` and exposes the resolved plan as `RayDatasetCreationResults.llm_stage_plan`. `RayDataLLMStageOptions(allow_model_facade_fallback=False)` turns the hook into a fail-fast verifier so a job does not silently run through the existing facade when the requested Ray Data LLM stage is unavailable or ineligible.

## Non-Goals For This Slice

- No new `ModelProvider.provider_type`.
- No required `vllm` dependency.
- No changes to existing external OpenAI-compatible provider behavior.
- No planner rewrite or GPU benchmark in this PR.

## Follow-Up Recommendation

[Issue #118](https://github.com/eric-tramel/DataDesigner/issues/118) tracks a RayBackend-only proof of concept and benchmark. The POC should compare the existing RayBackend plus OpenAI-compatible vLLM server path against a Ray Data LLM processor stage for a single independent text column, then expand only if the result preserves usage stats, errors, ordering, dropped-row behavior, and observability.

The remaining execution step needs an environment with `ray.data.llm` optional dependencies and a real local vLLM model source. Local package introspection on the current development environment found `ray.data.llm` imports fail because optional dependencies such as `boto3` are absent, so the merged hook intentionally fails fast or falls back instead of adding an unverified execution path.
