# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from data_designer.config.column_configs import EmbeddingColumnConfig, LLMTextColumnConfig, TraceType
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.engine.column_generators.utils.prompt_renderer import (
    PromptType,
    RecordBasedPromptRenderer,
    create_response_recipe,
)
from data_designer.engine.processing.utils import deserialize_json_values
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError

RayDataLLMPlacement = Literal["ray-backend-stage-optimization"]
RayDataLLMTaskType = Literal["generate", "embed", "classify", "score"]
RayDataLLMStageStatus = Literal["disabled", "unavailable", "eligible", "ineligible"]

_REQUIRED_RAY_DATA_LLM_SYMBOLS = ("build_processor", "vLLMEngineProcessorConfig")
_SUPPORTED_TASK_TYPES = ("generate", "embed", "classify", "score")
_RECOMMENDED_PLACEMENT: RayDataLLMPlacement = "ray-backend-stage-optimization"
_DEFAULT_SEMANTIC_CONTRACT = (
    "Ray Data LLM execution is opt-in and limited to one independent plain LLMTextColumnConfig stage.",
    "Usage stats remain unavailable for the Ray Data LLM execution path until responses are normalized into model usage deltas.",
    "Ray Data LLM execution wraps Ray/vLLM exceptions in RayDatasetGenerationError before replacing a worker stage.",
    "Ordering semantics remain owned by RayBackend ordering; dropped-row and processor semantics are not supported by the prototype.",
    "Artifact and observability output must use existing RayBackend artifact columns, metrics, and dataset analysis payloads.",
)
_RECOMMENDATION = (
    "Keep Ray Data LLM/vLLM behind the Ray integration as an optional stage optimization. "
    "Do not model it as a general ModelProvider until DataDesigner can preserve ModelFacade semantics "
    "for retries, MCP/tool loops, correction loops, usage accounting, and provider throttling."
)
_BLOCKING_GAPS = (
    "Ray Data LLM is a Ray Dataset processor stage, while DataDesigner generators call the ModelFacade "
    "per generated cell inside the engine execution plan.",
    "A direct provider adapter would bypass existing ModelFacade contracts for retries, usage stats, "
    "MCP/tool calls, parser correction, and health checks.",
    "Stage-level execution needs planner support to select only eligible LLM columns and to preserve column "
    "dependencies, validation, dropped-row handling, and artifact semantics.",
)
_SAFE_FIRST_SLICE = (
    "Detect Ray Data LLM availability and keep external providers on the existing OpenAI-compatible facade. "
    "The executable prototype is RayBackend-only and supports one eligible independent LLM text column."
)
_ORIGINAL_ROW_KEY = "__data_designer_ray_llm_original_row"
_RAY_RANGE_ID_COLUMN = "id"


@dataclass(frozen=True, slots=True)
class RayDataLLMStageOptions:
    """Explicit opt-in controls for the RayBackend Ray Data LLM/vLLM prototype hook."""

    enabled: bool = False
    execute: bool = False
    model_source: str | None = None
    column_names: tuple[str, ...] = ()
    model_aliases: tuple[str, ...] = ()
    task_type: RayDataLLMTaskType = "generate"
    batch_size: int | None = None
    concurrency: int | tuple[int, int] | tuple[int, int, int] | None = None
    engine_kwargs: Mapping[str, Any] | None = None
    dynamic_lora_loading_path: str | None = None
    allow_model_facade_fallback: bool = True

    def __post_init__(self) -> None:
        _validate_bool("enabled", self.enabled)
        _validate_bool("execute", self.execute)
        _validate_optional_non_empty_string("model_source", self.model_source)
        _validate_non_empty_string_tuple("column_names", self.column_names)
        _validate_non_empty_string_tuple("model_aliases", self.model_aliases)
        if self.task_type not in _SUPPORTED_TASK_TYPES:
            raise RayBackendConfigurationError(
                f"RayDataLLMStageOptions task_type must be one of {_SUPPORTED_TASK_TYPES!r}."
            )
        _validate_optional_positive_int("batch_size", self.batch_size)
        _validate_optional_concurrency(self.concurrency)
        if self.engine_kwargs is not None and not isinstance(self.engine_kwargs, Mapping):
            raise RayBackendConfigurationError("RayDataLLMStageOptions engine_kwargs must be a mapping.")
        _validate_optional_non_empty_string("dynamic_lora_loading_path", self.dynamic_lora_loading_path)
        _validate_bool("allow_model_facade_fallback", self.allow_model_facade_fallback)
        if self.enabled and self.model_source is None:
            raise RayBackendConfigurationError(
                "RayDataLLMStageOptions enabled=True requires model_source for the local vLLM engine."
            )
        if self.execute and not self.enabled:
            raise RayBackendConfigurationError("RayDataLLMStageOptions execute=True requires enabled=True.")
        if self.enabled and not self.column_names and not self.model_aliases:
            raise RayBackendConfigurationError(
                "RayDataLLMStageOptions enabled=True requires column_names or model_aliases."
            )

    @property
    def has_explicit_opt_in(self) -> bool:
        """Return whether the RayBackend should evaluate Ray Data LLM stage eligibility."""
        return self.enabled

    @property
    def should_execute(self) -> bool:
        """Return whether the RayBackend should replace worker execution with Ray Data LLM."""
        return self.enabled and self.execute

    def processor_config_kwargs(self) -> dict[str, Any]:
        """Return Ray Data LLM processor config kwargs for a future execution stage."""
        if self.model_source is None:
            return {}
        kwargs: dict[str, Any] = {
            "model_source": self.model_source,
            "task_type": self.task_type,
        }
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size
        if self.concurrency is not None:
            kwargs["concurrency"] = self.concurrency
        if self.engine_kwargs is not None:
            kwargs["engine_kwargs"] = dict(self.engine_kwargs)
        if self.dynamic_lora_loading_path is not None:
            kwargs["dynamic_lora_loading_path"] = self.dynamic_lora_loading_path
        return kwargs


@dataclass(frozen=True, slots=True)
class RayDataLLMCapabilities:
    """Runtime availability of Ray Data LLM APIs needed for local vLLM execution."""

    available: bool
    ray_version: str | None
    missing_symbols: tuple[str, ...]
    has_build_processor: bool
    has_vllm_engine_processor: bool
    has_http_request_processor: bool
    has_serve_deployment_processor: bool
    supported_task_types: tuple[str, ...] = _SUPPORTED_TASK_TYPES
    import_error: str | None = None

    @property
    def supports_local_vllm(self) -> bool:
        """Return whether Ray Data LLM has the local vLLM processor API."""
        return self.available and self.has_build_processor and self.has_vllm_engine_processor

    @property
    def supports_openai_compatible_endpoint(self) -> bool:
        """Return whether Ray Data LLM exposes an HTTP processor API."""
        return self.available and self.has_http_request_processor


@dataclass(frozen=True, slots=True)
class RayDataLLMIntegrationAssessment:
    """Recommendation for how Ray Data LLM should fit into DataDesigner."""

    capabilities: RayDataLLMCapabilities
    recommended_placement: RayDataLLMPlacement
    recommendation: str
    broad_integration_ready: bool
    safe_first_slice: str
    blocking_gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RayDataLLMStageCandidate:
    """Planning result for one model-generated column considered for Ray Data LLM."""

    column_name: str
    model_alias: str | None
    task_type: RayDataLLMTaskType
    model_source: str | None
    blocked_reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        """Return whether this column can use the future Ray Data LLM stage."""
        return not self.blocked_reasons


@dataclass(frozen=True, slots=True)
class RayDataLLMStagePlan:
    """RayBackend-only planning record for the optional Ray Data LLM stage."""

    status: RayDataLLMStageStatus
    options: RayDataLLMStageOptions
    capabilities: RayDataLLMCapabilities | None
    candidates: tuple[RayDataLLMStageCandidate, ...]
    selected_candidate: RayDataLLMStageCandidate | None
    disabled_reason: str | None
    semantic_contract: tuple[str, ...] = _DEFAULT_SEMANTIC_CONTRACT

    @property
    def enabled(self) -> bool:
        """Return whether the user explicitly enabled Ray Data LLM planning."""
        return self.options.enabled

    @property
    def has_eligible_stage(self) -> bool:
        """Return whether the planner found exactly one executable prototype stage."""
        return self.selected_candidate is not None

    @property
    def should_execute(self) -> bool:
        """Return whether this plan requests Ray Data LLM execution."""
        return self.options.should_execute


@dataclass(frozen=True, slots=True)
class _RayDataLLMExecutionAPI:
    build_processor: Any
    vllm_engine_processor_config: Any


@dataclass(frozen=True, slots=True)
class _RayDataLLMTextPreprocessor:
    column_config: LLMTextColumnConfig
    jinja_rendering_engine: Any
    sampling_params: Mapping[str, Any]
    preserve_input_columns: bool
    hidden_order_column: str | None
    range_order_column: str

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        original_row = dict(row) if self.preserve_input_columns else {}
        if self.hidden_order_column is not None:
            if self.hidden_order_column in row:
                original_row[self.hidden_order_column] = row[self.hidden_order_column]
            elif self.range_order_column in row:
                original_row[self.hidden_order_column] = row[self.range_order_column]
        deserialized_record = deserialize_json_values(dict(row))
        renderer = RecordBasedPromptRenderer(
            response_recipe=create_response_recipe(self.column_config),
            error_message_context={
                "column_name": self.column_config.name,
                "column_type": self.column_config.column_type,
                "model_alias": str(self.column_config.model_alias),
            },
            jinja_rendering_engine=self.jinja_rendering_engine,
        )
        prompt = renderer.render(
            record=deserialized_record,
            prompt_template=self.column_config.prompt,
            prompt_type=PromptType.USER_PROMPT,
        )
        system_prompt = renderer.render(
            record=deserialized_record,
            prompt_template=self.column_config.system_prompt,
            prompt_type=PromptType.SYSTEM_PROMPT,
        )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt or ""})
        output: dict[str, Any] = {
            _ORIGINAL_ROW_KEY: original_row,
            "messages": messages,
        }
        if self.sampling_params:
            output["sampling_params"] = dict(self.sampling_params)
        return output


@dataclass(frozen=True, slots=True)
class _RayDataLLMTextPostprocessor:
    column_name: str

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if _ORIGINAL_ROW_KEY not in row:
            raise RayDatasetGenerationError("Ray Data LLM postprocess did not receive the preserved original row.")
        if "generated_text" not in row:
            raise RayDatasetGenerationError("Ray Data LLM vLLM response did not contain generated_text.")
        output = dict(row[_ORIGINAL_ROW_KEY])
        output[self.column_name] = row["generated_text"]
        return output


def probe_ray_data_llm_capabilities() -> RayDataLLMCapabilities:
    """Probe optional Ray Data LLM symbols without importing Ray at module import time."""
    ray_version = _package_version("ray")
    try:
        ray_data_llm = importlib.import_module("ray.data.llm")
    except Exception as exc:
        return RayDataLLMCapabilities(
            available=False,
            ray_version=ray_version,
            missing_symbols=_REQUIRED_RAY_DATA_LLM_SYMBOLS,
            has_build_processor=False,
            has_vllm_engine_processor=False,
            has_http_request_processor=False,
            has_serve_deployment_processor=False,
            import_error=f"{type(exc).__name__}: {exc}",
        )

    missing_symbols = tuple(symbol for symbol in _REQUIRED_RAY_DATA_LLM_SYMBOLS if not hasattr(ray_data_llm, symbol))
    has_build_processor = hasattr(ray_data_llm, "build_processor")
    has_vllm_engine_processor = hasattr(ray_data_llm, "vLLMEngineProcessorConfig")
    return RayDataLLMCapabilities(
        available=not missing_symbols,
        ray_version=ray_version,
        missing_symbols=missing_symbols,
        has_build_processor=has_build_processor,
        has_vllm_engine_processor=has_vllm_engine_processor,
        has_http_request_processor=hasattr(ray_data_llm, "HttpRequestProcessorConfig"),
        has_serve_deployment_processor=hasattr(ray_data_llm, "ServeDeploymentProcessorConfig"),
    )


def assess_ray_data_llm_integration(
    capabilities: RayDataLLMCapabilities | None = None,
) -> RayDataLLMIntegrationAssessment:
    """Return the current integration recommendation for Ray Data LLM/vLLM."""
    resolved_capabilities = capabilities or probe_ray_data_llm_capabilities()
    return RayDataLLMIntegrationAssessment(
        capabilities=resolved_capabilities,
        recommended_placement=_RECOMMENDED_PLACEMENT,
        recommendation=_RECOMMENDATION,
        broad_integration_ready=False,
        safe_first_slice=_SAFE_FIRST_SLICE,
        blocking_gaps=_BLOCKING_GAPS,
    )


def plan_ray_data_llm_stage(
    *,
    config_builder: DataDesignerConfigBuilder,
    options: RayDataLLMStageOptions,
    capabilities: RayDataLLMCapabilities | None = None,
) -> RayDataLLMStagePlan:
    """Plan the RayBackend-only Ray Data LLM prototype stage without executing it."""
    if not options.has_explicit_opt_in:
        return RayDataLLMStagePlan(
            status="disabled",
            options=options,
            capabilities=None,
            candidates=(),
            selected_candidate=None,
            disabled_reason="Ray Data LLM stage planning is disabled.",
        )

    resolved_capabilities = capabilities or probe_ray_data_llm_capabilities()
    if not resolved_capabilities.supports_local_vllm:
        return RayDataLLMStagePlan(
            status="unavailable",
            options=options,
            capabilities=resolved_capabilities,
            candidates=(),
            selected_candidate=None,
            disabled_reason="Ray Data LLM local vLLM processor APIs are unavailable.",
        )

    candidates = tuple(
        candidate
        for column_config in config_builder.get_column_configs()
        if (candidate := _assess_stage_candidate(column_config, options)) is not None
    )
    eligible_candidates = tuple(candidate for candidate in candidates if candidate.eligible)
    if len(eligible_candidates) == 1:
        return RayDataLLMStagePlan(
            status="eligible",
            options=options,
            capabilities=resolved_capabilities,
            candidates=candidates,
            selected_candidate=eligible_candidates[0],
            disabled_reason=None,
        )
    disabled_reason = (
        "Ray Data LLM prototype requires exactly one eligible independent LLM text column."
        if eligible_candidates
        else "No eligible Ray Data LLM stage candidate was found."
    )
    return RayDataLLMStagePlan(
        status="ineligible",
        options=options,
        capabilities=resolved_capabilities,
        candidates=candidates,
        selected_candidate=None,
        disabled_reason=disabled_reason,
    )


def validate_ray_data_llm_execution_plan(
    *,
    config_builder: DataDesignerConfigBuilder,
    plan: RayDataLLMStagePlan,
) -> None:
    """Validate the executable Ray Data LLM prototype slice.

    The planner can record broader future candidates. Actual execution is much
    narrower so the backend does not silently bypass ModelFacade-only semantics.
    """
    if not plan.should_execute:
        return
    if not plan.has_eligible_stage:
        reason = plan.disabled_reason or "no eligible Ray Data LLM stage candidate was found"
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution was requested but cannot be planned: "
            f"{reason} Disable execute=True to use the existing ModelFacade execution path."
        )
    selected_candidate = plan.selected_candidate
    if selected_candidate is None or selected_candidate.task_type != "generate":
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution currently supports only task_type='generate'."
        )

    column_configs = config_builder.get_column_configs()
    if len(column_configs) != 1:
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution currently supports configs with exactly one selected "
            "LLMTextColumnConfig and no other generated columns."
        )
    column_config = column_configs[0]
    if type(column_config) is not LLMTextColumnConfig or column_config.name != selected_candidate.column_name:
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution currently supports exactly one selected LLMTextColumnConfig."
        )
    if config_builder.get_processor_configs():
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution does not yet support DataDesigner processor configs."
        )

    model_config = _model_config_for_alias(config_builder, selected_candidate.model_alias)
    blocked_reasons = _text_execution_blocked_reasons(column_config, model_config, plan.options)
    if blocked_reasons:
        raise RayBackendConfigurationError(
            "RayBackend Ray Data LLM execution cannot preserve this LLM text column yet: " + " ".join(blocked_reasons)
        )


def apply_ray_data_llm_stage(
    dataset: Any,
    *,
    config_builder: DataDesignerConfigBuilder,
    plan: RayDataLLMStagePlan,
    preserve_input_columns: bool,
    jinja_rendering_engine: Any,
    hidden_order_column: str | None = None,
    range_order_column: str = _RAY_RANGE_ID_COLUMN,
) -> Any:
    """Apply the optional Ray Data LLM processor stage to a Ray Dataset."""
    validate_ray_data_llm_execution_plan(config_builder=config_builder, plan=plan)
    selected_candidate = plan.selected_candidate
    if selected_candidate is None:
        raise RayDatasetGenerationError("RayBackend Ray Data LLM execution has no selected stage candidate.")
    column_config = _selected_text_column_config(config_builder, selected_candidate)
    model_config = _model_config_for_alias(config_builder, selected_candidate.model_alias)
    execution_api = _load_ray_data_llm_execution_api()
    try:
        processor_config = execution_api.vllm_engine_processor_config(**plan.options.processor_config_kwargs())
        processor = execution_api.build_processor(
            processor_config,
            preprocess=_RayDataLLMTextPreprocessor(
                column_config=column_config,
                jinja_rendering_engine=jinja_rendering_engine,
                sampling_params=_sampling_params_from_model_config(model_config),
                preserve_input_columns=preserve_input_columns,
                hidden_order_column=hidden_order_column,
                range_order_column=range_order_column,
            ),
            postprocess=_RayDataLLMTextPostprocessor(column_name=column_config.name),
        )
        return processor(dataset)
    except RayDatasetGenerationError:
        raise
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend Ray Data LLM execution failed.") from exc


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _load_ray_data_llm_execution_api() -> _RayDataLLMExecutionAPI:
    try:
        ray_data_llm = importlib.import_module("ray.data.llm")
    except Exception as exc:
        raise RayDatasetGenerationError(
            "RayBackend Ray Data LLM execution requires Ray LLM optional dependencies. "
            "Install the Ray LLM extra, including dependencies such as boto3 and vLLM."
        ) from exc

    missing_symbols = tuple(symbol for symbol in _REQUIRED_RAY_DATA_LLM_SYMBOLS if not hasattr(ray_data_llm, symbol))
    if missing_symbols:
        raise RayDatasetGenerationError(
            "RayBackend Ray Data LLM execution requires ray.data.llm symbols "
            f"{_REQUIRED_RAY_DATA_LLM_SYMBOLS!r}; missing {missing_symbols!r}."
        )
    try:
        importlib.import_module("vllm")
    except Exception as exc:
        raise RayDatasetGenerationError(
            "RayBackend Ray Data LLM execution requires vLLM to be importable on the driver and Ray workers."
        ) from exc
    return _RayDataLLMExecutionAPI(
        build_processor=ray_data_llm.build_processor,
        vllm_engine_processor_config=ray_data_llm.vLLMEngineProcessorConfig,
    )


def _selected_text_column_config(
    config_builder: DataDesignerConfigBuilder,
    selected_candidate: RayDataLLMStageCandidate,
) -> LLMTextColumnConfig:
    for column_config in config_builder.get_column_configs():
        if type(column_config) is LLMTextColumnConfig and column_config.name == selected_candidate.column_name:
            return column_config
    raise RayBackendConfigurationError(
        f"RayBackend Ray Data LLM selected column {selected_candidate.column_name!r} was not found."
    )


def _model_config_for_alias(
    config_builder: DataDesignerConfigBuilder,
    model_alias: str | None,
) -> ModelConfig:
    if model_alias is None:
        raise RayBackendConfigurationError("RayBackend Ray Data LLM selected column has no model alias.")
    for model_config in config_builder.model_configs:
        if model_config.alias == model_alias:
            return model_config
    raise RayBackendConfigurationError(f"RayBackend Ray Data LLM model alias {model_alias!r} was not found.")


def _text_execution_blocked_reasons(
    column_config: LLMTextColumnConfig,
    model_config: ModelConfig,
    options: RayDataLLMStageOptions,
) -> list[str]:
    blocked_reasons: list[str] = []
    if column_config.drop:
        blocked_reasons.append("drop=True is not supported.")
    if column_config.skip is not None:
        blocked_reasons.append("skip expressions are not supported.")
    if column_config.allow_resize:
        blocked_reasons.append("allow_resize is not supported.")
    if not isinstance(model_config.inference_parameters, ChatCompletionInferenceParams):
        blocked_reasons.append("the selected model must use ChatCompletionInferenceParams.")
    if isinstance(options.concurrency, tuple) and len(options.concurrency) != 2:
        blocked_reasons.append("execution supports only integer concurrency or a 2-item autoscaling tuple.")
    blocked_reasons.extend(_sampling_params_blocked_reasons(model_config))
    return blocked_reasons


def _sampling_params_blocked_reasons(model_config: ModelConfig) -> list[str]:
    inference_params = model_config.inference_parameters
    if not isinstance(inference_params, ChatCompletionInferenceParams):
        return []
    blocked_reasons: list[str] = []
    if inference_params.timeout is not None:
        blocked_reasons.append("per-request timeout is not a vLLM sampling parameter.")
    if inference_params.extra_body:
        blocked_reasons.append("extra_body passthrough is not supported by the prototype.")
    for field_name in ("temperature", "top_p"):
        value = getattr(inference_params, field_name)
        if hasattr(value, "sample"):
            blocked_reasons.append(f"{field_name} distributions are not supported by the prototype.")
    return blocked_reasons


def _sampling_params_from_model_config(model_config: ModelConfig) -> dict[str, Any]:
    inference_params = model_config.inference_parameters
    if not isinstance(inference_params, ChatCompletionInferenceParams):
        return {}
    sampling_params: dict[str, Any] = {}
    if inference_params.temperature is not None:
        sampling_params["temperature"] = inference_params.temperature
    if inference_params.top_p is not None:
        sampling_params["top_p"] = inference_params.top_p
    if inference_params.max_tokens is not None:
        sampling_params["max_tokens"] = inference_params.max_tokens
    return sampling_params


def _assess_stage_candidate(column_config: Any, options: RayDataLLMStageOptions) -> RayDataLLMStageCandidate | None:
    model_alias = getattr(column_config, "model_alias", None)
    column_name = getattr(column_config, "name", None)
    if not _matches_explicit_selection(column_name=column_name, model_alias=model_alias, options=options):
        return None

    blocked_reasons: list[str] = []
    if type(column_config) is LLMTextColumnConfig:
        blocked_reasons.extend(_text_stage_blocked_reasons(column_config, options))
    elif isinstance(column_config, EmbeddingColumnConfig):
        blocked_reasons.extend(_embedding_stage_blocked_reasons(column_config, options))
    else:
        blocked_reasons.append(
            "Only LLMTextColumnConfig and EmbeddingColumnConfig are supported by this prototype slice."
        )
    return RayDataLLMStageCandidate(
        column_name=str(column_name),
        model_alias=str(model_alias) if model_alias is not None else None,
        task_type=options.task_type,
        model_source=options.model_source,
        blocked_reasons=tuple(blocked_reasons),
    )


def _matches_explicit_selection(
    *,
    column_name: Any,
    model_alias: Any,
    options: RayDataLLMStageOptions,
) -> bool:
    if options.column_names and column_name not in options.column_names:
        return False
    if options.model_aliases and model_alias not in options.model_aliases:
        return False
    return bool(options.column_names or options.model_aliases)


def _text_stage_blocked_reasons(
    column_config: LLMTextColumnConfig,
    options: RayDataLLMStageOptions,
) -> list[str]:
    blocked_reasons: list[str] = []
    if options.task_type != "generate":
        blocked_reasons.append("LLMTextColumnConfig requires task_type='generate'.")
    if column_config.required_columns:
        blocked_reasons.append("The prototype supports only independent text columns with no prompt dependencies.")
    if column_config.tool_alias is not None:
        blocked_reasons.append("MCP tool calls are not supported by the Ray Data LLM stage prototype.")
    if column_config.multi_modal_context:
        blocked_reasons.append("Multi-modal context is not supported by the Ray Data LLM stage prototype.")
    if column_config.with_trace != TraceType.NONE:
        blocked_reasons.append("Trace side-effect columns are not supported by the Ray Data LLM stage prototype.")
    if column_config.extract_reasoning_content:
        blocked_reasons.append(
            "Reasoning-content side-effect columns are not supported by the Ray Data LLM stage prototype."
        )
    return blocked_reasons


def _embedding_stage_blocked_reasons(
    column_config: EmbeddingColumnConfig,
    options: RayDataLLMStageOptions,
) -> list[str]:
    blocked_reasons: list[str] = []
    del column_config
    if options.task_type != "embed":
        blocked_reasons.append("EmbeddingColumnConfig requires task_type='embed'.")
    return blocked_reasons


def _validate_bool(field_name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise RayBackendConfigurationError(f"RayDataLLMStageOptions {field_name} must be a boolean.")


def _validate_optional_non_empty_string(field_name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise RayBackendConfigurationError(f"RayDataLLMStageOptions {field_name} must be a non-empty string.")


def _validate_non_empty_string_tuple(field_name: str, value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple):
        raise RayBackendConfigurationError(f"RayDataLLMStageOptions {field_name} must be a tuple of strings.")
    for item in value:
        if not isinstance(item, str) or not item:
            raise RayBackendConfigurationError(
                f"RayDataLLMStageOptions {field_name} must contain only non-empty strings."
            )


def _validate_optional_positive_int(field_name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise RayBackendConfigurationError(f"RayDataLLMStageOptions {field_name} must be a positive integer.")


def _validate_optional_concurrency(value: int | tuple[int, int] | tuple[int, int, int] | None) -> None:
    if value is None:
        return
    if isinstance(value, int):
        _validate_optional_positive_int("concurrency", value)
        return
    if not isinstance(value, tuple) or len(value) not in (2, 3):
        raise RayBackendConfigurationError(
            "RayDataLLMStageOptions concurrency must be a positive integer or a 2/3-item tuple."
        )
    for item in value:
        _validate_optional_positive_int("concurrency", item)
    if value[0] > value[1]:
        raise RayBackendConfigurationError("RayDataLLMStageOptions concurrency minimum must not exceed maximum.")
