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
from data_designer.integrations.ray.errors import RayBackendConfigurationError

RayDataLLMPlacement = Literal["ray-backend-stage-optimization"]
RayDataLLMTaskType = Literal["generate", "embed", "classify", "score"]
RayDataLLMStageStatus = Literal["disabled", "unavailable", "eligible", "ineligible"]

_REQUIRED_RAY_DATA_LLM_SYMBOLS = ("build_processor", "vLLMEngineProcessorConfig")
_SUPPORTED_TASK_TYPES = ("generate", "embed", "classify", "score")
_RECOMMENDED_PLACEMENT: RayDataLLMPlacement = "ray-backend-stage-optimization"
_DEFAULT_SEMANTIC_CONTRACT = (
    "The prototype hook is planning-only; selected stages still execute through the existing ModelFacade path.",
    "Usage stats remain the ModelFacade source of truth until Ray Data LLM responses are normalized into model usage deltas.",
    "Ray Data LLM execution must wrap Ray/vLLM exceptions in RayDatasetGenerationError before it replaces a worker stage.",
    "Ordering and dropped-row semantics must remain owned by RayBackend ordering and row-count validation.",
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
    "A future prototype can add a RayBackend-only stage for one eligible independent LLM text column."
)


@dataclass(frozen=True, slots=True)
class RayDataLLMStageOptions:
    """Explicit opt-in controls for the RayBackend Ray Data LLM/vLLM prototype hook."""

    enabled: bool = False
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
        if self.enabled and not self.column_names and not self.model_aliases:
            raise RayBackendConfigurationError(
                "RayDataLLMStageOptions enabled=True requires column_names or model_aliases."
            )

    @property
    def has_explicit_opt_in(self) -> bool:
        """Return whether the RayBackend should evaluate Ray Data LLM stage eligibility."""
        return self.enabled

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


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


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
