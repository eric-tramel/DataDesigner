# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

RayDataLLMPlacement = Literal["ray-backend-stage-optimization"]

_REQUIRED_RAY_DATA_LLM_SYMBOLS = ("build_processor", "vLLMEngineProcessorConfig")
_SUPPORTED_TASK_TYPES = ("generate", "embed", "classify", "score")
_RECOMMENDED_PLACEMENT: RayDataLLMPlacement = "ray-backend-stage-optimization"
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
    "A future prototype can add a RayBackend-only stage for eligible chat or embedding columns."
)


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


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None
