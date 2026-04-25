# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from data_designer.config.column_configs import EmbeddingColumnConfig, LLMTextColumnConfig, TraceType
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, EmbeddingInferenceParams, ModelConfig
from data_designer.integrations.ray import llm as ray_llm
from data_designer.integrations.ray.errors import RayBackendConfigurationError

pytestmark = pytest.mark.ray_fake


def test_probe_ray_data_llm_capabilities_reports_missing_optional_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> Any:
        assert name == "ray.data.llm"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(ray_llm.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ray_llm, "_package_version", lambda package_name: None)

    capabilities = ray_llm.probe_ray_data_llm_capabilities()

    assert capabilities.available is False
    assert capabilities.ray_version is None
    assert capabilities.missing_symbols == ("build_processor", "vLLMEngineProcessorConfig")
    assert capabilities.supports_local_vllm is False
    assert capabilities.import_error is not None


def test_probe_ray_data_llm_capabilities_detects_vllm_processor_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(
        build_processor=object(),
        vLLMEngineProcessorConfig=object(),
        HttpRequestProcessorConfig=object(),
        ServeDeploymentProcessorConfig=object(),
    )

    def fake_import_module(name: str) -> Any:
        assert name == "ray.data.llm"
        return fake_module

    monkeypatch.setattr(ray_llm.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(ray_llm, "_package_version", lambda package_name: "2.54.0")

    capabilities = ray_llm.probe_ray_data_llm_capabilities()

    assert capabilities.available is True
    assert capabilities.ray_version == "2.54.0"
    assert capabilities.missing_symbols == ()
    assert capabilities.supports_local_vllm is True
    assert capabilities.supports_openai_compatible_endpoint is True
    assert capabilities.supported_task_types == ("generate", "embed", "classify", "score")


def test_assess_ray_data_llm_integration_keeps_vllm_out_of_provider_abstraction() -> None:
    capabilities = ray_llm.RayDataLLMCapabilities(
        available=True,
        ray_version="2.54.0",
        missing_symbols=(),
        has_build_processor=True,
        has_vllm_engine_processor=True,
        has_http_request_processor=False,
        has_serve_deployment_processor=False,
    )

    assessment = ray_llm.assess_ray_data_llm_integration(capabilities)

    assert assessment.capabilities is capabilities
    assert assessment.recommended_placement == "ray-backend-stage-optimization"
    assert assessment.broad_integration_ready is False
    assert "Do not model it as a general ModelProvider" in assessment.recommendation
    assert any("ModelFacade contracts" in gap for gap in assessment.blocking_gaps)


def test_ray_data_llm_stage_options_require_explicit_model_source() -> None:
    with pytest.raises(RayBackendConfigurationError, match="model_source"):
        ray_llm.RayDataLLMStageOptions(enabled=True, column_names=("summary",))


def test_ray_data_llm_stage_options_build_processor_config_kwargs() -> None:
    options = ray_llm.RayDataLLMStageOptions(
        enabled=True,
        model_source="local-model",
        column_names=("summary",),
        batch_size=8,
        concurrency=(1, 2),
        engine_kwargs={"tensor_parallel_size": 2},
        dynamic_lora_loading_path="/models/lora",
    )

    assert options.processor_config_kwargs() == {
        "model_source": "local-model",
        "task_type": "generate",
        "batch_size": 8,
        "concurrency": (1, 2),
        "engine_kwargs": {"tensor_parallel_size": 2},
        "dynamic_lora_loading_path": "/models/lora",
    }


def test_plan_ray_data_llm_stage_stays_disabled_without_opt_in() -> None:
    plan = ray_llm.plan_ray_data_llm_stage(
        config_builder=_text_config_builder(),
        options=ray_llm.RayDataLLMStageOptions(),
        capabilities=_available_capabilities(),
    )

    assert plan.status == "disabled"
    assert plan.selected_candidate is None
    assert plan.disabled_reason is not None
    assert plan.capabilities is None


def test_plan_ray_data_llm_stage_marks_unavailable_when_optional_surface_is_missing() -> None:
    capabilities = ray_llm.RayDataLLMCapabilities(
        available=False,
        ray_version="2.55.1",
        missing_symbols=("build_processor", "vLLMEngineProcessorConfig"),
        has_build_processor=False,
        has_vllm_engine_processor=False,
        has_http_request_processor=False,
        has_serve_deployment_processor=False,
        import_error="ModuleNotFoundError: boto3",
    )

    plan = ray_llm.plan_ray_data_llm_stage(
        config_builder=_text_config_builder(),
        options=ray_llm.RayDataLLMStageOptions(
            enabled=True,
            model_source="local-model",
            column_names=("summary",),
        ),
        capabilities=capabilities,
    )

    assert plan.status == "unavailable"
    assert plan.capabilities is capabilities
    assert plan.candidates == ()
    assert plan.selected_candidate is None


def test_plan_ray_data_llm_stage_selects_single_independent_text_column() -> None:
    plan = ray_llm.plan_ray_data_llm_stage(
        config_builder=_text_config_builder(),
        options=ray_llm.RayDataLLMStageOptions(
            enabled=True,
            model_source="local-model",
            column_names=("summary",),
        ),
        capabilities=_available_capabilities(),
    )

    assert plan.status == "eligible"
    assert plan.selected_candidate is not None
    assert plan.selected_candidate.column_name == "summary"
    assert plan.selected_candidate.model_alias == "text-model"
    assert plan.selected_candidate.task_type == "generate"
    assert plan.selected_candidate.blocked_reasons == ()


def test_plan_ray_data_llm_stage_reports_text_column_semantic_blockers() -> None:
    config_builder = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias="text-model",
                model="local-model",
                inference_parameters=ChatCompletionInferenceParams(),
            )
        ]
    )
    config_builder.add_column(
        LLMTextColumnConfig(
            name="summary",
            prompt="Summarize {{ source_text }}.",
            model_alias="text-model",
            with_trace=TraceType.LAST_MESSAGE,
        )
    )

    plan = ray_llm.plan_ray_data_llm_stage(
        config_builder=config_builder,
        options=ray_llm.RayDataLLMStageOptions(
            enabled=True,
            model_source="local-model",
            column_names=("summary",),
        ),
        capabilities=_available_capabilities(),
    )

    assert plan.status == "ineligible"
    assert plan.selected_candidate is None
    assert len(plan.candidates) == 1
    assert any("prompt dependencies" in reason for reason in plan.candidates[0].blocked_reasons)
    assert any("Trace side-effect" in reason for reason in plan.candidates[0].blocked_reasons)


def test_plan_ray_data_llm_stage_selects_embedding_column_with_embed_task() -> None:
    config_builder = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias="embedding-model",
                model="local-embedding-model",
                inference_parameters=EmbeddingInferenceParams(),
            )
        ]
    )
    config_builder.add_column(
        EmbeddingColumnConfig(
            name="text_embedding",
            target_column="source_text",
            model_alias="embedding-model",
        )
    )

    plan = ray_llm.plan_ray_data_llm_stage(
        config_builder=config_builder,
        options=ray_llm.RayDataLLMStageOptions(
            enabled=True,
            model_source="local-embedding-model",
            model_aliases=("embedding-model",),
            task_type="embed",
        ),
        capabilities=_available_capabilities(),
    )

    assert plan.status == "eligible"
    assert plan.selected_candidate is not None
    assert plan.selected_candidate.column_name == "text_embedding"
    assert plan.selected_candidate.task_type == "embed"


def _available_capabilities() -> ray_llm.RayDataLLMCapabilities:
    return ray_llm.RayDataLLMCapabilities(
        available=True,
        ray_version="2.55.1",
        missing_symbols=(),
        has_build_processor=True,
        has_vllm_engine_processor=True,
        has_http_request_processor=True,
        has_serve_deployment_processor=True,
    )


def _text_config_builder() -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias="text-model",
                model="local-model",
                inference_parameters=ChatCompletionInferenceParams(),
            )
        ]
    )
    config_builder.add_column(
        LLMTextColumnConfig(
            name="summary",
            prompt="Summarize briefly.",
            model_alias="text-model",
        )
    )
    return config_builder
