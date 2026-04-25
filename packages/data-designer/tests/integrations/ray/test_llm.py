# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from data_designer.integrations.ray import llm as ray_llm

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
