# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import data_designer.integrations.ray as ray

pytestmark = pytest.mark.ray_fake


def test_ray_package_root_exports_stable_public_api() -> None:
    expected_public_api = {
        "DataDesignerRayDatasink",
        "RayBackend",
        "RayBackendConfigurationError",
        "RayBackendRowCountError",
        "RayBlockPlanning",
        "RayDataCheckpointConfig",
        "RayDataContextOptions",
        "RayDataLLMCapabilities",
        "RayDataLLMIntegrationAssessment",
        "RayDataLLMStageCandidate",
        "RayDataLLMStageOptions",
        "RayDataLLMStagePlan",
        "RayDatasetAnalysis",
        "RayDatasetCreationResults",
        "RayDatasetGenerationError",
        "RayDatasetMetrics",
        "RayDatasetStats",
        "RayExecutionOptions",
        "RayExecutionResources",
        "RayInputRepartition",
        "RayIntegrationError",
        "RayMetricsError",
        "RayThrottleSnapshot",
        "RayTraceEvent",
        "RayWorkerMetrics",
        "RayWorkerProfile",
        "assess_ray_data_llm_integration",
        "plan_ray_data_llm_stage",
        "probe_ray_data_llm_capabilities",
    }

    assert set(ray.__all__) == expected_public_api
    for name in expected_public_api:
        assert getattr(ray, name).__name__ == name


def test_ray_package_root_does_not_export_internal_helpers() -> None:
    internal_helpers = {
        "RayResolvedBlockPlan",
        "aggregate_ray_metrics",
        "normalize_ray_throttle_snapshot",
        "normalize_ray_trace_event",
        "normalize_ray_worker_metrics",
        "normalize_ray_worker_profile",
        "RayResultArtifacts",
    }

    assert internal_helpers.isdisjoint(ray.__all__)
    for name in internal_helpers:
        assert not hasattr(ray, name)
