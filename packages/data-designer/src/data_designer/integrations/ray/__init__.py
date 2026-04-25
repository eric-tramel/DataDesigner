# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable public imports for the experimental Ray integration.

Low-level telemetry normalizers, metric aggregators, and resolved planning
internals remain available from their defining modules for tests and advanced
debugging, but they are intentionally excluded from the package root surface.
"""

from __future__ import annotations

from data_designer.integrations.ray.backend import RayBackend, RayDatasetCreationResults
from data_designer.integrations.ray.errors import (
    RayBackendConfigurationError,
    RayDatasetGenerationError,
    RayIntegrationError,
    RayMetricsError,
)
from data_designer.integrations.ray.metrics import (
    RayDatasetMetrics,
    RayWorkerMetrics,
)
from data_designer.integrations.ray.observability import (
    RayDatasetAnalysis,
    RayThrottleSnapshot,
    RayTraceEvent,
    RayWorkerProfile,
)
from data_designer.integrations.ray.options import (
    RayBlockPlanning,
    RayExecutionOptions,
)

__all__ = [
    "RayBackend",
    "RayBackendConfigurationError",
    "RayBlockPlanning",
    "RayDatasetAnalysis",
    "RayDatasetCreationResults",
    "RayDatasetGenerationError",
    "RayDatasetMetrics",
    "RayExecutionOptions",
    "RayIntegrationError",
    "RayMetricsError",
    "RayThrottleSnapshot",
    "RayTraceEvent",
    "RayWorkerMetrics",
    "RayWorkerProfile",
]
