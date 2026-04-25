# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.errors import DataDesignerError
from data_designer.interface.errors import DataDesignerGenerationError


class RayIntegrationError(DataDesignerError):
    """Base error for Data Designer Ray integration failures."""


class RayBackendConfigurationError(RayIntegrationError):
    """Raised when the Ray backend is configured with unsupported options."""


class RayDatasetGenerationError(RayIntegrationError, DataDesignerGenerationError):
    """Raised when Ray-backed dataset generation fails."""


class RayBackendRowCountError(RayDatasetGenerationError):
    """Raised when a Ray worker violates a row-preserving map_batches contract."""


class RayMetricsError(RayIntegrationError):
    """Raised when Ray metrics cannot be normalized or aggregated."""
