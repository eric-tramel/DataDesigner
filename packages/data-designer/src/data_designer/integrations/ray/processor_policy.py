# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ProcessorConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import ProcessorDistributedSafety
from data_designer.integrations.ray.errors import RayBackendConfigurationError


def validate_ray_safe_processors(config_builder: DataDesignerConfigBuilder) -> None:
    """Validate that configured processors are explicitly safe for Ray execution."""
    unsupported: list[str] = []
    for processor in config_builder.get_processor_configs():
        safety = _get_distributed_safety(processor)
        if safety is None or not safety.ray_safe or safety.requires_global_order:
            unsupported.append(_format_processor_violation(processor, safety))

    if not unsupported:
        return

    raise RayBackendConfigurationError(
        "RayBackend supports only processors explicitly declared distributed-safe. "
        f"Unsupported processor(s): {'; '.join(unsupported)}. "
        "Review the processor's distributed_safety metadata, or pass "
        "allow_unsafe_processors=True to bypass this experimental guard."
    )


def _get_distributed_safety(processor: ProcessorConfig) -> ProcessorDistributedSafety | None:
    safety = getattr(processor, "distributed_safety", None)
    if isinstance(safety, ProcessorDistributedSafety):
        return safety
    return None


def _format_processor_violation(processor: ProcessorConfig, safety: ProcessorDistributedSafety | None) -> str:
    processor_type = getattr(processor.processor_type, "value", processor.processor_type)
    label = f"{processor.name} ({processor_type})"
    if safety is None:
        return f"{label}: missing distributed_safety metadata"

    reasons: list[str] = []
    if not safety.ray_safe:
        reasons.append("ray_safe=False")
    if safety.requires_global_order:
        reasons.append("requires_global_order=True")
    reasons.append(f"side_effects={safety.side_effects}")
    if safety.reason is not None:
        reasons.append(f"reason={safety.reason}")
    return f"{label}: {', '.join(reasons)}"
