# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ProcessorConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import ProcessorDistributedSafety, ProcessorSideEffect
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.registry.errors import NotFoundInRegistryError
from data_designer.integrations.ray.errors import RayBackendConfigurationError

_RAY_SUPPORTED_SIDE_EFFECTS = frozenset(
    {
        ProcessorSideEffect.NONE.value,
        ProcessorSideEffect.BATCH_ARTIFACT.value,
    }
)


def validate_ray_safe_processors(config_builder: DataDesignerConfigBuilder) -> None:
    """Validate configured processor capabilities against Ray execution constraints."""
    unsupported: list[str] = []
    for processor in config_builder.get_processor_configs():
        safety = _get_distributed_safety(processor)
        if safety is None or not _is_supported_by_ray(safety):
            unsupported.append(_format_processor_violation(processor, safety))

    if not unsupported:
        return

    raise RayBackendConfigurationError(
        "RayBackend supports only processors whose distributed_safety metadata is compatible "
        "with partitioned execution. "
        f"Unsupported processor(s): {'; '.join(unsupported)}. "
        "Review the processor's partition safety and side-effect metadata, or pass "
        "allow_unsafe_processors=True to bypass this experimental guard."
    )


def validate_no_after_generation_processors(config_builder: DataDesignerConfigBuilder) -> None:
    """Reject processors whose implementation requires full-dataset processing."""
    unsupported: list[str] = []
    registry = DataDesignerRegistry().processors
    for processor in config_builder.get_processor_configs():
        try:
            processor_cls = registry.get_for_config_type(type(processor))
        except NotFoundInRegistryError:
            continue
        if _overrides_process_after_generation(processor_cls):
            unsupported.append(_format_processor_implementation_violation(processor, processor_cls))

    if not unsupported:
        return

    raise RayBackendConfigurationError(
        "RayBackend does not support processors that override process_after_generation(), "
        "because RayBackend currently runs processor callbacks inside Ray map_batches workers "
        "where each callback sees only one Ray block. "
        f"Unsupported processor implementation(s): {'; '.join(unsupported)}. "
        "Remove these processors or run the dataset with a non-Ray backend."
    )


def _get_distributed_safety(processor: ProcessorConfig) -> ProcessorDistributedSafety | None:
    safety = getattr(processor, "distributed_safety", None)
    if isinstance(safety, ProcessorDistributedSafety):
        return safety
    return None


def _is_supported_by_ray(safety: ProcessorDistributedSafety) -> bool:
    return (
        safety.partition_safe
        and not safety.requires_global_order
        and safety.side_effects in _RAY_SUPPORTED_SIDE_EFFECTS
    )


def _format_processor_violation(processor: ProcessorConfig, safety: ProcessorDistributedSafety | None) -> str:
    processor_type = getattr(processor.processor_type, "value", processor.processor_type)
    label = f"{processor.name} ({processor_type})"
    if safety is None:
        return f"{label}: missing distributed_safety metadata"

    reasons: list[str] = []
    if not safety.partition_safe:
        reasons.append("partition_safe=False")
    if safety.requires_global_order:
        reasons.append("requires_global_order=True")
    if safety.side_effects in _RAY_SUPPORTED_SIDE_EFFECTS:
        reasons.append(f"side_effects={safety.side_effects}")
    else:
        reasons.append(f"side_effects={safety.side_effects} unsupported by RayBackend")
    if safety.reason is not None:
        reasons.append(f"reason={safety.reason}")
    return f"{label}: {', '.join(reasons)}"


def _overrides_process_after_generation(processor_cls: type[Processor]) -> bool:
    return getattr(processor_cls, "process_after_generation") is not getattr(Processor, "process_after_generation")


def _format_processor_implementation_violation(
    processor: ProcessorConfig,
    processor_cls: type[Processor],
) -> str:
    processor_type = getattr(processor.processor_type, "value", processor.processor_type)
    return f"{processor.name} ({processor_type}): {processor_cls.__name__}.process_after_generation"
