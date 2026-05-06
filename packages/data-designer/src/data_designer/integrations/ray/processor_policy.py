# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.base import ProcessorConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import ProcessorDistributedSafety, ProcessorSideEffect
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.integrations.ray.errors import RayBackendConfigurationError

_RAY_ALWAYS_SUPPORTED_SIDE_EFFECTS = frozenset(
    {
        ProcessorSideEffect.NONE.value,
        ProcessorSideEffect.BATCH_ARTIFACT.value,
    }
)


def validate_ray_safe_processors(
    config_builder: DataDesignerConfigBuilder,
    *,
    allow_dataset_artifacts: bool = False,
) -> None:
    """Validate configured processor capabilities against Ray execution constraints."""
    unsupported: list[str] = []
    for processor in config_builder.get_processor_configs():
        safety = _get_distributed_safety(processor)
        if safety is None or not _is_supported_by_ray(safety, allow_dataset_artifacts=allow_dataset_artifacts):
            unsupported.append(
                _format_processor_violation(processor, safety, allow_dataset_artifacts=allow_dataset_artifacts)
            )

    if not unsupported:
        return

    raise RayBackendConfigurationError(
        "RayBackend supports only processors whose distributed_safety metadata is compatible "
        "with partitioned execution. "
        f"Unsupported processor(s): {'; '.join(unsupported)}. "
        "Enable write_artifacts=True for dataset artifact processors, review the processor's partition safety "
        "and side-effect metadata, or pass allow_unsafe_processors=True to bypass this experimental guard."
    )


def validate_no_ray_after_generation_processors(config_builder: DataDesignerConfigBuilder) -> None:
    """Reject processors whose implementation requires a completed dataset view.

    RayBackend currently runs DataDesigner generation inside Ray Data
    ``map_batches`` workers. A ``process_after_generation`` implementation would
    therefore receive only one Ray block instead of the complete final dataset.
    """
    unsupported: list[str] = []
    processor_registry = DataDesignerRegistry().processors
    for processor in config_builder.get_processor_configs():
        try:
            processor_impl = processor_registry.get_for_config_type(type(processor))
        except Exception:
            continue
        if getattr(processor_impl, "process_after_generation") is getattr(Processor, "process_after_generation"):
            continue
        unsupported.append(_format_processor_label(processor))

    if not unsupported:
        return

    raise RayBackendConfigurationError(
        "RayBackend does not support processors that implement process_after_generation because "
        "Ray workers process independent data blocks, not the completed dataset. "
        f"Unsupported processor(s): {'; '.join(unsupported)}. "
        "Use the local backend or remove the after-generation processor until Ray-native global "
        "processor execution is implemented."
    )


def _get_distributed_safety(processor: ProcessorConfig) -> ProcessorDistributedSafety | None:
    safety = getattr(processor, "distributed_safety", None)
    if isinstance(safety, ProcessorDistributedSafety):
        return safety
    return None


def _is_supported_by_ray(
    safety: ProcessorDistributedSafety,
    *,
    allow_dataset_artifacts: bool,
) -> bool:
    return (
        safety.partition_safe
        and not safety.requires_global_order
        and _side_effect_is_supported(safety.side_effects, allow_dataset_artifacts=allow_dataset_artifacts)
    )


def _side_effect_is_supported(side_effect: ProcessorSideEffect, *, allow_dataset_artifacts: bool) -> bool:
    if side_effect in _RAY_ALWAYS_SUPPORTED_SIDE_EFFECTS:
        return True
    return allow_dataset_artifacts and side_effect == ProcessorSideEffect.DATASET_ARTIFACT.value


def _format_processor_violation(
    processor: ProcessorConfig,
    safety: ProcessorDistributedSafety | None,
    *,
    allow_dataset_artifacts: bool,
) -> str:
    label = _format_processor_label(processor)
    if safety is None:
        return f"{label}: missing distributed_safety metadata"

    reasons: list[str] = []
    if not safety.partition_safe:
        reasons.append("partition_safe=False")
    if safety.requires_global_order:
        reasons.append("requires_global_order=True")
    if _side_effect_is_supported(safety.side_effects, allow_dataset_artifacts=allow_dataset_artifacts):
        reasons.append(f"side_effects={safety.side_effects}")
    else:
        reasons.append(f"side_effects={safety.side_effects} unsupported by RayBackend")
    if safety.reason is not None:
        reasons.append(f"reason={safety.reason}")
    return f"{label}: {', '.join(reasons)}"


def _format_processor_label(processor: ProcessorConfig) -> str:
    processor_type = getattr(processor.processor_type, "value", processor.processor_type)
    return f"{processor.name} ({processor_type})"
