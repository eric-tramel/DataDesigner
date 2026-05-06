# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_designer.config.errors import InvalidConfigError
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorDistributedSafety,
    ProcessorSideEffect,
    ProcessorType,
    SchemaTransformProcessorConfig,
    get_processor_config_from_kwargs,
)


def test_drop_columns_processor_config_creation() -> None:
    config = DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["col1", "col2"])

    assert config.column_names == ["col1", "col2"]
    assert config.processor_type == ProcessorType.DROP_COLUMNS
    assert isinstance(config, ProcessorConfig)


def test_drop_columns_processor_config_validation() -> None:
    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        DropColumnsProcessorConfig(name="drop_columns_processor")


def test_drop_columns_processor_config_serialization() -> None:
    config = DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["col1", "col2"])

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["column_names"] == ["col1", "col2"]

    # Deserialize from dict
    config_restored = DropColumnsProcessorConfig.model_validate(config_dict)
    assert config_restored.column_names == config.column_names


def test_schema_transform_processor_config_creation() -> None:
    config = SchemaTransformProcessorConfig(
        name="output_format_processor",
        template={"text": "{{ col1 }}"},
    )

    assert config.template == {"text": "{{ col1 }}"}
    assert config.processor_type == ProcessorType.SCHEMA_TRANSFORM
    assert isinstance(config, ProcessorConfig)


def test_schema_transform_processor_config_validation() -> None:
    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        SchemaTransformProcessorConfig(name="schema_transform_processor")

    # Test invalid template raises error
    with pytest.raises(InvalidConfigError, match="Template must be JSON serializable"):
        SchemaTransformProcessorConfig(name="schema_transform_processor", template={"text": {1, 2, 3}})


def test_schema_transform_processor_config_serialization() -> None:
    config = SchemaTransformProcessorConfig(
        name="schema_transform_processor",
        template={"text": "{{ col1 }}"},
    )

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["template"] == {"text": "{{ col1 }}"}

    # Deserialize from dict
    config_restored = SchemaTransformProcessorConfig.model_validate(config_dict)
    assert config_restored.template == config.template


def test_processor_distributed_safety_uses_backend_neutral_partition_field() -> None:
    safety = ProcessorDistributedSafety(
        partition_safe=True,
        side_effects=ProcessorSideEffect.NONE,
        reason="Can run per partition.",
    )

    assert safety.partition_safe is True
    assert safety.ray_safe is True
    assert safety.model_dump() == {
        "partition_safe": True,
        "requires_global_order": False,
        "side_effects": "none",
        "reason": "Can run per partition.",
    }


def test_processor_distributed_safety_accepts_legacy_ray_safe_alias() -> None:
    safety = ProcessorDistributedSafety(
        ray_safe=False,
        side_effects=ProcessorSideEffect.EXTERNAL,
        reason="Legacy plugin metadata.",
    )

    assert safety.partition_safe is False
    assert safety.ray_safe is False
    assert "ray_safe" not in ProcessorDistributedSafety.model_fields
    assert "ray_safe" not in safety.model_dump()


def test_processor_distributed_safety_rejects_conflicting_legacy_alias() -> None:
    with pytest.raises(ValueError, match="deprecated alias"):
        ProcessorDistributedSafety(
            partition_safe=True,
            ray_safe=False,
            side_effects=ProcessorSideEffect.NONE,
        )


def test_builtin_processor_distributed_safety_metadata_is_backend_neutral() -> None:
    assert DropColumnsProcessorConfig.distributed_safety.partition_safe is True
    assert SchemaTransformProcessorConfig.distributed_safety.partition_safe is True
    assert SchemaTransformProcessorConfig.distributed_safety.side_effects == "dataset_artifact"
    assert "RayBackend" not in (SchemaTransformProcessorConfig.distributed_safety.reason or "")


def test_get_processor_config_from_kwargs() -> None:
    # Test successful creation
    config_drop_columns = get_processor_config_from_kwargs(
        ProcessorType.DROP_COLUMNS,
        name="drop_columns_processor",
        column_names=["col1"],
    )
    assert isinstance(config_drop_columns, DropColumnsProcessorConfig)
    assert config_drop_columns.column_names == ["col1"]
    assert config_drop_columns.processor_type == ProcessorType.DROP_COLUMNS

    config_schema_transform = get_processor_config_from_kwargs(
        ProcessorType.SCHEMA_TRANSFORM,
        name="output_format_processor",
        template={"text": "{{ col1 }}"},
    )
    assert isinstance(config_schema_transform, SchemaTransformProcessorConfig)
    assert config_schema_transform.template == {"text": "{{ col1 }}"}
    assert config_schema_transform.processor_type == ProcessorType.SCHEMA_TRANSFORM

    # Test with unknown processor type returns None
    from enum import Enum

    class UnknownProcessorType(str, Enum):
        UNKNOWN = "unknown"

    result = get_processor_config_from_kwargs(
        UnknownProcessorType.UNKNOWN, name="unknown_processor", column_names=["col1"]
    )
    assert result is None
