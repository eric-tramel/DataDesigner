# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from data_designer.config.base import ConfigBase, ProcessorConfig
from data_designer.config.errors import InvalidConfigError


class ProcessorType(str, Enum):
    """Enumeration of available processor types.

    Attributes:
        DROP_COLUMNS: Processor that removes specified columns from the output dataset.
        SCHEMA_TRANSFORM: Processor that creates a new dataset with a transformed schema using Jinja2 templates.
    """

    DROP_COLUMNS = "drop_columns"
    SCHEMA_TRANSFORM = "schema_transform"


class ProcessorSideEffect(str, Enum):
    """Side-effect behavior relevant to distributed processor execution."""

    NONE = "none"
    BATCH_ARTIFACT = "batch_artifact"
    DATASET_ARTIFACT = "dataset_artifact"
    EXTERNAL = "external"


class ProcessorDistributedSafety(ConfigBase):
    """Declarative distributed-execution safety metadata for processor configs.

    Processor configs declare this as class-level metadata so execution backends
    can fail closed for processors that have not been reviewed for distributed
    semantics. Legacy ``ray_safe`` constructor input is accepted as a deprecated
    alias for ``partition_safe`` to preserve compatibility with existing
    processor plugins.
    """

    partition_safe: bool = Field(
        description="Whether the processor is safe to run independently on distributed data partitions.",
    )
    requires_global_order: bool = Field(
        default=False,
        description="Whether the processor requires a globally ordered or complete dataset view.",
    )
    side_effects: ProcessorSideEffect = Field(
        default=ProcessorSideEffect.NONE,
        description="Side-effect behavior the backend must account for when distributing this processor.",
    )
    reason: str | None = Field(
        default=None,
        description="Human-readable explanation for unsafe or constrained distributed execution.",
    )

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_ray_safe(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "ray_safe" not in data:
            return data

        values = dict(data)
        ray_safe = values.pop("ray_safe")
        partition_safe = values.get("partition_safe")
        if partition_safe is not None and partition_safe != ray_safe:
            raise ValueError("ray_safe is a deprecated alias for partition_safe and cannot disagree with it")
        values["partition_safe"] = ray_safe
        return values

    @property
    def ray_safe(self) -> bool:
        """Deprecated compatibility alias for ``partition_safe``."""
        return self.partition_safe

    @ray_safe.setter
    def ray_safe(self, value: bool) -> None:
        self.partition_safe = value


def get_processor_config_from_kwargs(processor_type: ProcessorType, **kwargs: Any) -> ProcessorConfig:
    """Create a processor configuration from a processor type and keyword arguments.

    Args:
        processor_type: The type of processor to create.
        **kwargs: Additional keyword arguments passed to the processor constructor.

    Returns:
        A processor configuration object of the specified type.
    """
    if processor_type == ProcessorType.DROP_COLUMNS:
        return DropColumnsProcessorConfig(**kwargs)
    elif processor_type == ProcessorType.SCHEMA_TRANSFORM:
        return SchemaTransformProcessorConfig(**kwargs)


class DropColumnsProcessorConfig(ProcessorConfig):
    """Drop columns from the output dataset (prefer ``drop=True`` in the column config).

    This processor removes specified columns from the generated dataset. The dropped
    columns are saved separately in a `dropped-columns` directory for reference.
    When this processor is added via the config builder, the corresponding column
    configs are automatically marked with `drop = True`.

    Attributes:
        column_names (required): List of column names to remove from the output dataset.

    Inherited Attributes:
        name (required): Name of the processor.
    """

    column_names: list[str] = Field(description="List of column names to drop from the output dataset.")
    processor_type: Literal[ProcessorType.DROP_COLUMNS] = ProcessorType.DROP_COLUMNS
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        partition_safe=True,
        requires_global_order=False,
        side_effects=ProcessorSideEffect.BATCH_ARTIFACT,
        reason="Drops columns independently per batch; dropped-column artifacts are batch-local.",
    )


class SchemaTransformProcessorConfig(ProcessorConfig):
    """Configuration for transforming the dataset schema using Jinja2 templates.

    This processor creates a new dataset with a transformed schema. Each key in the
    template becomes a column in the output, and values are Jinja2 templates that
    can reference any column in the batch. The transformed dataset is written to
    a `processors-outputs/{processor_name}/` directory alongside the main dataset.

    Attributes:
        template (required): Dictionary defining the output schema. Keys are new column names,
            values are Jinja2 templates (strings, lists, or nested structures).
            Must be JSON-serializable.

    Inherited Attributes:
        name (required): Name of the processor.
    """

    template: dict[str, Any] = Field(
        ...,
        description="""
        Dictionary specifying columns and templates to use in the new dataset with transformed schema.

        Each key is a new column name, and each value is an object containing Jinja2 templates - for instance, a string or a list of strings.
        Values must be JSON-serializable.

        Example:

        ```python
        template = {
            "list_of_strings": ["{{ col1 }}", "{{ col2 }}"],
            "uppercase_string": "{{ col1 | upper }}",
            "lowercase_string": "{{ col2 | lower }}",
        }
        ```

        The above templates will create an new dataset with three columns: "list_of_strings", "uppercase_string", and "lowercase_string".
        References to columns "col1" and "col2" in the templates will be replaced with the actual values of the columns in the dataset.
        """,
    )
    processor_type: Literal[ProcessorType.SCHEMA_TRANSFORM] = ProcessorType.SCHEMA_TRANSFORM
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        partition_safe=True,
        requires_global_order=False,
        side_effects=ProcessorSideEffect.DATASET_ARTIFACT,
        reason="Writes dataset-level processor output artifacts that require backend artifact collection.",
    )

    @field_validator("template")
    def validate_template(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except TypeError as e:
            if "not JSON serializable" in str(e):
                raise InvalidConfigError("Template must be JSON serializable")
        return v
