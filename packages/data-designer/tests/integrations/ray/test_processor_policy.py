# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from fake_ray_harness import FakeRayDataset, install_fake_ray

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import ProcessorConfig
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorDistributedSafety,
    ProcessorSideEffect,
    SchemaTransformProcessorConfig,
)
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.registry import ProcessorRegistry
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend, RayBackendConfigurationError
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray.processor_policy import validate_ray_safe_processors
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.ray_fake


class MissingSafetyProcessorConfig(ProcessorConfig):
    processor_type: Literal["missing_safety"] = "missing_safety"


class GlobalOrderProcessorConfig(ProcessorConfig):
    processor_type: Literal["global_order"] = "global_order"
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        partition_safe=True,
        requires_global_order=True,
        side_effects=ProcessorSideEffect.NONE,
        reason="Requires a globally ordered dataset.",
    )


class PartitionUnsafeProcessorConfig(ProcessorConfig):
    processor_type: Literal["partition_unsafe"] = "partition_unsafe"
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        partition_safe=False,
        side_effects=ProcessorSideEffect.NONE,
        reason="Requires cross-partition state.",
    )


class LegacyRaySafeProcessorConfig(ProcessorConfig):
    processor_type: Literal["legacy_ray_safe"] = "legacy_ray_safe"
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        ray_safe=True,
        side_effects=ProcessorSideEffect.NONE,
        reason="Uses legacy compatibility metadata.",
    )


class AfterGenerationProcessorConfig(ProcessorConfig):
    processor_type: Literal["ray_after_generation_test"] = "ray_after_generation_test"
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        partition_safe=True,
        side_effects=ProcessorSideEffect.NONE,
        reason="Metadata is otherwise safe, but the implementation needs the final dataset.",
    )


class AfterGenerationProcessor(Processor[AfterGenerationProcessorConfig]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("RayBackend validation must not instantiate processors.")

    def process_after_generation(self, data: Any) -> Any:
        return data


ProcessorRegistry.register(
    "ray_after_generation_test",
    AfterGenerationProcessor,
    AfterGenerationProcessorConfig,
    False,
)


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_expression_config_builder(stub_model_configs: Any) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="x_label",
            expr="{{ x }}-{{ label }}",
        )
    )
    config_builder.add_column(
        ExpressionColumnConfig(
            name="kept_label",
            expr="{{ label }}",
        )
    )
    return config_builder


def _designer(
    tmp_path: Path,
    stub_model_providers: Any,
    *,
    backend: RayBackend,
) -> DataDesigner:
    return DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )


def test_ray_backend_allows_drop_columns_processor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(DropColumnsProcessorConfig(name="drop-x-label", column_names=["x_label"]))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend(batch_size=2))

    results = designer.create(config_builder, input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    assert results.load_dataset().to_pandas().to_dict(orient="records") == [
        {"x": 1, "label": "a", "kept_label": "a"},
        {"x": 2, "label": "b", "kept_label": "b"},
    ]


def test_ray_backend_rejects_schema_transform_processor_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(
        SchemaTransformProcessorConfig(
            name="schema-transform",
            template={"combined": "{{ x_label }}"},
        )
    )
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="Unsupported processor") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    message = str(exc_info.value)
    assert "schema-transform (schema_transform)" in message
    assert "side_effects=dataset_artifact unsupported by RayBackend" in message
    assert "artifact collection" in message
    assert "allow_unsafe_processors=True" in message
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_rejects_after_generation_processor_implementation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(AfterGenerationProcessorConfig(name="custom-after-generation"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="process_after_generation") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    message = str(exc_info.value)
    assert "custom-after-generation (ray_after_generation_test)" in message
    assert "AfterGenerationProcessor.process_after_generation" in message
    assert "allow_unsafe_processors=True" not in message
    assert input_dataset.map_batches_kwargs is None
    assert input_dataset.map_batches_calls == []


def test_ray_backend_allow_unsafe_processors_does_not_bypass_after_generation_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(AfterGenerationProcessorConfig(name="custom-after-generation"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend(allow_unsafe_processors=True))

    with pytest.raises(RayBackendConfigurationError, match="process_after_generation"):
        designer.create(config_builder, input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is None
    assert input_dataset.map_batches_calls == []


def test_ray_backend_after_generation_guard_runs_before_input_dataset_map_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(AfterGenerationProcessorConfig(name="custom-after-generation"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    def fail_map_batches(*_: Any, **__: Any) -> FakeRayDataset:
        raise AssertionError("map_batches must not run before RayBackend processor validation.")

    monkeypatch.setattr(input_dataset, "map_batches", fail_map_batches)

    with pytest.raises(RayBackendConfigurationError, match="process_after_generation"):
        designer.create(config_builder, input_dataset=input_dataset)


def test_ray_backend_rejects_processor_without_distributed_safety_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(MissingSafetyProcessorConfig(name="custom-processor"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="missing distributed_safety metadata") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    assert "custom-processor (missing_safety)" in str(exc_info.value)
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_rejects_partition_unsafe_processor_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(PartitionUnsafeProcessorConfig(name="partition-unsafe-processor"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="partition_safe=False") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    assert "partition-unsafe-processor (partition_unsafe)" in str(exc_info.value)
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_rejects_global_order_processor_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(GlobalOrderProcessorConfig(name="global-order-processor"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="requires_global_order=True") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    assert "global-order-processor (global_order)" in str(exc_info.value)
    assert input_dataset.map_batches_kwargs is None


def test_ray_processor_policy_accepts_legacy_ray_safe_metadata(stub_model_configs: Any) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_processor(LegacyRaySafeProcessorConfig(name="legacy-processor"))

    validate_ray_safe_processors(config_builder)


def test_ray_backend_allow_unsafe_processors_bypasses_schema_transform_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(
        SchemaTransformProcessorConfig(
            name="schema-transform",
            template={"combined": "{{ x_label }}"},
        )
    )

    def passthrough_generate_batch(batch: Any, **_: Any) -> Any:
        return batch

    monkeypatch.setattr(ray_backend_module, "_generate_batch", passthrough_generate_batch)
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend(allow_unsafe_processors=True))

    results = designer.create(config_builder, input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    assert results.load_dataset().to_pandas().to_dict(orient="records") == [{"x": 1, "label": "a"}]
