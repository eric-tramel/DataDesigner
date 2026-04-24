# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest

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
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend, RayBackendConfigurationError
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.interface.data_designer import DataDesigner


class FakeRayDataset:
    def __init__(self, blocks: list[lazy.pd.DataFrame]) -> None:
        self.blocks = blocks
        self.map_batches_kwargs: dict[str, Any] | None = None

    def map_batches(self, fn: Any, **kwargs: Any) -> FakeRayDataset:
        self.map_batches_kwargs = kwargs
        fn_kwargs = kwargs.get("fn_kwargs") or {}
        return FakeRayDataset([fn(block, **fn_kwargs) for block in self.blocks])

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat(self.blocks, ignore_index=True)

    def to_arrow_refs(self) -> list[str]:
        return [f"arrow-ref-{index}" for index, _ in enumerate(self.blocks)]

    def num_blocks(self) -> int:
        return len(self.blocks)


class FakeRayDataModule:
    Dataset = FakeRayDataset

    def range(self, num_records: int) -> FakeRayDataset:
        return FakeRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})])


class MissingSafetyProcessorConfig(ProcessorConfig):
    processor_type: Literal["missing_safety"] = "missing_safety"


class GlobalOrderProcessorConfig(ProcessorConfig):
    processor_type: Literal["global_order"] = "global_order"
    distributed_safety: ClassVar[ProcessorDistributedSafety] = ProcessorDistributedSafety(
        ray_safe=True,
        requires_global_order=True,
        side_effects=ProcessorSideEffect.NONE,
        reason="Requires a globally ordered dataset.",
    )


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = FakeRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)


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
    _install_fake_ray(monkeypatch)
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
    _install_fake_ray(monkeypatch)
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
    assert "ray_safe=False" in message
    assert "side_effects=dataset_artifact" in message
    assert "worker-local storage" in message
    assert "allow_unsafe_processors=True" in message
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_rejects_processor_without_distributed_safety_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(MissingSafetyProcessorConfig(name="custom-processor"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="missing distributed_safety metadata") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    assert "custom-processor (missing_safety)" in str(exc_info.value)
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_rejects_global_order_processor_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(GlobalOrderProcessorConfig(name="global-order-processor"))
    designer = _designer(tmp_path, stub_model_providers, backend=RayBackend())

    with pytest.raises(RayBackendConfigurationError, match="requires_global_order=True") as exc_info:
        designer.create(config_builder, input_dataset=input_dataset)

    assert "global-order-processor (global_order)" in str(exc_info.value)
    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_allow_unsafe_processors_bypasses_schema_transform_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
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
