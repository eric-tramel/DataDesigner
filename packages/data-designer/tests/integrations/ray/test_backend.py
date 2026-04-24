# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import SchemaTransformProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import (
    RayBackend,
    RayBackendConfigurationError,
    RayDatasetCreationResults,
    RayDatasetGenerationError,
    RayDatasetMetrics,
)
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.interface.data_designer import DataDesigner


class FakeRayDataset:
    def __init__(self, blocks: list[lazy.pd.DataFrame], *, reverse_mapped_blocks: bool = False) -> None:
        self.blocks = blocks
        self.reverse_mapped_blocks = reverse_mapped_blocks
        self.map_batches_kwargs: dict[str, Any] | None = None

    def map_batches(self, fn: Any, **kwargs: Any) -> FakeRayDataset:
        self.map_batches_kwargs = kwargs
        blocks = _map_batches_blocks(fn, self.blocks, kwargs)
        if self.reverse_mapped_blocks:
            blocks.reverse()
        return FakeRayDataset(blocks, reverse_mapped_blocks=self.reverse_mapped_blocks)

    def zip(self, other: FakeRayDataset) -> FakeRayDataset:
        other_df = other.to_pandas()
        offset = 0
        blocks: list[lazy.pd.DataFrame] = []
        for block in self.blocks:
            block_len = len(block)
            other_block = other_df.iloc[offset : offset + block_len].reset_index(drop=True)
            blocks.append(lazy.pd.concat([block.reset_index(drop=True), other_block], axis=1))
            offset += block_len
        return FakeRayDataset(blocks, reverse_mapped_blocks=self.reverse_mapped_blocks)

    def to_arrow_refs(self) -> list[str]:
        return [f"arrow-ref-{i}" for i, _ in enumerate(self.blocks)]

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat(self.blocks, ignore_index=True)

    def sort(self, column: str) -> FakeRayDataset:
        sorted_df = self.to_pandas().sort_values(column, kind="stable").reset_index(drop=True)
        return FakeRayDataset([sorted_df])

    def drop_columns(self, columns: list[str]) -> FakeRayDataset:
        return FakeRayDataset([block.drop(columns=columns) for block in self.blocks])

    def num_blocks(self) -> int:
        return len(self.blocks)

    def count(self) -> int:
        return sum(len(block) for block in self.blocks)


class FakeRayDataModule:
    Dataset = FakeRayDataset

    def __init__(self) -> None:
        self.from_arrow_refs_input: list[Any] | None = None
        self.from_pandas_refs_input: list[Any] | None = None
        self.reverse_mapped_blocks = False

    def range(self, num_records: int) -> FakeRayDataset:
        return FakeRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})])

    def from_arrow_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_arrow_refs_input = refs
        return FakeRayDataset(
            [_arrow_ref_to_pandas(ref) for ref in refs],
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )

    def from_pandas_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_pandas_refs_input = refs
        return FakeRayDataset(list(refs), reverse_mapped_blocks=self.reverse_mapped_blocks)


class CountingRayDataset:
    def __init__(
        self,
        blocks: list[lazy.pd.DataFrame],
        data_module: CountingRayDataModule,
        *,
        parent: CountingRayDataset | None = None,
        map_fn: Any | None = None,
        map_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.blocks = blocks
        self.data_module = data_module
        self.parent = parent
        self.map_fn = map_fn
        self.map_kwargs = map_kwargs

    def map_batches(self, fn: Any, **kwargs: Any) -> CountingRayDataset:
        return CountingRayDataset([], self.data_module, parent=self, map_fn=fn, map_kwargs=kwargs)

    def to_arrow_refs(self) -> list[str]:
        refs: list[str] = []
        for block in self._evaluate():
            ref = f"arrow-ref-{len(self.data_module.ref_blocks)}"
            self.data_module.ref_blocks[ref] = block
            refs.append(ref)
        return refs

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat(self._evaluate(), ignore_index=True)

    def num_blocks(self) -> int:
        if self.parent is None:
            return len(self.blocks)
        return self.parent.num_blocks()

    def _evaluate(self) -> list[lazy.pd.DataFrame]:
        if self.parent is None:
            return self.blocks
        if self.map_fn is None:
            return self.parent._evaluate()
        return _map_batches_blocks(self.map_fn, self.parent._evaluate(), self.map_kwargs or {})


class CountingRayDataModule:
    Dataset = CountingRayDataset

    def __init__(self) -> None:
        self.from_arrow_refs_input: list[Any] | None = None
        self.ref_blocks: dict[str, lazy.pd.DataFrame] = {}

    def range(self, num_records: int) -> CountingRayDataset:
        return CountingRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})], self)

    def from_arrow_refs(self, refs: list[Any]) -> CountingRayDataset:
        self.from_arrow_refs_input = refs
        return CountingRayDataset([self.ref_blocks[ref] for ref in refs], self)


def _map_batches_blocks(fn: Any, blocks: list[lazy.pd.DataFrame], kwargs: dict[str, Any]) -> list[lazy.pd.DataFrame]:
    fn_kwargs = kwargs.get("fn_kwargs") or {}
    fn_constructor_kwargs = kwargs.get("fn_constructor_kwargs") or {}
    map_fn = fn(**fn_constructor_kwargs) if isinstance(fn, type) else fn
    return [map_fn(block, **fn_kwargs) for block in blocks]


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = FakeRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


def _install_counting_ray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = CountingRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


def _arrow_ref_to_pandas(ref: Any) -> lazy.pd.DataFrame:
    if hasattr(ref, "to_pandas"):
        return ref.to_pandas()
    return ref


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
    return config_builder


def test_ray_backend_uses_input_dataset_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, ray_remote_args={"num_cpus": 0.5}),
    )
    designer.set_run_config(RunConfig(buffer_size=2))

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_kwargs["num_cpus"] == 0.5
    assert "fn_kwargs" not in input_dataset.map_batches_kwargs
    assert input_dataset.map_batches_kwargs["fn_constructor_kwargs"]["execution_payload"].use_input_dataset is True
    assert "ray_remote_args" not in input_dataset.map_batches_kwargs
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_preserves_input_dataset_order_across_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}),
            lazy.pd.DataFrame({"x": [3], "label": ["c"]}),
            lazy.pd.DataFrame({"x": [4, 5], "label": ["d", "e"]}),
        ]
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
        {"x": 4, "label": "d", "x_label": "4-d"},
        {"x": 5, "label": "e", "x_label": "5-e"},
    ]


def test_ray_backend_can_sort_by_explicit_order_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"row_order": [2], "x": [3], "label": ["c"]}),
            lazy.pd.DataFrame({"row_order": [0, 1], "x": [1, 2], "label": ["a", "b"]}),
        ]
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, order_column="row_order", drop_order_column=True),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
    ]


def test_ray_backend_preserve_order_does_not_require_explicit_order_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"x": [1], "label": ["a"]}),
            lazy.pd.DataFrame({"x": [2], "label": ["b"]}),
        ],
        reverse_mapped_blocks=True,
    )
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(preserve_order=True),
    )
    config_builder = _input_expression_config_builder(stub_model_configs)

    output_df = designer.create(config_builder, input_dataset=input_dataset).load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_pandas_object_refs_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    input_refs = [lazy.pd.DataFrame({"x": [1], "label": ["a"]}), lazy.pd.DataFrame({"x": [2], "label": ["b"]})]
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, object_ref_format="pandas"),
    )

    results = designer.create(config_builder, input_dataset=input_refs)
    output_df = results.load_dataset().to_pandas()

    assert fake_ray.data.from_pandas_refs_input == input_refs
    assert fake_ray.data.from_arrow_refs_input is None
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_arrow_object_refs_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    input_refs = [
        lazy.pa.table({"x": [1], "label": ["a"]}),
        lazy.pa.table({"x": [2], "label": ["b"]}),
    ]
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, object_ref_format="arrow"),
    )

    results = designer.create(config_builder, input_dataset=input_refs)
    output_df = results.load_dataset().to_pandas()

    assert fake_ray.data.from_arrow_refs_input == input_refs
    assert fake_ray.data.from_pandas_refs_input is None
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_rejects_invalid_input_dataset_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(),
    )

    with pytest.raises(TypeError, match="ray.data.Dataset or a sequence of Ray ObjectRefs"):
        designer.create(stub_sampler_only_config_builder, input_dataset=object())


def test_ray_backend_can_return_arrow_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="arrow_refs"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output == ["arrow-ref-0"]
    assert results.to_arrow_refs() == ["arrow-ref-0"]
    assert isinstance(results.load_dataset(), FakeRayDataset)
    assert results.load_analysis() is None
    assert results.load_metrics() == RayDatasetMetrics(
        total_rows=2, blocks=1, elapsed_seconds=results.metrics.elapsed_seconds
    )


def test_ray_backend_arrow_refs_load_dataset_uses_materialized_refs_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_counting_ray(monkeypatch)
    generate_calls = 0

    def generate_batch(batch: lazy.pd.DataFrame, **_: Any) -> lazy.pd.DataFrame:
        nonlocal generate_calls
        generate_calls += 1
        output = batch.copy()
        output["generated"] = generate_calls
        return output

    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="arrow_refs"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output == ["arrow-ref-0"]
    assert generate_calls == 1

    dataset = results.load_dataset()
    output_df = dataset.to_pandas()

    assert fake_ray.data.from_arrow_refs_input == ["arrow-ref-0"]
    assert output_df.to_dict(orient="records") == [
        {"id": 0, "generated": 1},
        {"id": 1, "generated": 1},
    ]
    assert generate_calls == 1


def test_ray_results_to_arrow_refs_wraps_materialization_failures(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    class FailingArrowDataset:
        def to_arrow_refs(self) -> list[Any]:
            raise RuntimeError("object store unavailable")

    results = RayDatasetCreationResults(
        dataset=FailingArrowDataset(),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(),
    )

    with pytest.raises(RayDatasetGenerationError, match="materialize Arrow ObjectRefs") as exc_info:
        results.to_arrow_refs()

    assert "object store unavailable" in str(exc_info.value.__cause__)


def test_ray_backend_dataset_output_wraps_dataset_and_delegates_arrow_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="dataset"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output is results.load_dataset()
    assert isinstance(results.output, FakeRayDataset)
    assert results.to_arrow_refs() == ["arrow-ref-0"]
    assert results.load_analysis() is None
    assert results.load_metrics().total_rows == 2
    assert results.load_metrics().blocks == 1


def test_ray_backend_rejects_unsafe_processors_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(
        SchemaTransformProcessorConfig(
            name="schema-transform",
            template={"combined": "{{ x_label }}"},
        )
    )
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(),
    )

    with pytest.raises(RayBackendConfigurationError, match="Unsupported processor"):
        designer.create(config_builder, input_dataset=FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})]))


def test_ray_backend_import_is_lazy_when_ray_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = ray_backend_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray":
            raise ImportError
        return real_import_module(name)

    monkeypatch.setattr(ray_backend_module.importlib, "import_module", import_module)
    backend = RayBackend()

    with pytest.raises(ImportError, match="data-designer\\[ray\\]"):
        backend.create(
            data_designer=object(),
            config_builder=DataDesignerConfigBuilder(model_configs=[]),
            num_records=1,
            dataset_name="dataset",
        )


def test_local_backend_ignores_missing_ray_when_no_backend_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    real_import_module = ray_backend_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray":
            raise AssertionError("local DataDesigner.create() should not import ray")
        return real_import_module(name)

    monkeypatch.setattr(ray_backend_module.importlib, "import_module", import_module)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=1)

    assert len(results.load_dataset()) == 1
    assert results.load_analysis() is not None


def test_input_dataset_requires_backend(
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    with pytest.raises(ValueError, match="input_dataset requires an execution backend"):
        designer.create(stub_sampler_only_config_builder, input_dataset=object())


def test_same_process_local_ray_local_runs_do_not_share_backend_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_config_builder = _input_expression_config_builder(stub_model_configs)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    local_before = DataDesigner(
        artifact_path=tmp_path / "local-before",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-before"),
    )
    ray_designer = DataDesigner(
        artifact_path=tmp_path / "ray",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path / "ray"),
        backend=RayBackend(batch_size=1),
    )
    local_after = DataDesigner(
        artifact_path=tmp_path / "local-after",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-after"),
    )

    assert len(local_before.create(stub_sampler_only_config_builder, num_records=1).load_dataset()) == 1
    ray_records = ray_designer.create(input_config_builder, input_dataset=input_dataset).load_dataset().to_pandas()
    assert len(local_after.create(stub_sampler_only_config_builder, num_records=1).load_dataset()) == 1

    assert ray_records.to_dict(orient="records") == [{"x": 1, "label": "a", "x_label": "1-a"}]
    assert input_config_builder.get_seed_config() is None


def test_seed_readers_are_cloned_without_attachment_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clones = ray_backend_module._clone_seed_readers_for_worker([reader])

    assert len(clones) == 1
    assert clones[0] is not reader
    assert getattr(clones[0], "_duckdb_conn") is None
    assert not hasattr(clones[0], "source")
    assert not hasattr(clones[0], "secret_resolver")
