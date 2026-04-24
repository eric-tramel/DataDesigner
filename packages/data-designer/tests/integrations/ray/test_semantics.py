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
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner


class FakeRayDataset:
    def __init__(self, blocks: list[lazy.pd.DataFrame], *, reverse_mapped_blocks: bool = False) -> None:
        self.blocks = blocks
        self.reverse_mapped_blocks = reverse_mapped_blocks

    def map_batches(self, fn: Any, **kwargs: Any) -> FakeRayDataset:
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

    def to_arrow_refs(self) -> list[lazy.pd.DataFrame]:
        return self.blocks

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
        self.range_blocks: list[lazy.pd.DataFrame] | None = None
        self.reverse_mapped_blocks = False

    def range(self, num_records: int) -> FakeRayDataset:
        blocks = self.range_blocks or [lazy.pd.DataFrame({"id": list(range(num_records))})]
        return FakeRayDataset(blocks, reverse_mapped_blocks=self.reverse_mapped_blocks)

    def from_arrow_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_arrow_refs_input = refs
        return FakeRayDataset(
            [_arrow_ref_to_pandas(ref) for ref in refs],
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )

    def from_pandas_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_pandas_refs_input = refs
        return FakeRayDataset(list(refs), reverse_mapped_blocks=self.reverse_mapped_blocks)


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


def _arrow_ref_to_pandas(ref: Any) -> lazy.pd.DataFrame:
    if hasattr(ref, "to_pandas"):
        return ref.to_pandas()
    return ref


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _designer(
    *,
    tmp_path: Path,
    stub_model_providers: Any,
    backend: RayBackend | None,
) -> DataDesigner:
    return DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )


def _summary_config_builder(stub_model_configs: Any) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="summary",
            expr="{{ sku }}:{{ label }}",
        )
    )
    return config_builder


def _summary_seed_config_builder(stub_model_configs: Any, seed_df: lazy.pd.DataFrame) -> DataDesignerConfigBuilder:
    config_builder = _summary_config_builder(stub_model_configs)
    config_builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
    return config_builder


@pytest.mark.parametrize(
    ("drop_order_column", "expected_columns"),
    [
        (False, ["row_order", "sku", "label", "summary"]),
        (True, ["sku", "label", "summary"]),
    ],
)
def test_explicit_order_column_sorts_stably_and_can_be_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
    drop_order_column: bool,
    expected_columns: list[str],
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"row_order": [2, 4], "sku": ["c", "e"], "label": ["third", "fifth"]}),
            lazy.pd.DataFrame({"row_order": [0, 1], "sku": ["a", "b"], "label": ["first", "second"]}),
            lazy.pd.DataFrame({"row_order": [3], "sku": ["d"], "label": ["fourth"]}),
        ]
    )
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=2, order_column="row_order", drop_order_column=drop_order_column),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_dataset)
        .load_dataset()
        .to_pandas()
    )

    assert list(output_df.columns) == expected_columns
    assert output_df["sku"].tolist() == ["a", "b", "c", "d", "e"]
    assert output_df["label"].tolist() == ["first", "second", "third", "fourth", "fifth"]
    assert output_df["summary"].tolist() == ["a:first", "b:second", "c:third", "d:fourth", "e:fifth"]
    if not drop_order_column:
        assert output_df["row_order"].tolist() == [0, 1, 2, 3, 4]


def test_hidden_row_id_preserves_input_dataset_order_without_schema_pollution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"sku": ["a", "b"], "label": ["first", "second"]}),
            lazy.pd.DataFrame({"sku": ["c"], "label": ["third"]}),
            lazy.pd.DataFrame({"sku": ["d", "e"], "label": ["fourth", "fifth"]}),
        ],
        reverse_mapped_blocks=True,
    )
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, preserve_order=True),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_dataset)
        .load_dataset()
        .to_pandas()
    )

    assert list(output_df.columns) == ["sku", "label", "summary"]
    assert output_df["sku"].tolist() == ["a", "b", "c", "d", "e"]
    assert output_df["summary"].tolist() == [
        "a:first",
        "b:second",
        "c:third",
        "d:fourth",
        "e:fifth",
    ]
    assert not any(column.startswith("__data_designer_ray") for column in output_df.columns)


@pytest.mark.parametrize("object_ref_format", ["pandas", "arrow"])
def test_hidden_row_id_preserves_object_ref_order_without_schema_pollution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
    object_ref_format: str,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    fake_ray.data.reverse_mapped_blocks = True
    if object_ref_format == "pandas":
        input_refs: list[Any] = [
            lazy.pd.DataFrame({"sku": ["a"], "label": ["first"]}),
            lazy.pd.DataFrame({"sku": ["b"], "label": ["second"]}),
            lazy.pd.DataFrame({"sku": ["c"], "label": ["third"]}),
        ]
    else:
        input_refs = [
            lazy.pa.table({"sku": ["a"], "label": ["first"]}),
            lazy.pa.table({"sku": ["b"], "label": ["second"]}),
            lazy.pa.table({"sku": ["c"], "label": ["third"]}),
        ]
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, object_ref_format=object_ref_format, preserve_order=True),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_refs)
        .load_dataset()
        .to_pandas()
    )

    assert output_df.to_dict(orient="records") == [
        {"sku": "a", "label": "first", "summary": "a:first"},
        {"sku": "b", "label": "second", "summary": "b:second"},
        {"sku": "c", "label": "third", "summary": "c:third"},
    ]
    assert not any(column.startswith("__data_designer_ray") for column in output_df.columns)


def test_hidden_row_id_preserves_arrow_ref_output_order_without_schema_pollution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"sku": ["a"], "label": ["first"]}),
            lazy.pd.DataFrame({"sku": ["b"], "label": ["second"]}),
        ],
        reverse_mapped_blocks=True,
    )
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, output="arrow_refs", preserve_order=True),
    )

    results = designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert len(results.output) == 1
    assert output_df.to_dict(orient="records") == [
        {"sku": "a", "label": "first", "summary": "a:first"},
        {"sku": "b", "label": "second", "summary": "b:second"},
    ]
    assert not any(column.startswith("__data_designer_ray") for column in output_df.columns)


def test_hidden_row_id_preserves_from_scratch_order_without_schema_pollution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    fake_ray.data.range_blocks = [
        lazy.pd.DataFrame({"id": [0, 1]}),
        lazy.pd.DataFrame({"id": [2]}),
        lazy.pd.DataFrame({"id": [3, 4]}),
    ]
    fake_ray.data.reverse_mapped_blocks = True
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, preserve_order=True),
    )

    output_df = designer.create(stub_sampler_only_config_builder, num_records=5).load_dataset().to_pandas()

    assert len(output_df) == 5
    assert list(output_df.columns) == ["uuid", "category", "uniform"]
    assert not any(column.startswith("__data_designer_ray") for column in output_df.columns)


def test_seed_config_without_input_dataset_reads_partition_offsets_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    fake_ray.data.range_blocks = [
        lazy.pd.DataFrame({"id": [0, 1]}),
        lazy.pd.DataFrame({"id": [2, 3]}),
        lazy.pd.DataFrame({"id": [4]}),
    ]
    seed_df = lazy.pd.DataFrame(
        {
            "sku": ["a", "b", "c", "d", "e"],
            "label": ["first", "second", "third", "fourth", "fifth"],
        }
    )
    local_before = _designer(
        tmp_path=tmp_path / "local-before",
        stub_model_providers=stub_model_providers,
        backend=None,
    )
    ray_designer = _designer(
        tmp_path=tmp_path / "ray",
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=2),
    )
    local_after = _designer(
        tmp_path=tmp_path / "local-after",
        stub_model_providers=stub_model_providers,
        backend=None,
    )

    assert (
        len(local_before.preview(_summary_seed_config_builder(stub_model_configs, seed_df), num_records=1).dataset) == 1
    )
    output_df = (
        ray_designer.create(_summary_seed_config_builder(stub_model_configs, seed_df), num_records=5)
        .load_dataset()
        .to_pandas()
    )
    assert (
        len(local_after.preview(_summary_seed_config_builder(stub_model_configs, seed_df), num_records=1).dataset) == 1
    )

    assert output_df.to_dict(orient="records") == [
        {"sku": "a", "label": "first", "summary": "a:first"},
        {"sku": "b", "label": "second", "summary": "b:second"},
        {"sku": "c", "label": "third", "summary": "c:third"},
        {"sku": "d", "label": "fourth", "summary": "d:fourth"},
        {"sku": "e", "label": "fifth", "summary": "e:fifth"},
    ]


def test_pandas_object_refs_preserve_schema_column_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    input_refs = [
        lazy.pd.DataFrame({"sku": ["sku-1"], "quantity": [2], "label": ["alpha"]}),
        lazy.pd.DataFrame({"sku": ["sku-2"], "quantity": [5], "label": ["beta"]}),
    ]
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, object_ref_format="pandas"),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_refs)
        .load_dataset()
        .to_pandas()
    )

    assert fake_ray.data.from_pandas_refs_input is not None
    assert all(actual is expected for actual, expected in zip(fake_ray.data.from_pandas_refs_input, input_refs))
    assert fake_ray.data.from_arrow_refs_input is None
    assert list(output_df.columns) == ["sku", "quantity", "label", "summary"]
    assert output_df.to_dict(orient="records") == [
        {"sku": "sku-1", "quantity": 2, "label": "alpha", "summary": "sku-1:alpha"},
        {"sku": "sku-2", "quantity": 5, "label": "beta", "summary": "sku-2:beta"},
    ]


def test_arrow_object_refs_preserve_schema_column_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    input_refs = [
        lazy.pa.table({"label": ["alpha"], "sku": ["sku-1"], "quantity": [2]}),
        lazy.pa.table({"label": ["beta"], "sku": ["sku-2"], "quantity": [5]}),
    ]
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=1, object_ref_format="arrow"),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_refs)
        .load_dataset()
        .to_pandas()
    )

    assert fake_ray.data.from_arrow_refs_input == input_refs
    assert fake_ray.data.from_pandas_refs_input is None
    assert list(output_df.columns) == ["label", "sku", "quantity", "summary"]
    assert output_df.to_dict(orient="records") == [
        {"label": "alpha", "sku": "sku-1", "quantity": 2, "summary": "sku-1:alpha"},
        {"label": "beta", "sku": "sku-2", "quantity": 5, "summary": "sku-2:beta"},
    ]


@pytest.mark.parametrize("object_ref_format", ["pandas", "arrow"])
def test_object_refs_preserve_supported_dtypes_and_nulls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
    object_ref_format: str,
) -> None:
    _install_fake_ray(monkeypatch)
    input_refs: list[Any]
    if object_ref_format == "pandas":
        input_refs = [
            lazy.pd.DataFrame(
                {
                    "sku": ["sku-1", "sku-2"],
                    "label": ["alpha", "beta"],
                    "row_id": lazy.pd.Series([10, 11], dtype="int64"),
                    "score": lazy.pd.Series([1.5, None], dtype="float64"),
                    "active": lazy.pd.Series([True, False], dtype="bool"),
                    "comment": ["kept", None],
                }
            )
        ]
    else:
        input_refs = [
            lazy.pa.table(
                {
                    "sku": ["sku-1", "sku-2"],
                    "label": ["alpha", "beta"],
                    "row_id": lazy.pa.array([10, 11], type=lazy.pa.int64()),
                    "score": lazy.pa.array([1.5, None], type=lazy.pa.float64()),
                    "active": lazy.pa.array([True, False], type=lazy.pa.bool_()),
                    "comment": lazy.pa.array(["kept", None], type=lazy.pa.string()),
                }
            )
        ]
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=2, object_ref_format=object_ref_format),
    )

    output_df = (
        designer.create(_summary_config_builder(stub_model_configs), input_dataset=input_refs)
        .load_dataset()
        .to_pandas()
    )

    assert list(output_df.columns) == ["sku", "label", "row_id", "score", "active", "comment", "summary"]
    assert output_df["row_id"].dtype.kind in {"i", "u"}
    assert output_df["score"].dtype.kind == "f"
    assert output_df["active"].dtype.kind == "b"
    assert output_df["score"].isna().tolist() == [False, True]
    assert output_df["comment"].isna().tolist() == [False, True]
    assert output_df["row_id"].tolist() == [10, 11]
    assert output_df["active"].tolist() == [True, False]
