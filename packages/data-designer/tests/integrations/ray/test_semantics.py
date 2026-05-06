# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_ray_harness import FakeRayDataset, install_fake_ray

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.ray_fake


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


def _id_label_config_builder(stub_model_configs: Any) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="id_label", expr="row-{{ id }}"))
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
    install_fake_ray(monkeypatch)
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
    install_fake_ray(monkeypatch)
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
    fake_ray = install_fake_ray(monkeypatch)
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
    install_fake_ray(monkeypatch)
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
    fake_ray = install_fake_ray(monkeypatch)
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


def test_from_scratch_range_id_is_available_to_referenced_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    fake_ray.data.range_blocks = [
        lazy.pd.DataFrame({"id": [0, 1]}),
        lazy.pd.DataFrame({"id": [2, 3]}),
    ]
    designer = _designer(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        backend=RayBackend(batch_size=2),
    )

    output_df = (
        designer.create(_id_label_config_builder(stub_model_configs), num_records=4)
        .load_dataset()
        .to_pandas()
        .sort_values("id")
        .reset_index(drop=True)
    )

    assert output_df.to_dict(orient="records") == [
        {"id": 0, "id_label": "row-0"},
        {"id": 1, "id_label": "row-1"},
        {"id": 2, "id_label": "row-2"},
        {"id": 3, "id_label": "row-3"},
    ]


def test_seed_config_without_input_dataset_reads_partition_offsets_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
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
    fake_ray = install_fake_ray(monkeypatch)
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
    fake_ray = install_fake_ray(monkeypatch)
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
    install_fake_ray(monkeypatch)
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
