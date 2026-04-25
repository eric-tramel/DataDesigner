# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy, SeedConfig
from data_designer.config.seed_source import DirectorySeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing.seed_readers import LineFanoutDirectorySeedReader
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray.seed_planning import (
    RaySeedWindow,
    clone_seed_readers_for_worker,
    get_contiguous_range_offsets,
    plan_seed_execution,
    read_seed_window_dataframe,
    resolve_seed_config_index_range,
)
from data_designer.interface.data_designer import DataDesigner

pytestmark = [pytest.mark.ray_fake, pytest.mark.ray_worker_boundary]


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_context(
    *,
    tmp_path: Path,
    stub_model_providers: Any,
    seed_readers: list[Any] | None = None,
) -> Any:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        seed_readers=seed_readers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    return designer._create_backend_runtime_context()


def test_clone_seed_readers_for_worker_detaches_driver_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clone = clone_seed_readers_for_worker([reader])[0]

    assert clone is not reader
    assert isinstance(clone, DataFrameSeedReader)
    assert getattr(clone, "_duckdb_conn") is None
    assert not hasattr(clone, "source")
    assert not hasattr(clone, "secret_resolver")
    assert getattr(reader, "_duckdb_conn") is not None
    assert hasattr(reader, "source")
    assert hasattr(reader, "secret_resolver")


def test_plan_seed_execution_uses_partition_window_for_dataframe_seed(
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    runtime_context = _runtime_context(tmp_path=tmp_path, stub_model_providers=stub_model_providers)
    seed_config = SeedConfig(
        source=DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [0, 1, 2, 3, 4]})),
        selection_strategy=PartitionBlock(index=1, num_partitions=2),
    )

    seed_plan = plan_seed_execution(runtime_context=runtime_context, seed_config=seed_config, num_records=2)

    assert seed_plan.input_dataframe is None
    assert seed_plan.seed_window == RaySeedWindow(start=2, size=3)


def test_plan_seed_execution_materializes_filesystem_seed_on_driver(
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "a.txt").write_text("alpha\nbeta", encoding="utf-8")
    (seed_dir / "b.txt").write_text("gamma\ndelta", encoding="utf-8")
    runtime_context = _runtime_context(
        tmp_path=tmp_path,
        stub_model_providers=stub_model_providers,
        seed_readers=[LineFanoutDirectorySeedReader()],
    )
    seed_config = SeedConfig(source=DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"))

    seed_plan = plan_seed_execution(runtime_context=runtime_context, seed_config=seed_config, num_records=3)

    assert seed_plan.seed_window is None
    assert seed_plan.input_dataframe is not None
    assert seed_plan.input_dataframe.to_dict(orient="records") == [
        {"relative_path": "a.txt", "line_index": 0, "line": "alpha"},
        {"relative_path": "a.txt", "line_index": 1, "line": "beta"},
        {"relative_path": "b.txt", "line_index": 0, "line": "gamma"},
    ]


def test_read_seed_window_dataframe_uses_contiguous_partition_offsets(
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    runtime_context = _runtime_context(tmp_path=tmp_path, stub_model_providers=stub_model_providers)
    seed_config = SeedConfig(source=DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [10, 11, 12, 13, 14]})))

    seed_df = read_seed_window_dataframe(
        seed_config=seed_config,
        seed_readers=clone_seed_readers_for_worker(runtime_context.seed_readers),
        secret_resolver=runtime_context.secret_resolver,
        dataframe=lazy.pd.DataFrame({"id": [2, 3]}),
        seed_window=RaySeedWindow(start=0, size=5),
    )

    assert seed_df.to_dict(orient="records") == [{"x": 12}, {"x": 13}]


def test_seed_window_offsets_reject_missing_or_noncontiguous_range_id() -> None:
    with pytest.raises(RayDatasetGenerationError, match="'id'"):
        get_contiguous_range_offsets(lazy.pd.DataFrame({"row": [0]}))

    with pytest.raises(RayDatasetGenerationError, match="contiguous"):
        get_contiguous_range_offsets(lazy.pd.DataFrame({"id": [0, 2]}))


def test_seed_range_resolution_validates_bounds() -> None:
    seed_config = SeedConfig(
        source=DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [0]})),
        selection_strategy=IndexRange(start=0, end=2),
    )

    with pytest.raises(RayBackendConfigurationError, match="out of bounds"):
        resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=2)


def test_seed_plan_rejects_shuffled_non_input_seed(
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    runtime_context = _runtime_context(tmp_path=tmp_path, stub_model_providers=stub_model_providers)
    seed_config = SeedConfig(
        source=DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [0, 1]})),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )

    with pytest.raises(RayBackendConfigurationError, match="ordered sampling"):
        plan_seed_execution(runtime_context=runtime_context, seed_config=seed_config, num_records=1)
