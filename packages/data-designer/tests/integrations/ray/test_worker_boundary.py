# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import SamplingStrategy
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import BatchStage
from data_designer.integrations.ray import RayBackend, RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.interface.data_designer import DataDesigner


class BoundaryRayDataset:
    def __init__(self, blocks: list[Any]) -> None:
        self.blocks = blocks
        self.map_batches_called = False

    def map_batches(self, fn: Any, **kwargs: Any) -> BoundaryRayDataset:
        del fn, kwargs
        self.map_batches_called = True
        raise AssertionError("driver preflight should fail before Ray maps batches")

    def num_blocks(self) -> int:
        return len(self.blocks)


class BoundaryRayDataModule:
    Dataset = BoundaryRayDataset

    def __init__(self) -> None:
        self.range_dataset: BoundaryRayDataset | None = None

    def range(self, num_records: int) -> BoundaryRayDataset:
        self.range_dataset = BoundaryRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})])
        return self.range_dataset


def _install_boundary_ray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = BoundaryRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


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


def _worker_options_from_designer(data_designer: DataDesigner) -> ray_backend_module._RayWorkerOptions:
    return ray_backend_module._RayWorkerOptions(
        model_providers=list(data_designer._model_providers),
        default_provider_name=data_designer._model_provider_registry.get_default_provider_name(),
        secret_resolver=data_designer._secret_resolver,
        seed_readers=ray_backend_module._clone_seed_readers_for_worker(
            data_designer._seed_reader_registry._readers.values()
        ),
        managed_assets_path=str(data_designer._managed_assets_path),
        person_reader=data_designer._person_reader,
        mcp_providers=list(data_designer._mcp_providers),
        run_config=data_designer._run_config,
    )


def test_seed_reader_worker_clone_removes_attachment_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clone = ray_backend_module._clone_seed_readers_for_worker([reader])[0]

    assert clone is not reader
    assert isinstance(clone, DataFrameSeedReader)
    assert getattr(clone, "_duckdb_conn") is None
    assert not hasattr(clone, "source")
    assert not hasattr(clone, "secret_resolver")
    assert getattr(reader, "_duckdb_conn") is not None
    assert hasattr(reader, "source")
    assert hasattr(reader, "secret_resolver")


def test_worker_options_and_map_payload_are_pickle_serializable(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    designer.set_run_config(RunConfig(buffer_size=2))
    worker_options = _worker_options_from_designer(designer)
    config_builder = _input_expression_config_builder(stub_model_configs)
    payload = ray_backend_module._compile_ray_execution_payload(
        config_builder=config_builder,
        worker_options=worker_options,
        use_input_dataset=True,
    )

    round_tripped = pickle.loads(pickle.dumps(payload))

    assert round_tripped.use_input_dataset is True
    assert round_tripped.worker_options.run_config.buffer_size == 2
    assert DataDesignerConfig.model_validate_json(round_tripped.config_json).seed_config is None


def test_worker_options_and_map_payload_are_cloudpickle_serializable_when_available(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    worker_options = _worker_options_from_designer(designer)
    config_builder = _input_expression_config_builder(stub_model_configs)
    payload = ray_backend_module._compile_ray_execution_payload(
        config_builder=config_builder,
        worker_options=worker_options,
        use_input_dataset=True,
    )

    round_tripped = cloudpickle.loads(cloudpickle.dumps(payload))

    assert round_tripped.worker_options.default_provider_name == worker_options.default_provider_name
    assert DataDesignerConfig.model_validate_json(round_tripped.config_json).seed_config is None


def test_in_memory_preview_artifact_storage_keeps_processor_outputs_off_disk() -> None:
    storage = ray_backend_module._InMemoryPreviewArtifactStorage()
    dataframe = lazy.pd.DataFrame({"value": [1, 2]})

    path = storage.write_parquet_file(
        "schema-transform.parquet",
        dataframe,
        BatchStage.PROCESSORS_OUTPUTS,
    )

    assert not path.exists()
    assert storage.list_processor_names() == ["schema-transform"]
    assert storage.load_processor_dataset("schema-transform").to_dict(orient="records") == [
        {"value": 1},
        {"value": 2},
    ]

    storage.clear()

    assert storage.list_processor_names() == []
    assert storage.load_processor_dataset("schema-transform").empty


def test_ray_batch_worker_reuses_setup_and_clears_block_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    payload = ray_backend_module._compile_ray_execution_payload(
        config_builder=_input_expression_config_builder(stub_model_configs),
        worker_options=_worker_options_from_designer(designer),
        use_input_dataset=True,
    )
    resource_provider_ids: list[int] = []
    model_registry_ids: list[int] = []
    artifact_storage_ids: list[int] = []
    seed_records: list[list[dict[str, Any]]] = []
    processor_names_seen: list[list[str]] = []

    class RecordingBuilder:
        def __init__(
            self,
            *,
            data_designer_config: DataDesignerConfig,
            resource_provider: Any,
            use_async: bool | None = None,
        ) -> None:
            assert use_async is True
            assert data_designer_config.seed_config is not None
            self._resource_provider: Any = resource_provider
            resource_provider_ids.append(id(resource_provider))
            model_registry_ids.append(id(resource_provider.model_registry))
            artifact_storage_ids.append(id(resource_provider.artifact_storage))
            seed_records.append(data_designer_config.seed_config.source.df.to_dict(orient="records"))

        def build_preview(self, *, num_records: int) -> Any:
            return lazy.pd.DataFrame({"row": list(range(num_records))})

        def process_preview(self, dataset: Any) -> Any:
            self._resource_provider.artifact_storage.write_parquet_file(
                "processor.parquet",
                dataset,
                BatchStage.PROCESSORS_OUTPUTS,
            )
            processor_names_seen.append(self._resource_provider.artifact_storage.list_processor_names())
            return dataset

    monkeypatch.setattr(ray_backend_module, "DatasetBuilder", RecordingBuilder)
    worker = ray_backend_module._RayBatchWorker(execution_payload=payload)

    first = worker(lazy.pd.DataFrame({"x": [1], "label": ["a"]}))
    second = worker(lazy.pd.DataFrame({"x": [2, 3], "label": ["b", "c"]}))

    assert first.to_dict(orient="records") == [{"row": 0}]
    assert second.to_dict(orient="records") == [{"row": 0}, {"row": 1}]
    assert len(set(resource_provider_ids)) == 1
    assert len(set(model_registry_ids)) == 1
    assert len(set(artifact_storage_ids)) == 1
    assert seed_records == [
        [{"x": 1, "label": "a"}],
        [{"x": 2, "label": "b"}, {"x": 3, "label": "c"}],
    ]
    assert processor_names_seen == [["processor"], ["processor"]]
    assert worker._artifact_storage.list_processor_names() == []
    assert worker._resource_provider.seed_reader is None
    for reader in worker._seed_reader_registry._readers.values():
        assert not hasattr(reader, "source")
        assert not hasattr(reader, "secret_resolver")


def test_ray_execution_payload_is_driver_config_snapshot(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    payload = ray_backend_module._compile_ray_execution_payload(
        config_builder=config_builder,
        worker_options=_worker_options_from_designer(designer),
        use_input_dataset=True,
    )
    config_builder.add_column(
        ExpressionColumnConfig(
            name="driver_only",
            expr="{{ label }}",
        )
    )

    output_batch = ray_backend_module._RayBatchWorker(execution_payload=payload)(
        lazy.pd.DataFrame({"x": [1], "label": ["a"]})
    )

    assert output_batch.to_dict(orient="records") == [{"x": 1, "label": "a", "x_label": "1-a"}]
    assert config_builder.get_seed_config() is None
    assert DataDesignerConfig.model_validate_json(payload.config_json).seed_config is None


def test_worker_batch_input_seed_does_not_leak_into_driver_builder(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    local_before = DataDesigner(
        artifact_path=tmp_path / "local-before",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-before"),
    )
    ray_boundary_designer = DataDesigner(
        artifact_path=tmp_path / "ray-boundary",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path / "ray-boundary"),
    )
    local_after = DataDesigner(
        artifact_path=tmp_path / "local-after",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-after"),
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    input_batch = lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})

    assert len(local_before.preview(stub_sampler_only_config_builder, num_records=1).dataset) == 1
    output_batch = ray_backend_module._generate_batch(
        input_batch,
        config_builder=config_builder,
        worker_options=_worker_options_from_designer(ray_boundary_designer),
        use_input_dataset=True,
    )
    assert len(local_after.preview(stub_sampler_only_config_builder, num_records=1).dataset) == 1

    assert output_batch.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]
    assert config_builder.get_seed_config() is None
    assert input_batch.to_dict(orient="records") == [
        {"x": 1, "label": "a"},
        {"x": 2, "label": "b"},
    ]


def test_driver_preflight_rejects_input_dataset_with_existing_seed_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    _install_boundary_ray(monkeypatch)
    input_dataset = BoundaryRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [99], "label": ["seed"]})))
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    with pytest.raises(RayBackendConfigurationError, match="input_dataset is used as the seed dataset"):
        designer.create(config_builder, input_dataset=input_dataset)

    assert input_dataset.map_batches_called is False


def test_driver_preflight_rejects_shuffled_seed_config_without_input_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = _install_boundary_ray(monkeypatch)
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.with_seed_dataset(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [99], "label": ["seed"]})),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    with pytest.raises(RayBackendConfigurationError, match="ordered sampling"):
        designer.create(config_builder, num_records=2)

    assert fake_ray.data.range_dataset is None


def test_worker_failure_includes_block_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    class FailingDatasetBuilder:
        def __init__(self, **_: Any) -> None:
            pass

        def build_preview(self, *, num_records: int) -> lazy.pd.DataFrame:
            del num_records
            raise RuntimeError("worker boom")

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )
    monkeypatch.setattr(ray_backend_module, "DatasetBuilder", FailingDatasetBuilder)

    with pytest.raises(RayDatasetGenerationError, match="range rows 5-6") as exc_info:
        ray_backend_module._generate_batch(
            lazy.pd.DataFrame({"id": [5, 6]}),
            config_builder=_input_expression_config_builder(stub_model_configs),
            worker_options=_worker_options_from_designer(designer),
            use_input_dataset=False,
        )

    assert "2 input row(s)" in str(exc_info.value)
    assert "worker boom" in str(exc_info.value)
