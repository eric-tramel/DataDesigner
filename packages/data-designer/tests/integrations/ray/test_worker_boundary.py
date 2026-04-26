# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sys
import time
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
from data_designer.engine.dataset_builders.block_execution import (
    BlockExecutionChunk,
    BlockExecutionChunkStream,
    BlockExecutionResult,
    BlockExecutionStreamSummary,
)
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend, RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray import observability_collection as ray_observability_collection
from data_designer.integrations.ray import seed_planning as ray_seed_planning
from data_designer.integrations.ray import worker_pipeline as ray_worker_pipeline
from data_designer.interface.data_designer import DataDesigner

pytestmark = [pytest.mark.ray_fake, pytest.mark.ray_worker_boundary]


# Intentionally local: this fake fails if mapping starts, so worker-boundary
# tests can assert driver-side validation stops before Ray execution planning.
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
    runtime_context = data_designer._create_backend_runtime_context()
    return ray_backend_module._RayWorkerOptions(
        model_providers=list(runtime_context.model_providers),
        default_provider_name=runtime_context.default_provider_name,
        secret_resolver=runtime_context.secret_resolver,
        seed_readers=ray_seed_planning.clone_seed_readers_for_worker(runtime_context.seed_readers),
        managed_assets_path=str(runtime_context.managed_assets_path),
        person_reader=runtime_context.person_reader,
        mcp_providers=list(runtime_context.mcp_providers),
        run_config=runtime_context.run_config,
    )


def test_seed_reader_worker_clone_removes_attachment_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clone = ray_seed_planning.clone_seed_readers_for_worker([reader])[0]

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


def _block_result(*, dataframe: Any, input_rows: int, model_usage: dict[str, dict[str, Any]] | None = None) -> Any:
    output_rows = len(dataframe)
    dropped_rows = max(input_rows - output_rows, 0)
    return BlockExecutionResult(
        dataframe=dataframe,
        raw_dataframe=dataframe.copy(),
        task_traces=[],
        model_usage_stats=model_usage or {},
        model_usage_deltas={},
        processor_artifacts={},
        input_rows=input_rows,
        output_rows=output_rows,
        dropped_rows=dropped_rows,
        all_rows_dropped=input_rows > 0 and output_rows == 0,
        partial_rows_dropped=0 < output_rows < input_rows,
    )


def _block_chunk_stream(chunks: list[BlockExecutionChunk], *, input_rows: int) -> BlockExecutionChunkStream:
    summary_holder: dict[str, BlockExecutionStreamSummary] = {}

    def iterator() -> Any:
        output_rows = 0
        for chunk in chunks:
            output_rows += chunk.output_rows
            yield chunk
        summary_holder["summary"] = BlockExecutionStreamSummary(
            task_traces=[],
            model_usage_stats={},
            model_usage_deltas={},
            processor_artifacts={},
            input_rows=input_rows,
            output_rows=output_rows,
            dropped_rows=max(input_rows - output_rows, 0),
            all_rows_dropped=input_rows > 0 and output_rows == 0,
            partial_rows_dropped=0 < output_rows < input_rows,
        )

    return BlockExecutionChunkStream(iterator(), summary_holder)


def test_ray_batch_worker_delegates_to_engine_block_api(
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
    calls: list[dict[str, Any]] = []

    def execute_dataset_block(**kwargs: Any) -> BlockExecutionResult:
        calls.append(kwargs)
        num_records = kwargs["num_records"]
        return _block_result(dataframe=lazy.pd.DataFrame({"row": list(range(num_records))}), input_rows=num_records)

    monkeypatch.setattr(ray_backend_module, "execute_dataset_block", execute_dataset_block)
    worker = ray_backend_module._RayBatchWorker(execution_payload=payload)

    first = worker(lazy.pd.DataFrame({"x": [1], "label": ["a"]}))
    second = worker(lazy.pd.DataFrame({"x": [2, 3], "label": ["b", "c"]}))

    assert first.to_dict(orient="records") == [{"row": 0}]
    assert second.to_dict(orient="records") == [{"row": 0}, {"row": 1}]
    assert [call["input_frame"].to_dict(orient="records") for call in calls] == [
        [{"x": 1, "label": "a"}],
        [{"x": 2, "label": "b"}, {"x": 3, "label": "c"}],
    ]
    assert all(call["runtime_context"] is worker._worker_options for call in calls)
    assert all(call["options"].use_async is (sys.version_info >= (3, 11)) for call in calls)
    assert all(call["data_designer_config"].seed_config is None for call in calls)


def test_ray_batch_worker_uses_engine_block_stream_for_output_chunks(
    monkeypatch: pytest.MonkeyPatch,
    fake_ray_installer: Any,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = fake_ray_installer(with_remote=True)
    collector = ray_observability_collection._create_metrics_collector(fake_ray)
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
        output_chunk_rows=2,
    )
    calls: list[dict[str, Any]] = []

    def execute_dataset_block_stream(**kwargs: Any) -> BlockExecutionChunkStream:
        calls.append(kwargs)
        chunks = [
            BlockExecutionChunk(
                dataframe=lazy.pd.DataFrame({"row": [0, 1]}),
                raw_dataframe=lazy.pd.DataFrame({"row": [0, 1]}),
                row_group=0,
                input_start=0,
                input_rows=2,
                output_rows=2,
            ),
            BlockExecutionChunk(
                dataframe=lazy.pd.DataFrame({"row": [2]}),
                raw_dataframe=lazy.pd.DataFrame({"row": [2]}),
                row_group=1,
                input_start=2,
                input_rows=1,
                output_rows=1,
            ),
        ]
        return _block_chunk_stream(chunks, input_rows=3)

    monkeypatch.setattr(ray_backend_module, "execute_dataset_block_stream", execute_dataset_block_stream)
    worker = ray_backend_module._RayBatchWorker(
        execution_payload=payload,
        metrics_collector=collector,
        observability_options=ray_observability_collection._RayObservabilityOptions(profile_workers=True),
    )

    output = list(worker(lazy.pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})))
    observability = fake_ray.get(collector.observability_snapshot.remote())
    worker_profile = observability["worker_profiles"][0]

    assert [chunk.to_dict(orient="records") for chunk in output] == [[{"row": 0}, {"row": 1}], [{"row": 2}]]
    assert worker_profile["total_rows"] == 3
    assert worker_profile["memory_usage_bytes"] > 0
    assert worker_profile["input_memory_usage_bytes"] > 0
    assert worker_profile["process_maxrss_bytes"] is None or worker_profile["process_maxrss_bytes"] > 0
    assert len(calls) == 1
    assert calls[0]["rows_per_chunk"] == 2
    assert calls[0]["input_frame"].to_dict(orient="records") == [
        {"x": 1, "label": "a"},
        {"x": 2, "label": "b"},
        {"x": 3, "label": "c"},
    ]


def test_worker_generation_pipeline_core_runs_without_metrics_actor_or_ray_runtime(
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
        hidden_order_column=ray_worker_pipeline._RAY_INTERNAL_ROW_ID_COLUMN,
    )
    calls: list[dict[str, Any]] = []

    def execute_dataset_block(**kwargs: Any) -> BlockExecutionResult:
        calls.append(kwargs)
        return _block_result(dataframe=lazy.pd.DataFrame({"row": [0, 1]}), input_rows=2)

    pipeline = ray_worker_pipeline._RayWorkerGenerationPipeline(
        execution_payload=payload,
        execute_block=execute_dataset_block,
    )
    worker_batch = ray_worker_pipeline._RayWorkerBatch(
        dataframe=lazy.pd.DataFrame(
            {
                "x": [1, 2],
                "label": ["a", "b"],
                ray_worker_pipeline._RAY_INTERNAL_ROW_ID_COLUMN: [7, 8],
            }
        ),
        num_records=2,
        block_id="block-core",
        start_time=time.perf_counter(),
        worker_context={
            "worker_hostname": "host",
            "worker_pid": 123,
            "ray_task_id": None,
            "ray_node_id": None,
        },
    )

    result = pipeline.generate_rows(worker_batch)

    assert result.dataframe.to_dict(orient="records") == [
        {"row": 0, ray_worker_pipeline._RAY_INTERNAL_ROW_ID_COLUMN: 7},
        {"row": 1, ray_worker_pipeline._RAY_INTERNAL_ROW_ID_COLUMN: 8},
    ]
    assert len(calls) == 1
    assert calls[0]["input_frame"].to_dict(orient="records") == [
        {"x": 1, "label": "a"},
        {"x": 2, "label": "b"},
    ]
    assert calls[0]["runtime_context"] is pipeline.worker_options
    assert calls[0]["num_records"] == 2


def test_worker_observer_records_empty_batch_without_row_generation(
    fake_ray_installer: Any,
) -> None:
    fake_ray = fake_ray_installer(with_remote=True)
    collector = ray_observability_collection._create_metrics_collector(fake_ray)
    observer = ray_worker_pipeline._RayWorkerObserver(
        metrics_collector=collector,
        observability_options=ray_observability_collection._RayObservabilityOptions(
            profile_workers=True,
            trace_enabled=True,
        ),
    )
    worker_batch = ray_worker_pipeline._RayWorkerBatch(
        dataframe=lazy.pd.DataFrame({"x": []}),
        num_records=0,
        block_id="block-empty",
        start_time=time.perf_counter(),
        worker_context={
            "worker_hostname": "host",
            "worker_pid": 123,
            "ray_task_id": "task-id",
            "ray_node_id": "node-id",
        },
    )

    batch_observer = observer.begin_batch(worker_batch)
    batch_observer.record_empty(worker_batch)

    metrics = fake_ray.get(collector.snapshot.remote())[0]
    observability = fake_ray.get(collector.observability_snapshot.remote())
    assert metrics["block_id"] == "block-empty"
    assert metrics["empty_input"] is True
    assert metrics["blocks"] == 1
    assert metrics["input_rows"] == 0
    assert [event["event_type"] for event in observability["trace_events"]] == [
        "block_started",
        "block_completed",
    ]
    assert observability["worker_profiles"][0]["block_id"] == "block-empty"
    assert observability["worker_profiles"][0]["total_rows"] == 0


def test_ray_batch_worker_records_partial_row_drop_metrics(
    monkeypatch: pytest.MonkeyPatch,
    fake_ray_installer: Any,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = fake_ray_installer(with_remote=True)
    collector = ray_observability_collection._create_metrics_collector(fake_ray)
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

    def execute_dataset_block(**_: Any) -> BlockExecutionResult:
        return _block_result(dataframe=lazy.pd.DataFrame({"row": [0]}), input_rows=2)

    monkeypatch.setattr(ray_backend_module, "execute_dataset_block", execute_dataset_block)

    output = ray_backend_module._RayBatchWorker(execution_payload=payload, metrics_collector=collector)(
        lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
    )

    metrics = fake_ray.get(collector.snapshot.remote())[0]
    assert output.to_dict(orient="records") == [{"row": 0}]
    assert metrics["input_rows"] == 2
    assert metrics["output_rows"] == 1
    assert metrics["dropped_rows"] == 1
    assert metrics["partial_rows_dropped"] is True
    assert metrics["all_rows_dropped"] is False
    assert metrics["failed_blocks"] == 0


def test_ray_batch_worker_fails_all_row_drop_blocks(
    monkeypatch: pytest.MonkeyPatch,
    fake_ray_installer: Any,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = fake_ray_installer(with_remote=True)
    collector = ray_observability_collection._create_metrics_collector(fake_ray)
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

    def execute_dataset_block(**_: Any) -> BlockExecutionResult:
        return _block_result(dataframe=lazy.pd.DataFrame({"row": []}), input_rows=2)

    monkeypatch.setattr(ray_backend_module, "execute_dataset_block", execute_dataset_block)

    with pytest.raises(RayDatasetGenerationError, match="all input rows were dropped"):
        ray_backend_module._RayBatchWorker(execution_payload=payload, metrics_collector=collector)(
            lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})
        )

    metrics = fake_ray.get(collector.snapshot.remote())[0]
    assert metrics["input_rows"] == 2
    assert metrics["output_rows"] == 0
    assert metrics["dropped_rows"] == 2
    assert metrics["all_rows_dropped"] is True
    assert metrics["partial_rows_dropped"] is False
    assert metrics["failed_blocks"] == 1


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
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    def fail_block(**_: Any) -> BlockExecutionResult:
        raise RuntimeError("worker boom")

    monkeypatch.setattr(ray_backend_module, "execute_dataset_block", fail_block)

    with pytest.raises(RayDatasetGenerationError, match="range rows 5-6") as exc_info:
        ray_backend_module._generate_batch(
            lazy.pd.DataFrame({"id": [5, 6]}),
            config_builder=_input_expression_config_builder(stub_model_configs),
            worker_options=_worker_options_from_designer(designer),
            use_input_dataset=False,
        )

    assert "2 input row(s)" in str(exc_info.value)
    assert "worker boom" in str(exc_info.value)
