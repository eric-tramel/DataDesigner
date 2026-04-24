# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

from pydantic import PrivateAttr

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import ProcessorType
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import SeedConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import create_resource_provider
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.storage.media_storage import StorageMode
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray.metrics import RayDatasetMetrics, RayWorkerMetrics, aggregate_ray_metrics

if TYPE_CHECKING:
    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider


RayOutputMode = Literal["dataset", "arrow_refs"]
RayObjectRefInputFormat = Literal["arrow", "pandas"]


@dataclass(frozen=True)
class _RayWorkerOptions:
    model_providers: list[ModelProvider]
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: list[SeedReader]
    managed_assets_path: str
    person_reader: PersonReader | None
    mcp_providers: list[MCPProviderT]
    run_config: RunConfig


@dataclass(frozen=True)
class _RayExecutionPayload:
    config_json: str
    worker_options: _RayWorkerOptions
    use_input_dataset: bool


class RayDatasetCreationResults:
    """Results wrapper for Ray-resident Data Designer outputs."""

    def __init__(
        self,
        *,
        dataset: Any,
        config_builder: DataDesignerConfigBuilder,
        metrics: RayDatasetMetrics,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        output: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self._config_builder = config_builder
        self._driver_metrics = metrics
        self._metrics_cache: RayDatasetMetrics | None = None
        self._ray = ray
        self._metrics_collector = metrics_collector
        self._output = output

    def load_dataset(self) -> Any:
        """Return the Ray Dataset without materializing it on the driver."""
        return self.dataset

    def load_analysis(self) -> None:
        """Ray-resident jobs do not produce local profiler artifacts."""
        return None

    def load_metrics(self) -> RayDatasetMetrics:
        """Return driver-visible Ray execution metrics."""
        if self._metrics_cache is not None:
            return self._metrics_cache
        if self._ray is None or self._metrics_collector is None:
            return self._driver_metrics
        try:
            payloads = self._ray.get(self._metrics_collector.snapshot.remote())
            if not payloads:
                self._materialize_dataset_for_metrics()
                payloads = self._ray.get(self._metrics_collector.snapshot.remote())
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to load worker metrics.") from exc
        if not payloads:
            return self._driver_metrics
        worker_metrics = aggregate_ray_metrics(payloads)
        self._metrics_cache = _merge_driver_and_worker_metrics(self._driver_metrics, worker_metrics)
        return self._metrics_cache

    @property
    def metrics(self) -> RayDatasetMetrics:
        """Return the latest available Ray execution metrics."""
        return self.load_metrics()

    def _materialize_dataset_for_metrics(self) -> None:
        materialize = getattr(self.dataset, "materialize", None)
        if not callable(materialize):
            return
        self.dataset = materialize()

    def to_arrow_refs(self) -> list[Any]:
        """Return Ray ObjectRefs containing PyArrow tables, one per Ray block."""
        if self._output is not None:
            return self._output
        try:
            return self.dataset.to_arrow_refs()
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to materialize Arrow ObjectRefs.") from exc

    @property
    def output(self) -> Any:
        """Backend-selected output object."""
        return self._output if self._output is not None else self.dataset


class RayBackend:
    """Ray Data execution backend for in-memory Data Designer jobs.

    The backend maps Data Designer generation over Ray Data blocks and returns
    Ray-resident outputs. Ray is imported lazily so base Data Designer installs
    do not require the optional dependency. input_dataset may be a Ray Dataset
    or a sequence of ObjectRefs containing Arrow tables or pandas DataFrames.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        output: RayOutputMode = "dataset",
        object_ref_format: RayObjectRefInputFormat = "arrow",
        auto_init: bool = False,
        zero_copy_batch: bool = True,
        ray_remote_args: dict[str, Any] | None = None,
        order_column: str | None = None,
        drop_order_column: bool = False,
        preserve_order: bool = False,
        allow_unsafe_processors: bool = False,
    ) -> None:
        if output not in ("dataset", "arrow_refs"):
            raise ValueError("RayBackend output must be 'dataset' or 'arrow_refs'.")
        if object_ref_format not in ("arrow", "pandas"):
            raise ValueError("RayBackend object_ref_format must be 'arrow' or 'pandas'.")
        if order_column is not None and order_column == "":
            raise ValueError("RayBackend order_column must be a non-empty string when provided.")
        self.batch_size = batch_size
        self.output = output
        self.object_ref_format = object_ref_format
        self.auto_init = auto_init
        self.zero_copy_batch = zero_copy_batch
        self.ray_remote_args = ray_remote_args
        self.order_column = order_column
        self.drop_order_column = drop_order_column
        self.preserve_order = preserve_order
        self.allow_unsafe_processors = allow_unsafe_processors

    def create(
        self,
        *,
        data_designer: Any,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> RayDatasetCreationResults:
        del dataset_name
        start_time = time.perf_counter()
        ray = _import_ray()
        if not ray.is_initialized():
            if not self.auto_init:
                raise RayBackendConfigurationError(
                    "Ray is not initialized. Call ray.init(...) before using RayBackend, "
                    "or construct RayBackend(auto_init=True)."
                )
            ray.init()

        use_input_dataset = input_dataset is not None
        if use_input_dataset and config_builder.get_seed_config() is not None:
            raise RayBackendConfigurationError(
                "RayBackend input_dataset is used as the seed dataset; remove the existing seed config."
            )
        if not use_input_dataset and config_builder.get_seed_config() is not None:
            raise RayBackendConfigurationError(
                "RayBackend does not yet support seed configs without input_dataset because partition offsets "
                "are not available. Pass seed data as input_dataset, or use the local backend until "
                "RayBackend partition-offset support exists."
            )
        if self.preserve_order and self.order_column is None:
            raise RayBackendConfigurationError(
                "RayBackend preserve_order=True requires order_column in this experimental backend. "
                "Automatic hidden row-id injection is tracked as follow-up hardening work."
            )
        if not self.allow_unsafe_processors:
            _validate_ray_safe_processors(config_builder)

        dataset = self._resolve_input_dataset(ray, input_dataset=input_dataset, num_records=num_records)
        input_blocks = _get_num_blocks(dataset)
        metrics_collector = _create_metrics_collector(ray)
        batch_size = self.batch_size or data_designer._run_config.buffer_size
        worker_options = _RayWorkerOptions(
            model_providers=list(data_designer._model_providers),
            default_provider_name=data_designer._model_provider_registry.get_default_provider_name(),
            secret_resolver=data_designer._secret_resolver,
            seed_readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values()),
            managed_assets_path=str(data_designer._managed_assets_path),
            person_reader=data_designer._person_reader,
            mcp_providers=list(data_designer._mcp_providers),
            run_config=data_designer._run_config,
        )
        execution_payload = _compile_ray_execution_payload(
            config_builder=config_builder,
            worker_options=worker_options,
            use_input_dataset=use_input_dataset,
        )

        map_batches_kwargs: dict[str, Any] = {
            "fn_constructor_kwargs": {
                "execution_payload": execution_payload,
                "metrics_collector": metrics_collector,
            },
            "batch_size": batch_size,
            "batch_format": "pandas",
            "zero_copy_batch": self.zero_copy_batch,
        }
        if self.ray_remote_args is not None:
            map_batches_kwargs.update(self.ray_remote_args)

        try:
            mapped = dataset.map_batches(_RayBatchWorker, **map_batches_kwargs)
            mapped = self._apply_ordering(mapped)
            output = mapped.to_arrow_refs() if self.output == "arrow_refs" else None
            result_dataset = _dataset_from_arrow_refs(ray, output) if output is not None else mapped
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed while constructing the Ray execution plan.") from exc

        output_blocks = len(output) if output is not None else _get_num_blocks(mapped)
        metrics = RayDatasetMetrics(
            total_rows=num_records if input_dataset is None else 0,
            blocks=output_blocks or input_blocks or 0,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        return RayDatasetCreationResults(
            dataset=result_dataset,
            config_builder=config_builder,
            metrics=metrics,
            ray=ray,
            metrics_collector=metrics_collector,
            output=output,
        )

    def _resolve_input_dataset(self, ray: Any, *, input_dataset: Any | None, num_records: int) -> Any:
        if input_dataset is None:
            return ray.data.range(num_records)
        if hasattr(input_dataset, "map_batches"):
            return input_dataset
        if isinstance(input_dataset, (list, tuple)):
            refs = list(input_dataset)
            if self.object_ref_format == "pandas":
                return ray.data.from_pandas_refs(refs)
            return ray.data.from_arrow_refs(refs)
        raise TypeError(
            "RayBackend input_dataset must be a ray.data.Dataset or a sequence of Ray ObjectRefs "
            "containing PyArrow tables or pandas DataFrames."
        )

    def _apply_ordering(self, dataset: Any) -> Any:
        if self.order_column is None:
            return dataset
        ordered = dataset.sort(self.order_column)
        if self.drop_order_column:
            return ordered.drop_columns([self.order_column])
        return ordered


def _validate_ray_safe_processors(config_builder: DataDesignerConfigBuilder) -> None:
    unsafe_processors = [
        processor
        for processor in config_builder.get_processor_configs()
        if processor.processor_type != ProcessorType.DROP_COLUMNS
    ]
    if not unsafe_processors:
        return
    processor_names = ", ".join(f"{processor.name} ({processor.processor_type})" for processor in unsafe_processors)
    raise RayBackendConfigurationError(
        "RayBackend currently supports only distributed-safe processors. "
        f"Unsupported processor(s): {processor_names}. "
        "Pass allow_unsafe_processors=True to bypass this experimental guard."
    )


def _get_num_blocks(dataset: Any) -> int | None:
    num_blocks = getattr(dataset, "num_blocks", None)
    if not callable(num_blocks):
        return None
    try:
        value = num_blocks()
    except Exception:
        return None
    return int(value) if value is not None else None


def _dataset_from_arrow_refs(ray: Any, refs: list[Any]) -> Any:
    from_arrow_refs = getattr(ray.data, "from_arrow_refs", None)
    if not callable(from_arrow_refs):
        raise RayDatasetGenerationError(
            "RayBackend output='arrow_refs' requires ray.data.from_arrow_refs to expose a dataset "
            "backed by the materialized Arrow ObjectRefs."
        )
    return from_arrow_refs(refs)


def _compile_ray_execution_payload(
    *,
    config_builder: DataDesignerConfigBuilder,
    worker_options: _RayWorkerOptions,
    use_input_dataset: bool,
) -> _RayExecutionPayload:
    config_json = config_builder.build().to_json(indent=None)
    if config_json is None:
        raise RayBackendConfigurationError("RayBackend failed to serialize the Data Designer configuration.")
    return _RayExecutionPayload(
        config_json=config_json,
        worker_options=worker_options,
        use_input_dataset=use_input_dataset,
    )


class _RayMetricsCollector:
    def __init__(self) -> None:
        self._payloads: list[dict[str, Any]] = []

    def record(self, payload: dict[str, Any]) -> None:
        self._payloads.append(payload)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._payloads)


def _create_metrics_collector(ray: Any) -> Any | None:
    remote = getattr(ray, "remote", None)
    if not callable(remote):
        return None
    return remote(_RayMetricsCollector).remote()


def _merge_driver_and_worker_metrics(
    driver_metrics: RayDatasetMetrics, worker_metrics: RayDatasetMetrics
) -> RayDatasetMetrics:
    return RayDatasetMetrics(
        total_rows=worker_metrics.total_rows,
        blocks=worker_metrics.blocks,
        failed_blocks=worker_metrics.failed_blocks,
        elapsed_seconds=worker_metrics.elapsed_seconds,
        model_usage=worker_metrics.model_usage or driver_metrics.model_usage,
    )


def _clone_seed_readers_for_worker(readers: Iterable[SeedReader]) -> list[SeedReader]:
    return [_clone_seed_reader_for_worker(reader) for reader in readers]


def _clone_seed_reader_for_worker(reader: SeedReader) -> SeedReader:
    clone = copy.copy(reader)
    clone._reset_attachment_state()
    for attr in ("source", "secret_resolver"):
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone


class _InMemoryPreviewArtifactStorage(ArtifactStorage):
    """ArtifactStorage implementation for Ray preview batches.

    Ray workers use ``DatasetBuilder.build_preview()``, which returns the main
    dataset in memory. Processor preview artifacts may still be written through
    the storage interface, so this class keeps those frames in memory and avoids
    per-block temporary filesystem setup.
    """

    _frames: dict[tuple[str, str, str], Any] = PrivateAttr(default_factory=dict)
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, *, dataset_name: str = "ray-block") -> None:
        super().__init__(artifact_path=Path.cwd(), dataset_name=dataset_name)
        self.set_media_storage_mode(StorageMode.DATAFRAME)

    def clear(self) -> None:
        self._frames.clear()
        self._metadata.clear()

    def load_dataset(self, batch_stage: BatchStage = BatchStage.FINAL_RESULT) -> Any:
        frames = [self._copy_dataframe(frame) for _, frame in self._iter_stage_frames(batch_stage)]
        if not frames:
            return lazy.pd.DataFrame()
        return lazy.pd.concat(frames, ignore_index=True)

    def load_processor_dataset(self, processor_name: str) -> Any:
        frames = [
            self._copy_dataframe(frame)
            for (stage, subfolder, parquet_file_name), frame in self._frames.items()
            if stage == BatchStage.PROCESSORS_OUTPUTS.value
            and (subfolder == processor_name or (not subfolder and Path(parquet_file_name).stem == processor_name))
        ]
        if not frames:
            return lazy.pd.DataFrame()
        return lazy.pd.concat(frames, ignore_index=True)

    def list_processor_names(self) -> list[str]:
        names = {
            subfolder or Path(parquet_file_name).stem
            for (stage, subfolder, parquet_file_name) in self._frames
            if stage == BatchStage.PROCESSORS_OUTPUTS.value
        }
        return sorted(names)

    def load_dataset_with_dropped_columns(self) -> Any:
        dataset = self.load_dataset()
        dropped = self.load_dataset(BatchStage.DROPPED_COLUMNS)
        if len(dropped) == 0:
            return dataset
        return lazy.pd.concat([dataset, dropped], axis=1)

    def move_partial_result_to_final_file_path(self, batch_number: int) -> Path:
        partial_file_name = self.create_batch_file_path(batch_number, BatchStage.PARTIAL_RESULT).name
        partial_key = (BatchStage.PARTIAL_RESULT.value, "", partial_file_name)
        final_key = (BatchStage.FINAL_RESULT.value, "", partial_file_name)
        frame = self._frames.pop(partial_key)
        self._frames[final_key] = frame
        return self.create_batch_file_path(batch_number, BatchStage.FINAL_RESULT)

    def write_batch_to_parquet_file(
        self,
        batch_number: int,
        dataframe: Any,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        file_path = self.create_batch_file_path(batch_number, batch_stage)
        self.write_parquet_file(file_path.name, dataframe, batch_stage, subfolder=subfolder)
        return file_path

    def write_parquet_file(
        self,
        parquet_file_name: str,
        dataframe: Any,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        subfolder = subfolder or ""
        stage = self._stage_value(batch_stage)
        self._frames[(stage, subfolder, parquet_file_name)] = self._copy_dataframe(dataframe)
        stage_path = self._get_stage_path(batch_stage)
        return stage_path / subfolder / parquet_file_name if subfolder else stage_path / parquet_file_name

    def get_parquet_file_paths(self) -> list[str]:
        return [
            str(Path(self.final_dataset_folder_name) / parquet_file_name)
            for (stage, _, parquet_file_name) in sorted(self._frames)
            if stage == BatchStage.FINAL_RESULT.value
        ]

    def get_processor_file_paths(self) -> dict[str, list[str]]:
        processor_files: dict[str, list[str]] = {}
        for stage, subfolder, parquet_file_name in sorted(self._frames):
            if stage != BatchStage.PROCESSORS_OUTPUTS.value:
                continue
            processor_name = subfolder or Path(parquet_file_name).stem
            path_parts = [self.processors_outputs_folder_name]
            if subfolder:
                path_parts.append(subfolder)
            path_parts.append(parquet_file_name)
            processor_files.setdefault(processor_name, []).append(str(Path(*path_parts)))
        return processor_files

    def get_file_paths(self) -> dict[str, list[str] | dict[str, list[str]]]:
        file_paths: dict[str, list[str] | dict[str, list[str]]] = {
            "parquet-files": self.get_parquet_file_paths(),
        }
        processor_file_paths = self.get_processor_file_paths()
        if processor_file_paths:
            file_paths["processor-files"] = processor_file_paths
        return file_paths

    def read_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def write_metadata(self, metadata: dict[str, Any]) -> Path:
        self._metadata = dict(metadata)
        return self.metadata_file_path

    def update_metadata(self, updates: dict[str, Any]) -> Path:
        self._metadata.update(updates)
        return self.metadata_file_path

    def _iter_stage_frames(self, batch_stage: BatchStage) -> list[tuple[tuple[str, str, str], Any]]:
        stage = self._stage_value(batch_stage)
        return sorted((key, frame) for key, frame in self._frames.items() if key[0] == stage)

    def _stage_value(self, batch_stage: BatchStage) -> str:
        return BatchStage(batch_stage).value

    def _copy_dataframe(self, dataframe: Any) -> Any:
        copy_method = getattr(dataframe, "copy", None)
        if not callable(copy_method):
            return dataframe
        try:
            return copy_method(deep=True)
        except TypeError:
            return copy_method()


class _RayBatchWorker:
    def __init__(
        self,
        *,
        execution_payload: _RayExecutionPayload,
        metrics_collector: Any | None = None,
    ) -> None:
        os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
        self._execution_payload: _RayExecutionPayload = execution_payload
        self._metrics_collector: Any | None = metrics_collector
        self._base_config: DataDesignerConfig = DataDesignerConfig.model_validate_json(execution_payload.config_json)
        self._artifact_storage: _InMemoryPreviewArtifactStorage = _InMemoryPreviewArtifactStorage()
        worker_options = execution_payload.worker_options
        self._seed_reader_registry: SeedReaderRegistry = SeedReaderRegistry(
            readers=copy.deepcopy(worker_options.seed_readers)
        )
        self._resource_provider: Any = create_resource_provider(
            artifact_storage=self._artifact_storage,
            model_configs=self._base_config.model_configs or [],
            secret_resolver=worker_options.secret_resolver,
            model_provider_registry=resolve_model_provider_registry(
                worker_options.model_providers,
                worker_options.default_provider_name,
            ),
            seed_reader_registry=self._seed_reader_registry,
            person_reader=worker_options.person_reader or create_person_reader(worker_options.managed_assets_path),
            seed_dataset_source=None,
            run_config=copy.deepcopy(worker_options.run_config),
            mcp_providers=worker_options.mcp_providers,
            tool_configs=self._base_config.tool_configs or [],
        )

    def __call__(self, batch: Any) -> Any:
        return _generate_batch(batch, worker=self, metrics_collector=self._metrics_collector)

    def close(self) -> None:
        self._release_block_state()

    def generate_batch(self, batch: Any) -> Any:
        start_time = time.perf_counter()
        dataframe = _coerce_pandas_dataframe(batch)
        num_records = len(dataframe)
        if num_records == 0:
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(total_rows=0, blocks=1, elapsed_seconds=time.perf_counter() - start_time),
            )
            return dataframe

        try:
            block_config = self._create_block_config(dataframe)
            self._attach_seed_reader(block_config)
            model_usage_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
            builder = DatasetBuilder(
                data_designer_config=block_config,
                resource_provider=self._resource_provider,
                use_async=True,
            )
            raw_dataset = builder.build_preview(num_records=num_records)
            output = builder.process_preview(raw_dataset)
            elapsed_seconds = time.perf_counter() - start_time
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(
                    total_rows=len(output),
                    blocks=1,
                    elapsed_seconds=elapsed_seconds,
                    model_usage=self._get_model_usage_delta(model_usage_snapshot, elapsed_seconds),
                ),
            )
            return output
        except Exception:
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(
                    total_rows=0,
                    blocks=1,
                    failed_blocks=1,
                    elapsed_seconds=time.perf_counter() - start_time,
                ),
            )
            raise
        finally:
            self._release_block_state()

    def _create_block_config(self, dataframe: Any) -> DataDesignerConfig:
        block_config = DataDesignerConfig.model_validate_json(self._execution_payload.config_json)
        if self._execution_payload.use_input_dataset:
            block_config.seed_config = SeedConfig(source=DataFrameSeedSource(df=dataframe.copy()))
        return block_config

    def _attach_seed_reader(self, block_config: DataDesignerConfig) -> None:
        if block_config.seed_config is None:
            self._resource_provider.seed_reader = None
            return
        self._resource_provider.seed_reader = self._seed_reader_registry.get_reader(
            block_config.seed_config.source,
            self._execution_payload.worker_options.secret_resolver,
        )

    def _get_model_usage_delta(
        self,
        model_usage_snapshot: Any,
        elapsed_seconds: float,
    ) -> dict[str, dict[str, Any]] | None:
        usage_deltas = self._resource_provider.model_registry.get_usage_deltas(model_usage_snapshot)
        if not usage_deltas:
            return None
        return {
            model_name: usage_stats.get_usage_stats(total_time_elapsed=elapsed_seconds)
            for model_name, usage_stats in usage_deltas.items()
        }

    def _release_block_state(self) -> None:
        self._resource_provider.seed_reader = None
        self._artifact_storage.clear()
        _detach_seed_readers(self._seed_reader_registry._readers.values())


def _detach_seed_readers(readers: Iterable[SeedReader]) -> None:
    for reader in readers:
        reader._reset_attachment_state()
        for attr in ("source", "secret_resolver"):
            if hasattr(reader, attr):
                delattr(reader, attr)


def _generate_batch(
    batch: Any,
    *,
    worker: _RayBatchWorker | None = None,
    config_builder: DataDesignerConfigBuilder | None = None,
    worker_options: _RayWorkerOptions | None = None,
    use_input_dataset: bool | None = None,
    metrics_collector: Any | None = None,
) -> Any:
    if worker is not None:
        return worker.generate_batch(batch)
    if config_builder is None or worker_options is None or use_input_dataset is None:
        raise TypeError("_generate_batch requires a worker or legacy config_builder/worker_options arguments.")
    execution_payload = _compile_ray_execution_payload(
        config_builder=config_builder,
        worker_options=worker_options,
        use_input_dataset=use_input_dataset,
    )
    return _RayBatchWorker(execution_payload=execution_payload, metrics_collector=metrics_collector)(batch)


def _record_worker_metrics(metrics_collector: Any | None, metrics: RayWorkerMetrics) -> None:
    if metrics_collector is None:
        return
    importlib.import_module("ray").get(metrics_collector.record.remote(metrics.to_dict()))


def _coerce_pandas_dataframe(batch: Any) -> Any:
    if isinstance(batch, lazy.pd.DataFrame):
        return batch
    if isinstance(batch, dict):
        return lazy.pd.DataFrame(batch)
    return lazy.pd.DataFrame(batch)


def _import_ray() -> Any:
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise ImportError("RayBackend requires Ray. Install Data Designer with `data-designer[ray]`.") from exc
