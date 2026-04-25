# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import SeedConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.dataset_builders.dataset_builder import DatasetBlockChunk, DatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.models.usage import ModelUsageStats
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import ResourceProvider, create_resource_provider
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider
    from data_designer.engine.dataset_builders.utils.task_model import TaskTrace
    from data_designer.engine.models.clients.throttle_manager import ThrottleManagerLike


class BlockRuntimeContext(Protocol):
    """Protocol for backend runtime snapshots accepted by block execution."""

    model_providers: list[ModelProvider]
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: list[SeedReader]
    managed_assets_path: str | Path
    person_reader: PersonReader | None
    mcp_providers: list[MCPProviderT]
    run_config: RunConfig
    throttle_manager: ThrottleManagerLike | None


@dataclass(frozen=True)
class DataDesignerBlockRuntimeContext:
    """Backend-neutral resources needed to construct an engine block runtime."""

    model_providers: list[ModelProvider]
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: list[SeedReader]
    managed_assets_path: str | Path
    person_reader: PersonReader | None = None
    mcp_providers: list[MCPProviderT] = field(default_factory=list)
    run_config: RunConfig = field(default_factory=RunConfig)
    throttle_manager: ThrottleManagerLike | None = None


@dataclass(frozen=True)
class BlockExecutionOptions:
    """Execution options for one backend block."""

    use_async: bool = True
    dataset_name: str = "dataset-block"
    current_batch_number: int | None = None
    capture_stream_artifacts: bool = False


@dataclass(frozen=True)
class BlockExecutionResult:
    """Result of executing one dataset block."""

    dataframe: pd.DataFrame
    raw_dataframe: pd.DataFrame
    task_traces: list[TaskTrace]
    model_usage_stats: dict[str, dict[str, Any]]
    model_usage_deltas: dict[str, ModelUsageStats]
    processor_artifacts: dict[str, pd.DataFrame]
    input_rows: int
    output_rows: int
    dropped_rows: int
    all_rows_dropped: bool
    partial_rows_dropped: bool
    artifact_storage: ArtifactStorage | None = None


@dataclass(frozen=True)
class BlockExecutionChunk:
    """One generated chunk from a backend block stream."""

    dataframe: pd.DataFrame
    raw_dataframe: pd.DataFrame
    row_group: int
    input_start: int
    input_rows: int
    output_rows: int
    processor_artifacts: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockExecutionStreamSummary:
    """Summary emitted after a block chunk stream is exhausted."""

    task_traces: list[TaskTrace]
    model_usage_stats: dict[str, dict[str, Any]]
    model_usage_deltas: dict[str, ModelUsageStats]
    processor_artifacts: dict[str, pd.DataFrame]
    input_rows: int
    output_rows: int
    dropped_rows: int
    all_rows_dropped: bool
    partial_rows_dropped: bool
    artifact_storage: ArtifactStorage | None = None


class BlockExecutionChunkStream(Iterator[BlockExecutionChunk]):
    """Iterator returned by ``execute_dataset_block_stream``.

    The final summary is available through ``summary`` after the iterator is
    exhausted. This keeps chunk delivery streaming while preserving the metrics
    and observability fields backends need once generation completes.
    """

    def __init__(
        self,
        iterator: Iterator[BlockExecutionChunk],
        summary_holder: dict[str, BlockExecutionStreamSummary],
    ) -> None:
        self._iterator = iterator
        self._summary_holder = summary_holder

    def __iter__(self) -> BlockExecutionChunkStream:
        return self

    def __next__(self) -> BlockExecutionChunk:
        return next(self._iterator)

    @property
    def summary(self) -> BlockExecutionStreamSummary:
        try:
            return self._summary_holder["summary"]
        except KeyError as exc:
            raise RuntimeError("Block execution stream summary is available only after exhaustion.") from exc


def execute_dataset_block(
    *,
    config_builder: DataDesignerConfigBuilder | None = None,
    data_designer_config: DataDesignerConfig | None = None,
    runtime_context: BlockRuntimeContext | None = None,
    resource_provider: ResourceProvider | None = None,
    input_frame: pd.DataFrame | None = None,
    num_records: int | None = None,
    options: BlockExecutionOptions | None = None,
    artifact_storage: ArtifactStorage | None = None,
) -> BlockExecutionResult:
    """Execute one in-memory dataset block for a backend.

    The API owns block-local seed injection, resource-provider construction, generator
    execution, post-processing, task traces, and model usage snapshots. Backends may
    pass either a reusable runtime context or a fully constructed resource provider.
    """
    block_options = options or BlockExecutionOptions()
    block_builder, block_config = _resolve_block_config(
        config_builder=config_builder,
        data_designer_config=data_designer_config,
        input_frame=input_frame,
    )
    block_num_records = _resolve_num_records(input_frame=input_frame, num_records=num_records)

    if artifact_storage is not None:
        return _execute_with_storage(
            data_designer_config=block_config,
            config_builder=block_builder,
            runtime_context=runtime_context,
            resource_provider=resource_provider,
            artifact_storage=artifact_storage,
            num_records=block_num_records,
            options=block_options,
            return_storage=True,
        )

    with tempfile.TemporaryDirectory(prefix="data-designer-block-") as artifact_dir:
        ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
        temp_storage = ArtifactStorage(artifact_path=artifact_dir, dataset_name=block_options.dataset_name)
        return _execute_with_storage(
            data_designer_config=block_config,
            config_builder=block_builder,
            runtime_context=runtime_context,
            resource_provider=resource_provider,
            artifact_storage=temp_storage,
            num_records=block_num_records,
            options=block_options,
            return_storage=False,
        )


def execute_dataset_block_stream(
    *,
    rows_per_chunk: int,
    config_builder: DataDesignerConfigBuilder | None = None,
    data_designer_config: DataDesignerConfig | None = None,
    runtime_context: BlockRuntimeContext | None = None,
    resource_provider: ResourceProvider | None = None,
    input_frame: pd.DataFrame | None = None,
    num_records: int | None = None,
    options: BlockExecutionOptions | None = None,
    artifact_storage: ArtifactStorage | None = None,
) -> BlockExecutionChunkStream:
    """Execute one backend block and stream row-group chunks as they complete."""
    summary_holder: dict[str, BlockExecutionStreamSummary] = {}

    def _iterator() -> Iterator[BlockExecutionChunk]:
        block_options = options or BlockExecutionOptions()
        block_builder, block_config = _resolve_block_config(
            config_builder=config_builder,
            data_designer_config=data_designer_config,
            input_frame=input_frame,
        )
        block_num_records = _resolve_num_records(input_frame=input_frame, num_records=num_records)

        if artifact_storage is not None:
            yield from _execute_stream_with_storage(
                data_designer_config=block_config,
                config_builder=block_builder,
                runtime_context=runtime_context,
                resource_provider=resource_provider,
                artifact_storage=artifact_storage,
                num_records=block_num_records,
                rows_per_chunk=rows_per_chunk,
                options=block_options,
                return_storage=True,
                summary_holder=summary_holder,
            )
            return

        with tempfile.TemporaryDirectory(prefix="data-designer-block-") as artifact_dir:
            ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
            temp_storage = ArtifactStorage(artifact_path=artifact_dir, dataset_name=block_options.dataset_name)
            yield from _execute_stream_with_storage(
                data_designer_config=block_config,
                config_builder=block_builder,
                runtime_context=runtime_context,
                resource_provider=resource_provider,
                artifact_storage=temp_storage,
                num_records=block_num_records,
                rows_per_chunk=rows_per_chunk,
                options=block_options,
                return_storage=False,
                summary_holder=summary_holder,
            )

    return BlockExecutionChunkStream(_iterator(), summary_holder)


def _resolve_block_config(
    *,
    config_builder: DataDesignerConfigBuilder | None,
    data_designer_config: DataDesignerConfig | None,
    input_frame: pd.DataFrame | None,
) -> tuple[DataDesignerConfigBuilder | None, DataDesignerConfig]:
    if (config_builder is None) == (data_designer_config is None):
        raise ValueError("Provide exactly one of config_builder or data_designer_config.")

    if config_builder is not None:
        block_builder = copy.deepcopy(config_builder)
        if input_frame is not None:
            block_builder.with_seed_dataset(DataFrameSeedSource(df=input_frame.copy()))
        return block_builder, block_builder.build()

    block_config = copy.deepcopy(data_designer_config)
    if input_frame is not None:
        block_config.seed_config = SeedConfig(source=DataFrameSeedSource(df=input_frame.copy()))
    return None, block_config


def _resolve_num_records(*, input_frame: pd.DataFrame | None, num_records: int | None) -> int:
    if input_frame is not None:
        return len(input_frame)
    if num_records is None:
        raise ValueError("num_records is required when input_frame is not provided.")
    return num_records


def _execute_with_storage(
    *,
    data_designer_config: DataDesignerConfig,
    config_builder: DataDesignerConfigBuilder | None,
    runtime_context: BlockRuntimeContext | None,
    resource_provider: ResourceProvider | None,
    artifact_storage: ArtifactStorage,
    num_records: int,
    options: BlockExecutionOptions,
    return_storage: bool,
) -> BlockExecutionResult:
    if (runtime_context is None) == (resource_provider is None):
        raise ValueError("Provide exactly one of runtime_context or resource_provider.")

    start_time = time.perf_counter()
    try:
        with _async_engine_environment(options.use_async):
            block_resource_provider = resource_provider or _create_resource_provider(
                config_builder=config_builder,
                data_designer_config=data_designer_config,
                runtime_context=runtime_context,
                artifact_storage=artifact_storage,
            )
            usage_snapshot = block_resource_provider.model_registry.get_model_usage_snapshot()
            builder = DatasetBuilder(
                data_designer_config=data_designer_config,
                resource_provider=block_resource_provider,
                use_async=options.use_async,
            )
            raw_dataframe, dataframe = builder.build_block(
                num_records=num_records,
                current_batch_number=options.current_batch_number,
            )
            elapsed = time.perf_counter() - start_time
            model_usage_stats = block_resource_provider.model_registry.get_model_usage_stats(elapsed)
            model_usage_deltas = block_resource_provider.model_registry.get_usage_deltas(usage_snapshot)
            processor_artifacts = _load_processor_artifacts(artifact_storage)
            output_rows = len(dataframe)
            dropped_rows = max(num_records - output_rows, 0)
            return BlockExecutionResult(
                dataframe=dataframe,
                raw_dataframe=raw_dataframe,
                task_traces=builder.task_traces,
                model_usage_stats=model_usage_stats,
                model_usage_deltas=model_usage_deltas,
                processor_artifacts=processor_artifacts,
                input_rows=num_records,
                output_rows=output_rows,
                dropped_rows=dropped_rows,
                all_rows_dropped=num_records > 0 and output_rows == 0,
                partial_rows_dropped=0 < output_rows < num_records,
                artifact_storage=artifact_storage if return_storage else None,
            )
    except DataDesignerError:
        raise
    except Exception as exc:
        raise DatasetGenerationError(f"🛑 Failed to execute dataset block: {exc}") from exc


def _execute_stream_with_storage(
    *,
    data_designer_config: DataDesignerConfig,
    config_builder: DataDesignerConfigBuilder | None,
    runtime_context: BlockRuntimeContext | None,
    resource_provider: ResourceProvider | None,
    artifact_storage: ArtifactStorage,
    num_records: int,
    rows_per_chunk: int,
    options: BlockExecutionOptions,
    return_storage: bool,
    summary_holder: dict[str, BlockExecutionStreamSummary],
) -> Iterator[BlockExecutionChunk]:
    if (runtime_context is None) == (resource_provider is None):
        raise ValueError("Provide exactly one of runtime_context or resource_provider.")

    start_time = time.perf_counter()
    output_rows = 0
    try:
        with _async_engine_environment(options.use_async):
            block_resource_provider = resource_provider or _create_resource_provider(
                config_builder=config_builder,
                data_designer_config=data_designer_config,
                runtime_context=runtime_context,
                artifact_storage=artifact_storage,
            )
            usage_snapshot = block_resource_provider.model_registry.get_model_usage_snapshot()
            builder = DatasetBuilder(
                data_designer_config=data_designer_config,
                resource_provider=block_resource_provider,
                use_async=options.use_async,
            )
            for chunk in builder.build_block_chunks(
                num_records=num_records,
                rows_per_chunk=rows_per_chunk,
                current_batch_number=options.current_batch_number,
                capture_artifacts=options.capture_stream_artifacts,
            ):
                output_rows += len(chunk.dataframe)
                yield _block_execution_chunk(chunk)

            elapsed = time.perf_counter() - start_time
            model_usage_stats = block_resource_provider.model_registry.get_model_usage_stats(elapsed)
            model_usage_deltas = block_resource_provider.model_registry.get_usage_deltas(usage_snapshot)
            processor_artifacts = (
                {} if options.capture_stream_artifacts else _load_processor_artifacts(artifact_storage)
            )
            dropped_rows = max(num_records - output_rows, 0)
            summary_holder["summary"] = BlockExecutionStreamSummary(
                task_traces=builder.task_traces,
                model_usage_stats=model_usage_stats,
                model_usage_deltas=model_usage_deltas,
                processor_artifacts=processor_artifacts,
                input_rows=num_records,
                output_rows=output_rows,
                dropped_rows=dropped_rows,
                all_rows_dropped=num_records > 0 and output_rows == 0,
                partial_rows_dropped=0 < output_rows < num_records,
                artifact_storage=artifact_storage if return_storage else None,
            )
    except DataDesignerError:
        raise
    except Exception as exc:
        raise DatasetGenerationError(f"🛑 Failed to execute dataset block stream: {exc}") from exc


def _block_execution_chunk(chunk: DatasetBlockChunk) -> BlockExecutionChunk:
    return BlockExecutionChunk(
        dataframe=chunk.dataframe,
        raw_dataframe=chunk.raw_dataframe,
        row_group=chunk.row_group,
        input_start=chunk.input_start,
        input_rows=chunk.input_rows,
        output_rows=len(chunk.dataframe),
        processor_artifacts=chunk.processor_artifacts,
    )


def _create_resource_provider(
    *,
    config_builder: DataDesignerConfigBuilder | None,
    data_designer_config: DataDesignerConfig,
    runtime_context: BlockRuntimeContext | None,
    artifact_storage: ArtifactStorage,
) -> ResourceProvider:
    if runtime_context is None:
        raise ValueError("runtime_context is required when resource_provider is not provided.")

    seed_source = data_designer_config.seed_config.source if data_designer_config.seed_config is not None else None
    return create_resource_provider(
        artifact_storage=artifact_storage,
        model_configs=data_designer_config.model_configs,
        secret_resolver=runtime_context.secret_resolver,
        model_provider_registry=resolve_model_provider_registry(
            runtime_context.model_providers,
            runtime_context.default_provider_name,
        ),
        seed_reader_registry=SeedReaderRegistry(readers=copy.deepcopy(runtime_context.seed_readers)),
        person_reader=runtime_context.person_reader or create_person_reader(runtime_context.managed_assets_path),
        seed_dataset_source=seed_source,
        run_config=copy.deepcopy(runtime_context.run_config),
        mcp_providers=runtime_context.mcp_providers,
        tool_configs=(config_builder.tool_configs if config_builder is not None else data_designer_config.tool_configs),
        throttle_manager=getattr(runtime_context, "throttle_manager", None),
    )


def _load_processor_artifacts(artifact_storage: ArtifactStorage) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    for processor_name in artifact_storage.list_processor_names():
        artifacts[processor_name] = artifact_storage.load_processor_dataset(processor_name)
    return artifacts


@contextmanager
def _async_engine_environment(enabled: bool) -> Iterator[None]:
    previous = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE")
    if enabled:
        os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DATA_DESIGNER_ASYNC_ENGINE", None)
        else:
            os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = previous
