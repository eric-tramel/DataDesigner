# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy, SeedConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    FileSystemSeedReader,
    SeedReader,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.interface.backends import BackendRuntimeContext

RAY_RANGE_ID_COLUMN = "id"


@dataclass(frozen=True)
class RaySeedWindow:
    start: int
    size: int


@dataclass(frozen=True)
class RaySeedPlan:
    input_dataframe: Any | None = None
    seed_window: RaySeedWindow | None = None


def plan_seed_execution(
    *,
    runtime_context: BackendRuntimeContext,
    seed_config: SeedConfig,
    num_records: int,
) -> RaySeedPlan:
    if should_materialize_seed_on_driver(runtime_context=runtime_context, seed_config=seed_config):
        return RaySeedPlan(
            input_dataframe=materialize_seed_dataframe(
                runtime_context=runtime_context,
                seed_config=seed_config,
                num_records=num_records,
            )
        )
    return RaySeedPlan(
        seed_window=preflight_seed_window(
            runtime_context=runtime_context,
            seed_config=seed_config,
            num_records=num_records,
        )
    )


def preflight_seed_window(
    *,
    runtime_context: BackendRuntimeContext,
    seed_config: SeedConfig,
    num_records: int,
) -> RaySeedWindow:
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend seed configs without input_dataset currently support only ordered sampling. "
            "Pass the seed data as input_dataset or use the local backend for shuffled seed sampling."
        )

    try:
        seed_reader = _get_detached_seed_reader(runtime_context=runtime_context, seed_config=seed_config)
        seed_dataset_size = seed_reader.get_seed_dataset_size()
        index_range = resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=seed_dataset_size)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayBackendConfigurationError(
            "RayBackend failed to preflight the seed config for partition offsets."
        ) from exc

    if num_records > index_range.size:
        raise RayBackendConfigurationError(
            "RayBackend seed configs without input_dataset require num_records to fit inside the selected seed "
            f"range. Requested {num_records} records from {index_range.size} selected seed rows."
        )
    return RaySeedWindow(start=index_range.start, size=index_range.size)


def should_materialize_seed_on_driver(*, runtime_context: BackendRuntimeContext, seed_config: SeedConfig) -> bool:
    """Return whether Ray should treat a seed source as a materialized input dataset."""
    try:
        seed_reader = _get_detached_seed_reader(runtime_context=runtime_context, seed_config=seed_config)
    except Exception as exc:
        raise RayBackendConfigurationError("RayBackend failed to inspect the seed reader for Ray planning.") from exc
    return isinstance(seed_reader, FileSystemSeedReader)


def materialize_seed_dataframe(
    *,
    runtime_context: BackendRuntimeContext,
    seed_config: SeedConfig,
    num_records: int,
) -> Any:
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend driver-materialized filesystem seed configs currently support only ordered sampling. "
            "Pass the seed data as input_dataset or use the local backend for shuffled seed sampling."
        )

    try:
        seed_reader = _get_detached_seed_reader(runtime_context=runtime_context, seed_config=seed_config)
        seed_dataset_size = seed_reader.get_seed_dataset_size()
        index_range = resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=seed_dataset_size)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayBackendConfigurationError("RayBackend failed to materialize the seed config for Ray input.") from exc

    batch_reader = seed_reader.create_batch_reader(batch_size=num_records, index_range=index_range, shuffle=False)
    output = lazy.pd.DataFrame()
    empty_batch_reads = 0
    while len(output) < num_records:
        try:
            seed_batch = batch_reader.read_next_batch().to_pandas().reset_index(drop=True)
        except StopIteration:
            batch_reader = seed_reader.create_batch_reader(
                batch_size=num_records,
                index_range=index_range,
                shuffle=False,
            )
            continue

        if len(seed_batch) == 0:
            empty_batch_reads += 1
            if empty_batch_reads > max(num_records * 2, 1):
                raise RayBackendConfigurationError(
                    "RayBackend filesystem seed materialization did not produce any rows after repeated reads."
                )
            continue
        output = lazy.pd.concat([output, seed_batch], ignore_index=True)

    return output.iloc[:num_records].reset_index(drop=True)


def resolve_seed_config_index_range(*, seed_config: SeedConfig, seed_dataset_size: int) -> IndexRange:
    if seed_dataset_size <= 0:
        raise RayBackendConfigurationError("RayBackend seed config resolved to an empty seed dataset.")
    if seed_config.selection_strategy is None:
        return IndexRange(start=0, end=seed_dataset_size - 1)
    if isinstance(seed_config.selection_strategy, IndexRange):
        if seed_config.selection_strategy.end >= seed_dataset_size:
            raise RayBackendConfigurationError(
                "RayBackend seed config selection_strategy is out of bounds for the seed dataset. "
                f"Selection end={seed_config.selection_strategy.end}, seed rows={seed_dataset_size}."
            )
        return seed_config.selection_strategy
    if isinstance(seed_config.selection_strategy, PartitionBlock):
        if seed_config.selection_strategy.num_partitions > seed_dataset_size:
            raise RayBackendConfigurationError(
                "RayBackend seed config selection_strategy is out of bounds for the seed dataset. "
                f"Partition count={seed_config.selection_strategy.num_partitions}, seed rows={seed_dataset_size}."
            )
        return seed_config.selection_strategy.to_index_range(seed_dataset_size)
    raise RayBackendConfigurationError(
        "RayBackend seed config uses an unsupported selection_strategy for partitioned execution."
    )


def input_dataset_seed_readers() -> list[SeedReader]:
    return [DataFrameSeedReader()]


def clone_seed_readers_for_worker(readers: Iterable[SeedReader]) -> list[SeedReader]:
    clones = [_clone_seed_reader_for_worker(reader) for reader in readers]
    return ensure_dataframe_seed_reader(clones)


def ensure_dataframe_seed_reader(readers: Sequence[SeedReader]) -> list[SeedReader]:
    seed_readers = list(readers)
    seed_types = {reader.get_seed_type() for reader in seed_readers}
    dataframe_reader = DataFrameSeedReader()
    if dataframe_reader.get_seed_type() not in seed_types:
        seed_readers.append(dataframe_reader)
    return seed_readers


def create_seed_config_for_window(
    *,
    seed_config: SeedConfig,
    seed_readers: Sequence[SeedReader],
    secret_resolver: SecretResolver,
    dataframe: Any,
    seed_window: RaySeedWindow,
    range_id_column: str = RAY_RANGE_ID_COLUMN,
) -> SeedConfig:
    return SeedConfig(
        source=DataFrameSeedSource(
            df=read_seed_window_dataframe(
                seed_config=seed_config,
                seed_readers=seed_readers,
                secret_resolver=secret_resolver,
                dataframe=dataframe,
                seed_window=seed_window,
                range_id_column=range_id_column,
            )
        )
    )


def read_seed_window_dataframe(
    *,
    seed_config: SeedConfig,
    seed_readers: Sequence[SeedReader],
    secret_resolver: SecretResolver,
    dataframe: Any,
    seed_window: RaySeedWindow,
    range_id_column: str = RAY_RANGE_ID_COLUMN,
) -> Any:
    row_offsets = get_contiguous_range_offsets(dataframe, range_id_column=range_id_column)
    index_range = IndexRange(
        start=seed_window.start + row_offsets[0],
        end=seed_window.start + row_offsets[-1],
    )
    if row_offsets[-1] >= seed_window.size:
        raise RayDatasetGenerationError(
            "RayBackend seed partition offset exceeded the selected seed range. "
            f"Offset={row_offsets[-1]}, selected seed rows={seed_window.size}."
        )

    seed_reader = _create_seed_reader(
        seed_readers=copy.deepcopy(seed_readers),
        seed_config=seed_config,
        secret_resolver=secret_resolver,
    )
    batch_reader = seed_reader.create_batch_reader(
        batch_size=len(row_offsets),
        index_range=index_range,
        shuffle=False,
    )
    seed_dataframe = batch_reader.read_next_batch().to_pandas().reset_index(drop=True)
    if len(seed_dataframe) != len(row_offsets):
        raise RayDatasetGenerationError(
            "RayBackend seed reader returned an unexpected row count for a partition offset. "
            f"Expected {len(row_offsets)} rows, got {len(seed_dataframe)}."
        )
    return seed_dataframe


def get_contiguous_range_offsets(dataframe: Any, *, range_id_column: str = RAY_RANGE_ID_COLUMN) -> list[int]:
    if range_id_column not in dataframe.columns:
        raise RayDatasetGenerationError(
            f"RayBackend seed partition offsets require {range_id_column!r} from ray.data.range()."
        )
    row_offsets = [int(value) for value in dataframe[range_id_column].tolist()]
    if not row_offsets:
        raise RayDatasetGenerationError("RayBackend seed partition offsets received an empty Ray worker batch.")
    expected_offsets = list(range(row_offsets[0], row_offsets[0] + len(row_offsets)))
    if row_offsets != expected_offsets:
        raise RayDatasetGenerationError(
            "RayBackend seed partition offsets require contiguous ray.data.range() batches. "
            f"Received offsets {row_offsets!r}."
        )
    return row_offsets


def _get_detached_seed_reader(*, runtime_context: BackendRuntimeContext, seed_config: SeedConfig) -> SeedReader:
    return _create_seed_reader(
        seed_readers=clone_seed_readers_for_worker(runtime_context.seed_readers),
        seed_config=seed_config,
        secret_resolver=runtime_context.secret_resolver,
        runtime_context=runtime_context,
    )


def _create_seed_reader(
    *,
    seed_readers: Sequence[SeedReader],
    seed_config: SeedConfig,
    secret_resolver: SecretResolver,
    runtime_context: BackendRuntimeContext | None = None,
) -> SeedReader:
    if runtime_context is not None:
        registry = runtime_context.create_seed_reader_registry(seed_readers=seed_readers)
    else:
        registry = SeedReaderRegistry(readers=seed_readers)
    return registry.get_reader(seed_config.source, secret_resolver)


def _clone_seed_reader_for_worker(reader: SeedReader) -> SeedReader:
    clone = copy.copy(reader)
    # Compatibility shim: SeedReader currently has no public detached/snapshot
    # API, but Ray workers must not inherit driver-local attachment state such as
    # DuckDB connections or attached sources. Replace this with the public engine
    # lifecycle API when one exists.
    clone._reset_attachment_state()
    for attr in ("source", "secret_resolver"):
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone
