# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy, SeedConfig
from data_designer.config.seed_source import FileContentsSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    FileContentsSeedReader,
    FileSystemSeedReader,
    LocalFileSeedReader,
    SeedReader,
    SeedReaderRegistry,
    create_seed_reader_output_dataframe,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.interface.backends import BackendRuntimeContext

RAY_RANGE_ID_COLUMN = "id"
RAY_SEED_ORDINAL_COLUMN = "__data_designer_ray_seed_ordinal"


@dataclass(frozen=True)
class RaySeedWindow:
    start: int
    size: int


@dataclass(frozen=True)
class RaySeedPlan:
    input_dataframe: Any | None = None
    input_dataset: Any | None = None
    seed_window: RaySeedWindow | None = None


def plan_seed_execution(
    *,
    runtime_context: BackendRuntimeContext,
    seed_config: SeedConfig,
    num_records: int,
    ray: Any | None = None,
) -> RaySeedPlan:
    seed_reader = _get_detached_seed_reader(runtime_context=runtime_context, seed_config=seed_config)
    if ray is not None and isinstance(seed_reader, LocalFileSeedReader):
        return RaySeedPlan(
            input_dataset=create_ray_local_file_seed_dataset(
                ray=ray,
                seed_reader=seed_reader,
                seed_config=seed_config,
                num_records=num_records,
            )
        )
    if ray is not None and isinstance(seed_reader, FileContentsSeedReader):
        return RaySeedPlan(
            input_dataset=create_ray_file_contents_seed_dataset(
                ray=ray,
                seed_reader=seed_reader,
                seed_config=seed_config,
                num_records=num_records,
            )
        )
    if ray is not None and isinstance(seed_reader, FileSystemSeedReader):
        return RaySeedPlan(
            input_dataset=create_ray_filesystem_seed_dataset(
                ray=ray,
                seed_reader=seed_reader,
                seed_config=seed_config,
                num_records=num_records,
                secret_resolver=runtime_context.secret_resolver,
            )
        )

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


def create_ray_local_file_seed_dataset(
    *,
    ray: Any,
    seed_reader: LocalFileSeedReader,
    seed_config: SeedConfig,
    num_records: int,
) -> Any:
    """Create a Ray Dataset for an ordered local structured-file seed."""
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend Ray-native local-file seed ingestion currently supports only ordered sampling. "
            "Use input_dataset for shuffled Ray-native seeds or the local backend for shuffled seed sampling."
        )
    seed_dataset_size = seed_reader.get_seed_dataset_size()
    index_range = resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=seed_dataset_size)
    _validate_no_seed_ordinal_column(seed_reader.get_column_names())
    dataset = _read_local_file_seed_dataset(ray, seed_reader.get_dataset_uri())
    return _apply_ordered_seed_selection_and_cycling(
        ray=ray,
        dataset=dataset,
        seed_dataset_size=seed_dataset_size,
        index_range=index_range,
        num_records=num_records,
    )


def create_ray_file_contents_seed_dataset(
    *,
    ray: Any,
    seed_reader: FileContentsSeedReader,
    seed_config: SeedConfig,
    num_records: int,
) -> Any:
    """Create a Ray Dataset for ordered file-content seeds without driver content reads."""
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend Ray-native file-content seed ingestion currently supports only ordered sampling. "
            "Use input_dataset for shuffled Ray-native seeds or the local backend for shuffled seed sampling."
        )
    source = seed_reader.source
    if not isinstance(source, FileContentsSeedSource):
        raise RayBackendConfigurationError("RayBackend file-content seed ingestion expected FileContentsSeedSource.")
    context = seed_reader.create_filesystem_context(source.runtime_path)
    relative_paths = seed_reader.get_matching_relative_paths(
        context=context,
        file_pattern=source.file_pattern,
        recursive=source.recursive,
    )
    index_range = resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=len(relative_paths))
    absolute_paths = [str(context.root_path / relative_path) for relative_path in relative_paths]
    dataset = ray.data.read_binary_files(
        absolute_paths,
        include_paths=True,
        partitioning=None,
    ).map_batches(
        _file_contents_seed_batch_to_dataframe,
        fn_kwargs={"root_path": str(context.root_path), "encoding": source.encoding},
        batch_format="pandas",
    )
    return _apply_ordered_seed_selection_and_cycling(
        ray=ray,
        dataset=dataset,
        seed_dataset_size=len(relative_paths),
        index_range=index_range,
        num_records=num_records,
    )


def create_ray_filesystem_seed_dataset(
    *,
    ray: Any,
    seed_reader: FileSystemSeedReader[Any],
    seed_config: SeedConfig,
    num_records: int,
    secret_resolver: SecretResolver,
) -> Any:
    """Create a Ray Dataset that hydrates filesystem seed manifest rows on workers."""
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend Ray-native filesystem seed ingestion currently supports only ordered sampling. "
            "Use input_dataset for shuffled Ray-native seeds or the local backend for shuffled seed sampling."
        )
    manifest_dataframe = _filesystem_manifest_dataframe(seed_reader)
    index_range = resolve_seed_config_index_range(
        seed_config=seed_config,
        seed_dataset_size=len(manifest_dataframe),
    )
    output_columns = seed_reader.get_output_column_names()
    _validate_no_seed_ordinal_column(output_columns)
    manifest_dataset = ray.data.from_items(manifest_dataframe.to_dict(orient="records"))
    selected_manifest_dataset = _apply_ordered_seed_selection_and_cycling(
        ray=ray,
        dataset=manifest_dataset,
        seed_dataset_size=len(manifest_dataframe),
        index_range=index_range,
        num_records=num_records,
    )
    return selected_manifest_dataset.map_batches(
        _hydrate_filesystem_seed_manifest_batch,
        fn_kwargs={
            "seed_reader": seed_reader,
            "seed_source": seed_reader.source,
            "secret_resolver": secret_resolver,
            "output_columns": output_columns,
        },
        batch_format="pandas",
    ).limit(num_records)


def _filesystem_manifest_dataframe(seed_reader: FileSystemSeedReader[Any]) -> Any:
    context = seed_reader.create_filesystem_context(seed_reader.source.runtime_path)
    manifest = seed_reader.build_manifest(context=context)
    manifest_dataframe = manifest.copy() if isinstance(manifest, lazy.pd.DataFrame) else lazy.pd.DataFrame(manifest)
    if manifest_dataframe.empty:
        raise RayBackendConfigurationError(
            f"RayBackend filesystem seed source at {seed_reader.source.runtime_path} did not produce any rows."
        )
    return manifest_dataframe


def _hydrate_filesystem_seed_manifest_batch(
    batch: Any,
    *,
    seed_reader: FileSystemSeedReader[Any],
    seed_source: Any,
    secret_resolver: SecretResolver,
    output_columns: list[str],
) -> Any:
    reader = _clone_seed_reader_for_worker(seed_reader)
    reader.attach(seed_source, secret_resolver)
    context = reader.create_filesystem_context(seed_source.runtime_path)
    hydrated_records = reader._hydrate_rows(
        manifest_rows=lazy.pd.DataFrame(batch).to_dict(orient="records"),
        context=context,
    )
    return create_seed_reader_output_dataframe(records=hydrated_records, output_columns=output_columns)


def _read_local_file_seed_dataset(ray: Any, dataset_uri: str) -> Any:
    normalized_uri = dataset_uri.lower()
    read_kwargs: dict[str, Any] = {"partitioning": None}
    if _path_matches_suffix(normalized_uri, ".parquet"):
        return ray.data.read_parquet(dataset_uri, **read_kwargs)
    if _path_matches_suffix(normalized_uri, ".csv"):
        return ray.data.read_csv(dataset_uri, **read_kwargs)
    if _path_matches_suffix(normalized_uri, ".jsonl"):
        return ray.data.read_json(dataset_uri, lines=True, **read_kwargs)
    if _path_matches_suffix(normalized_uri, ".json"):
        return ray.data.read_json(dataset_uri, lines=False, **read_kwargs)
    raise RayBackendConfigurationError(
        "RayBackend Ray-native local-file seed ingestion supports parquet, csv, json, and jsonl seed files."
    )


def _path_matches_suffix(path: str, suffix: str) -> bool:
    return path.endswith((suffix, f"{suffix}'")) or f"*{suffix}" in path


def _apply_ordered_seed_selection_and_cycling(
    *,
    ray: Any,
    dataset: Any,
    seed_dataset_size: int,
    index_range: IndexRange,
    num_records: int,
) -> Any:
    if num_records < 0:
        raise RayBackendConfigurationError("RayBackend seed num_records must be non-negative.")
    selected = _select_seed_dataset_range(
        ray=ray,
        dataset=dataset,
        seed_dataset_size=seed_dataset_size,
        index_range=index_range,
    )
    return _cycle_seed_dataset(selected, selected_size=index_range.size, num_records=num_records)


def _select_seed_dataset_range(
    *,
    ray: Any,
    dataset: Any,
    seed_dataset_size: int,
    index_range: IndexRange,
) -> Any:
    if index_range.start == 0 and index_range.end == seed_dataset_size - 1:
        return dataset
    ordinal_dataset = ray.data.range(seed_dataset_size).map_batches(
        _rename_range_id_to_seed_ordinal,
        batch_format="pandas",
    )
    return (
        dataset.zip(ordinal_dataset)
        .filter(
            _row_in_seed_index_range,
            fn_kwargs={"start": index_range.start, "end": index_range.end},
        )
        .drop_columns([RAY_SEED_ORDINAL_COLUMN])
    )


def _cycle_seed_dataset(dataset: Any, *, selected_size: int, num_records: int) -> Any:
    if selected_size <= 0:
        raise RayBackendConfigurationError("RayBackend seed config resolved to an empty selected seed range.")
    if num_records <= selected_size:
        return dataset.limit(num_records)
    full_repetitions, remainder = divmod(num_records, selected_size)
    pieces = [dataset for _ in range(full_repetitions)]
    if remainder:
        pieces.append(dataset.limit(remainder))
    if not pieces:
        return dataset.limit(0)
    return pieces[0].union(*pieces[1:]) if len(pieces) > 1 else pieces[0]


def _rename_range_id_to_seed_ordinal(batch: Any) -> Any:
    dataframe = lazy.pd.DataFrame(batch)
    if RAY_RANGE_ID_COLUMN not in dataframe.columns:
        raise RayDatasetGenerationError(
            f"RayBackend expected ray.data.range() batches to include {RAY_RANGE_ID_COLUMN!r}."
        )
    return lazy.pd.DataFrame({RAY_SEED_ORDINAL_COLUMN: dataframe[RAY_RANGE_ID_COLUMN].tolist()})


def _row_in_seed_index_range(row: dict[str, Any], *, start: int, end: int) -> bool:
    ordinal = int(row[RAY_SEED_ORDINAL_COLUMN])
    return start <= ordinal <= end


def _file_contents_seed_batch_to_dataframe(batch: Any, *, root_path: str, encoding: str) -> Any:
    dataframe = lazy.pd.DataFrame(batch)
    path_column = _first_present_column(dataframe, ("path", "paths"))
    bytes_column = _first_present_column(dataframe, ("bytes", "data", "item"))
    records: list[dict[str, Any]] = []
    root = Path(root_path).resolve()
    for row in dataframe.to_dict(orient="records"):
        source_path = Path(str(row[path_column])).resolve()
        try:
            relative_path = source_path.relative_to(root).as_posix()
        except ValueError:
            relative_path = source_path.name
        payload = row[bytes_column]
        content = (
            bytes(payload).decode(encoding) if isinstance(payload, (bytes, bytearray, memoryview)) else str(payload)
        )
        records.append(
            {
                "source_kind": "file_contents",
                "source_path": str(source_path),
                "relative_path": relative_path,
                "file_name": source_path.name,
                "content": content,
            }
        )
    return lazy.pd.DataFrame(records)


def _first_present_column(dataframe: Any, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataframe.columns:
            return name
    raise RayDatasetGenerationError(f"RayBackend file-content seed ingestion expected one of {names!r}.")


def _validate_no_seed_ordinal_column(column_names: Sequence[str]) -> None:
    if RAY_SEED_ORDINAL_COLUMN in column_names:
        raise RayBackendConfigurationError(
            f"RayBackend Ray-native seed ingestion reserves column {RAY_SEED_ORDINAL_COLUMN!r}."
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
