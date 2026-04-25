# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.dataset_builders.block_execution import BlockExecutionResult
from data_designer.engine.storage.artifact_storage import (
    BATCH_FILE_NAME_FORMAT,
    FINAL_DATASET_FOLDER_NAME,
    METADATA_FILENAME,
    PROCESSORS_OUTPUTS_FOLDER_NAME,
    ArtifactStorage,
)

_ARTIFACT_COLUMN_PREFIX = "__data_designer_ray_artifact__"
_DROPPED_ARTIFACT_KIND = "dropped"
_PROCESSOR_ARTIFACT_KIND = "processor"
_DROPPED_COLUMNS_FOLDER_NAME = "dropped-columns-parquet-files"


class DataDesignerRayDatasink:
    """Datasink-compatible writer for Data Designer Ray artifacts."""

    def __init__(
        self,
        *,
        base_dataset_path: Path,
        dataset_name: str,
        target_num_records: int,
        buffer_size: int,
        min_rows_per_write: int | None,
        supports_distributed_writes: bool,
    ) -> None:
        self.base_dataset_path = Path(base_dataset_path)
        self.dataset_name = dataset_name
        self.target_num_records = target_num_records
        self.buffer_size = buffer_size
        self._min_rows_per_write = min_rows_per_write
        self._supports_distributed_writes = supports_distributed_writes

    def get_name(self) -> str:
        return "DataDesignerRayDatasink"

    @property
    def min_rows_per_write(self) -> int | None:
        return self._min_rows_per_write

    @property
    def supports_distributed_writes(self) -> bool:
        return self._supports_distributed_writes

    @property
    def final_dataset_path(self) -> Path:
        return self.base_dataset_path / FINAL_DATASET_FOLDER_NAME

    @property
    def dropped_columns_dataset_path(self) -> Path:
        return self.base_dataset_path / _DROPPED_COLUMNS_FOLDER_NAME

    @property
    def processors_outputs_path(self) -> Path:
        return self.base_dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME

    @property
    def metadata_file_path(self) -> Path:
        return self.base_dataset_path / METADATA_FILENAME

    def on_write_start(self, schema: Any | None = None) -> None:
        del schema
        ArtifactStorage.mkdir_if_needed(self.final_dataset_path)

    def write(self, blocks: Iterable[Any], ctx: Any) -> dict[str, Any] | None:
        dataframe = _blocks_to_pandas_dataframe(blocks)
        if len(dataframe) == 0:
            return None

        batch_number = int(getattr(ctx, "task_idx", 0))
        parquet_file_name = BATCH_FILE_NAME_FORMAT.format(batch_number=batch_number)
        final_dataframe, dropped_dataframe, processor_dataframes = split_ray_artifact_columns(dataframe)
        file_paths: dict[str, Any] = {FINAL_DATASET_FOLDER_NAME: []}

        ArtifactStorage.mkdir_if_needed(self.final_dataset_path)
        final_path = self.final_dataset_path / parquet_file_name
        final_dataframe.to_parquet(final_path, index=False)
        file_paths[FINAL_DATASET_FOLDER_NAME].append(str(final_path.relative_to(self.base_dataset_path)))

        if len(dropped_dataframe.columns) > 0:
            ArtifactStorage.mkdir_if_needed(self.dropped_columns_dataset_path)
            dropped_path = self.dropped_columns_dataset_path / parquet_file_name
            dropped_dataframe.to_parquet(dropped_path, index=False)
            file_paths[_DROPPED_COLUMNS_FOLDER_NAME] = [str(dropped_path.relative_to(self.base_dataset_path))]

        processor_file_paths: dict[str, list[str]] = {}
        for processor_name, processor_dataframe in processor_dataframes.items():
            processor_path = self.processors_outputs_path / processor_name / parquet_file_name
            ArtifactStorage.mkdir_if_needed(processor_path.parent)
            processor_dataframe.to_parquet(processor_path, index=False)
            processor_file_paths[processor_name] = [str(processor_path.relative_to(self.base_dataset_path))]
        if processor_file_paths:
            file_paths["processor-files"] = processor_file_paths

        return {
            "num_rows": len(final_dataframe),
            "schema": _dataframe_schema(final_dataframe),
            "file_paths": file_paths,
        }

    def on_write_complete(self, write_result: Any) -> None:
        write_returns = [item for item in _write_result_returns(write_result) if item is not None]
        actual_num_records = sum(int(item["num_rows"]) for item in write_returns)
        metadata: dict[str, Any] = {
            "target_num_records": self.target_num_records,
            "actual_num_records": actual_num_records,
            "total_num_batches": len(write_returns),
            "buffer_size": self.buffer_size,
            "dataset_name": self.dataset_name,
            "file_paths": _combine_file_paths(item["file_paths"] for item in write_returns),
            "num_completed_batches": len(write_returns),
        }
        if write_returns:
            metadata["schema"] = write_returns[0]["schema"]

        ArtifactStorage.mkdir_if_needed(self.base_dataset_path)
        with open(self.metadata_file_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

    def on_write_failed(self, exception: Exception) -> None:
        del exception


def append_ray_artifact_columns(dataframe: Any, block_result: BlockExecutionResult) -> Any:
    """Append internal artifact columns that the Ray datasink can split back out."""
    output = dataframe.copy()
    dropped_dataframe = _dropped_columns_dataframe(
        raw_dataframe=block_result.raw_dataframe,
        final_dataframe=block_result.dataframe,
    )
    if len(dropped_dataframe.columns) > 0:
        output = _append_prefixed_dataframe_columns(output, dropped_dataframe, (_DROPPED_ARTIFACT_KIND,))

    for processor_name, processor_dataframe in block_result.processor_artifacts.items():
        output = _append_prefixed_dataframe_columns(
            output,
            processor_dataframe.reset_index(drop=True),
            (_PROCESSOR_ARTIFACT_KIND, processor_name),
        )
    return output


def split_ray_artifact_columns(dataframe: Any) -> tuple[Any, Any, dict[str, Any]]:
    final_columns = [column for column in dataframe.columns if not str(column).startswith(_ARTIFACT_COLUMN_PREFIX)]
    dropped_columns: dict[str, Any] = {}
    processor_columns: dict[str, dict[str, Any]] = {}

    for column in dataframe.columns:
        column_name = str(column)
        if not column_name.startswith(_ARTIFACT_COLUMN_PREFIX):
            continue
        kind, parts = _parse_artifact_column_name(column_name)
        if kind == _DROPPED_ARTIFACT_KIND:
            dropped_columns[parts[0]] = dataframe[column]
        elif kind == _PROCESSOR_ARTIFACT_KIND:
            processor_name, processor_column = parts
            processor_columns.setdefault(processor_name, {})[processor_column] = dataframe[column]

    processor_dataframes = {
        processor_name: lazy.pd.DataFrame(columns) for processor_name, columns in processor_columns.items()
    }
    return lazy.pd.DataFrame(dataframe[final_columns]), lazy.pd.DataFrame(dropped_columns), processor_dataframes


def _dropped_columns_dataframe(*, raw_dataframe: Any, final_dataframe: Any) -> Any:
    dropped_column_names = [column for column in raw_dataframe.columns if column not in final_dataframe.columns]
    if not dropped_column_names:
        return lazy.pd.DataFrame(index=range(len(final_dataframe)))
    if len(raw_dataframe) == len(final_dataframe):
        return raw_dataframe[dropped_column_names].reset_index(drop=True)
    if final_dataframe.index.isin(raw_dataframe.index).all():
        return raw_dataframe.loc[final_dataframe.index, dropped_column_names].reset_index(drop=True)
    raise ValueError("Ray artifact dropped-column outputs must preserve row identity.")


def _append_prefixed_dataframe_columns(dataframe: Any, artifact_dataframe: Any, path_parts: tuple[str, ...]) -> Any:
    if len(dataframe) != len(artifact_dataframe):
        raise ValueError("Ray artifact processor outputs must preserve row counts.")
    for column_name in artifact_dataframe.columns:
        dataframe[_artifact_column_name(*path_parts, str(column_name))] = artifact_dataframe[column_name].to_list()
    return dataframe


def _artifact_column_name(kind: str, *parts: str) -> str:
    encoded_parts = "__".join(_encode_artifact_part(part) for part in parts)
    return f"{_ARTIFACT_COLUMN_PREFIX}{kind}__{encoded_parts}"


def _encode_artifact_part(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_artifact_part(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii")).decode("utf-8")


def _parse_artifact_column_name(column_name: str) -> tuple[str, tuple[str, ...]]:
    suffix = column_name.removeprefix(_ARTIFACT_COLUMN_PREFIX)
    kind, _, encoded_parts = suffix.partition("__")
    return kind, tuple(_decode_artifact_part(part) for part in encoded_parts.split("__") if part)


def _blocks_to_pandas_dataframe(blocks: Iterable[Any]) -> Any:
    dataframes = [_coerce_block_to_pandas_dataframe(block) for block in blocks]
    if not dataframes:
        return lazy.pd.DataFrame()
    if len(dataframes) == 1:
        return dataframes[0]
    return lazy.pd.concat(dataframes, ignore_index=True)


def _coerce_block_to_pandas_dataframe(block: Any) -> Any:
    if isinstance(block, lazy.pd.DataFrame):
        return block
    to_pandas = getattr(block, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return lazy.pd.DataFrame(block)


def _dataframe_schema(dataframe: Any) -> dict[str, str]:
    schema = lazy.pa.Table.from_pandas(dataframe, preserve_index=False).schema
    return {field.name: str(field.type) for field in schema}


def _write_result_returns(write_result: Any) -> list[Any]:
    if isinstance(write_result, list):
        return write_result
    write_returns = getattr(write_result, "write_returns", None)
    if write_returns is None:
        return []
    return list(write_returns)


def _combine_file_paths(file_paths_items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {FINAL_DATASET_FOLDER_NAME: []}
    processor_files: dict[str, list[str]] = {}
    dropped_files: list[str] = []

    for file_paths in file_paths_items:
        combined[FINAL_DATASET_FOLDER_NAME].extend(file_paths.get(FINAL_DATASET_FOLDER_NAME, []))
        dropped_files.extend(file_paths.get(_DROPPED_COLUMNS_FOLDER_NAME, []))
        for processor_name, paths in file_paths.get("processor-files", {}).items():
            processor_files.setdefault(processor_name, []).extend(paths)

    combined[FINAL_DATASET_FOLDER_NAME] = sorted(combined[FINAL_DATASET_FOLDER_NAME])
    if dropped_files:
        combined[_DROPPED_COLUMNS_FOLDER_NAME] = sorted(dropped_files)
    if processor_files:
        combined["processor-files"] = {name: sorted(paths) for name, paths in sorted(processor_files.items())}
    return combined
