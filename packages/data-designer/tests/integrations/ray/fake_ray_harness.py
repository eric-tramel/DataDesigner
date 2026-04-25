# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy


@dataclass(frozen=True)
class FakeMapBatchesCall:
    fn: Any
    kwargs: dict[str, Any]

    @property
    def fn_constructor_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs.get("fn_constructor_kwargs") or {})

    @property
    def fn_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs.get("fn_kwargs") or {})


class FakeRayDataset:
    def __init__(
        self,
        blocks: list[Any],
        *,
        data_module: FakeRayDataModule | None = None,
        reverse_mapped_blocks: bool = False,
    ) -> None:
        self.blocks = blocks
        self.data_module = data_module
        self.reverse_mapped_blocks = reverse_mapped_blocks
        self.map_batches_kwargs: dict[str, Any] | None = None
        self.map_batches_fn: Any | None = None
        self.map_batches_calls: list[FakeMapBatchesCall] = []
        self.repartition_calls: list[dict[str, Any]] = []

    def map_batches(self, fn: Any, **kwargs: Any) -> FakeRayDataset:
        if self.data_module is not None:
            self.data_module.validate_map_batches_kwargs(kwargs)
        self.map_batches_kwargs = kwargs
        self.map_batches_fn = fn
        self.map_batches_calls.append(FakeMapBatchesCall(fn=fn, kwargs=dict(kwargs)))
        blocks = map_batches_blocks(fn, self.blocks, kwargs)
        if self.reverse_mapped_blocks:
            blocks.reverse()
        mapped = FakeRayDataset(
            blocks,
            data_module=self.data_module,
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )
        mapped.repartition_calls = list(self.repartition_calls)
        return mapped

    def repartition(
        self,
        num_blocks: int | None = None,
        *,
        target_num_rows_per_block: int | None = None,
        shuffle: bool = False,
    ) -> FakeRayDataset:
        self.repartition_calls.append(
            {
                "num_blocks": num_blocks,
                "target_num_rows_per_block": target_num_rows_per_block,
                "shuffle": shuffle,
            }
        )
        repartitioned = FakeRayDataset(
            repartition_blocks(
                self.blocks,
                num_blocks=num_blocks,
                target_num_rows_per_block=target_num_rows_per_block,
            ),
            data_module=self.data_module,
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )
        repartitioned.repartition_calls = list(self.repartition_calls)
        return repartitioned

    def zip(self, other: FakeRayDataset) -> FakeRayDataset:
        other_df = other.to_pandas()
        offset = 0
        blocks: list[lazy.pd.DataFrame] = []
        for block in self.blocks:
            block_df = coerce_pandas_dataframe(block)
            block_len = len(block_df)
            other_block = other_df.iloc[offset : offset + block_len].reset_index(drop=True)
            blocks.append(lazy.pd.concat([block_df.reset_index(drop=True), other_block], axis=1))
            offset += block_len
        zipped = FakeRayDataset(
            blocks,
            data_module=self.data_module or other.data_module,
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )
        zipped.repartition_calls = list(self.repartition_calls)
        return zipped

    def to_arrow_refs(self) -> list[Any]:
        if self.data_module is None:
            return list(self.blocks)
        refs: list[str] = []
        for block in self.blocks:
            ref = f"arrow-ref-{len(self.data_module.ref_blocks)}"
            self.data_module.ref_blocks[ref] = coerce_pandas_dataframe(block)
            refs.append(ref)
        return refs

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat([coerce_pandas_dataframe(block) for block in self.blocks], ignore_index=True)

    def sort(self, column: str) -> FakeRayDataset:
        sorted_df = self.to_pandas().sort_values(column, kind="stable").reset_index(drop=True)
        sorted_dataset = FakeRayDataset([sorted_df], data_module=self.data_module)
        sorted_dataset.repartition_calls = list(self.repartition_calls)
        return sorted_dataset

    def drop_columns(self, columns: list[str]) -> FakeRayDataset:
        dropped = FakeRayDataset(
            [coerce_pandas_dataframe(block).drop(columns=columns) for block in self.blocks],
            data_module=self.data_module,
            reverse_mapped_blocks=self.reverse_mapped_blocks,
        )
        dropped.repartition_calls = list(self.repartition_calls)
        return dropped

    def num_blocks(self) -> int:
        return len(self.blocks)

    def count(self) -> int:
        return sum(len(coerce_pandas_dataframe(block)) for block in self.blocks)


class FakeActorPoolStrategy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeRayDataModule:
    Dataset = FakeRayDataset

    def __init__(
        self,
        *,
        range_blocks: list[lazy.pd.DataFrame] | None = None,
        reverse_mapped_blocks: bool = False,
        validate_map_batches: bool = True,
    ) -> None:
        self.from_arrow_refs_input: list[Any] | None = None
        self.from_pandas_refs_input: list[Any] | None = None
        self.from_pandas_input: Any | None = None
        self.from_items_input: list[Any] | None = None
        self.from_items_kwargs: dict[str, Any] | None = None
        self.range_blocks = range_blocks
        self.reverse_mapped_blocks = reverse_mapped_blocks
        self.range_kwargs: dict[str, Any] | None = None
        self.ref_blocks: dict[str, lazy.pd.DataFrame] = {}
        self.validate_map_batches = validate_map_batches

    def dataset(
        self,
        blocks: list[Any],
        *,
        reverse_mapped_blocks: bool | None = None,
    ) -> FakeRayDataset:
        return FakeRayDataset(
            blocks,
            data_module=self,
            reverse_mapped_blocks=self.reverse_mapped_blocks
            if reverse_mapped_blocks is None
            else reverse_mapped_blocks,
        )

    def ActorPoolStrategy(self, **kwargs: Any) -> FakeActorPoolStrategy:
        return FakeActorPoolStrategy(**kwargs)

    def range(self, num_records: int, **kwargs: Any) -> FakeRayDataset:
        self.range_kwargs = kwargs
        if self.range_blocks is not None:
            return self.dataset(list(self.range_blocks))
        block_count = kwargs.get("override_num_blocks") or 1
        if num_records > 0:
            block_count = min(block_count, num_records)
        blocks = []
        for index in range(block_count):
            start = index * num_records // block_count
            end = (index + 1) * num_records // block_count
            blocks.append(lazy.pd.DataFrame({"id": list(range(start, end))}))
        return self.dataset(blocks)

    def from_arrow_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_arrow_refs_input = refs
        return self.dataset([self._object_ref_to_pandas(ref) for ref in refs])

    def from_items(self, items: list[Any], **kwargs: Any) -> FakeRayDataset:
        self.from_items_input = items
        self.from_items_kwargs = kwargs
        return self.dataset([lazy.pd.DataFrame(items)])

    def from_pandas(self, dataframes: Any) -> FakeRayDataset:
        self.from_pandas_input = dataframes
        if isinstance(dataframes, list):
            return self.dataset(list(dataframes))
        return self.dataset([dataframes])

    def from_pandas_refs(self, refs: list[Any]) -> FakeRayDataset:
        self.from_pandas_refs_input = refs
        return self.dataset(list(refs))

    def validate_map_batches_kwargs(self, kwargs: dict[str, Any]) -> None:
        if not self.validate_map_batches:
            return
        batch_format = kwargs.get("batch_format")
        if batch_format is not None and batch_format != "pandas":
            raise AssertionError(f"fake Ray only supports pandas batches, got {batch_format!r}")
        batch_size = kwargs.get("batch_size")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise AssertionError(f"fake Ray batch_size must be a positive integer, got {batch_size!r}")
        fn_constructor_kwargs = kwargs.get("fn_constructor_kwargs")
        if fn_constructor_kwargs is not None and not isinstance(fn_constructor_kwargs, dict):
            raise AssertionError("fake Ray fn_constructor_kwargs must be a dict when provided")
        fn_kwargs = kwargs.get("fn_kwargs")
        if fn_kwargs is not None and not isinstance(fn_kwargs, dict):
            raise AssertionError("fake Ray fn_kwargs must be a dict when provided")

    def _object_ref_to_pandas(self, ref: Any) -> lazy.pd.DataFrame:
        if isinstance(ref, str) and ref in self.ref_blocks:
            return self.ref_blocks[ref]
        return coerce_pandas_dataframe(ref)


class FakeObjectRef:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __await__(self) -> Any:
        async def _value() -> Any:
            return self.value

        return _value().__await__()


class FakeRemoteMethod:
    def __init__(self, method: Any) -> None:
        self._method = method

    def remote(self, *args: Any, **kwargs: Any) -> FakeObjectRef:
        return FakeObjectRef(self._method(*args, **kwargs))


class FakeActorHandle:
    def __init__(self, actor: Any) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> FakeRemoteMethod:
        return FakeRemoteMethod(getattr(self._actor, name))


class FakeRemoteClass:
    def __init__(self, cls: type) -> None:
        self._cls = cls

    def remote(self, *args: Any, **kwargs: Any) -> FakeActorHandle:
        return FakeActorHandle(self._cls(*args, **kwargs))


def install_fake_ray(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data_module: FakeRayDataModule | None = None,
    with_remote: bool = False,
    initialized: bool = True,
) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = data_module or FakeRayDataModule()
    fake_ray.is_initialized = lambda: initialized
    fake_ray.init = lambda: None
    fake_ray.get = fake_ray_get
    if with_remote:
        fake_ray.remote = lambda cls: FakeRemoteClass(cls)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


def fake_ray_get(ref: Any) -> Any:
    if isinstance(ref, FakeObjectRef):
        return ref.value
    if isinstance(ref, list):
        return [fake_ray_get(item) for item in ref]
    return ref


def map_batches_blocks(fn: Any, blocks: list[Any], kwargs: dict[str, Any]) -> list[lazy.pd.DataFrame]:
    fn_kwargs = kwargs.get("fn_kwargs") or {}
    fn_constructor_kwargs = kwargs.get("fn_constructor_kwargs") or {}
    map_fn = fn(**fn_constructor_kwargs) if isinstance(fn, type) else fn
    return [coerce_pandas_dataframe(map_fn(coerce_pandas_dataframe(block), **fn_kwargs)) for block in blocks]


def repartition_blocks(
    blocks: list[Any],
    *,
    num_blocks: int | None,
    target_num_rows_per_block: int | None,
) -> list[lazy.pd.DataFrame]:
    if num_blocks is None and target_num_rows_per_block is None:
        return [coerce_pandas_dataframe(block) for block in blocks]
    dataframe = (
        lazy.pd.concat([coerce_pandas_dataframe(block) for block in blocks], ignore_index=True)
        if blocks
        else lazy.pd.DataFrame()
    )
    if num_blocks is not None:
        return split_exact_blocks(dataframe, num_blocks)
    if target_num_rows_per_block is None:
        return [coerce_pandas_dataframe(block) for block in blocks]
    return [
        dataframe.iloc[start : start + target_num_rows_per_block].reset_index(drop=True)
        for start in range(0, max(len(dataframe), 1), target_num_rows_per_block)
    ]


def split_exact_blocks(dataframe: lazy.pd.DataFrame, num_blocks: int) -> list[lazy.pd.DataFrame]:
    block_size, remainder = divmod(len(dataframe), num_blocks)
    blocks: list[lazy.pd.DataFrame] = []
    start = 0
    for block_index in range(num_blocks):
        stop = start + block_size + (1 if block_index < remainder else 0)
        blocks.append(dataframe.iloc[start:stop].reset_index(drop=True))
        start = stop
    return blocks


def coerce_pandas_dataframe(value: Any) -> lazy.pd.DataFrame:
    if isinstance(value, lazy.pd.DataFrame):
        return value
    to_pandas = getattr(value, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    return value
