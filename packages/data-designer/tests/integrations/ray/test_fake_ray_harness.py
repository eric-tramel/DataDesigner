# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from fake_ray_harness import FakeRayDataModule

import data_designer.lazy_heavy_imports as lazy

pytestmark = pytest.mark.ray_fake


def test_fake_ray_records_constructor_batch_and_compute_kwargs(
    fake_ray_installer: Any,
) -> None:
    fake_ray = fake_ray_installer()
    dataset = fake_ray.data.dataset(
        [
            lazy.pd.DataFrame({"id": [0, 1]}),
            lazy.pd.DataFrame({"id": [2]}),
        ]
    )
    compute = fake_ray.data.ActorPoolStrategy(min_size=1, max_size=2)

    class PrefixWorker:
        def __init__(self, *, prefix: str) -> None:
            self._prefix = prefix

        def __call__(self, batch: lazy.pd.DataFrame, *, suffix: str) -> lazy.pd.DataFrame:
            output = batch.copy()
            output["label"] = [f"{self._prefix}-{value}-{suffix}" for value in output["id"]]
            return output

    mapped = dataset.map_batches(
        PrefixWorker,
        batch_size=2,
        batch_format="pandas",
        compute=compute,
        fn_constructor_kwargs={"prefix": "row"},
        fn_kwargs={"suffix": "done"},
    )

    assert dataset.map_batches_fn is PrefixWorker
    assert dataset.map_batches_kwargs is not None
    assert dataset.map_batches_kwargs["compute"] is compute
    assert dataset.map_batches_calls[0].fn_constructor_kwargs == {"prefix": "row"}
    assert dataset.map_batches_calls[0].fn_kwargs == {"suffix": "done"}
    assert mapped.to_pandas().to_dict(orient="records") == [
        {"id": 0, "label": "row-0-done"},
        {"id": 1, "label": "row-1-done"},
        {"id": 2, "label": "row-2-done"},
    ]


def test_fake_ray_can_reverse_mapped_blocks() -> None:
    data_module = FakeRayDataModule(reverse_mapped_blocks=True)
    dataset = data_module.dataset(
        [
            lazy.pd.DataFrame({"id": [0]}),
            lazy.pd.DataFrame({"id": [1]}),
        ]
    )

    mapped = dataset.map_batches(lambda batch: batch, batch_size=1, batch_format="pandas")

    assert mapped.to_pandas()["id"].tolist() == [1, 0]


def test_fake_ray_materializes_arrow_refs_without_rerunning_map(
    fake_ray_installer: Any,
) -> None:
    fake_ray = fake_ray_installer()
    calls = 0
    dataset = fake_ray.data.range(2)

    def add_generated_column(batch: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        nonlocal calls
        calls += 1
        output = batch.copy()
        output["generated"] = calls
        return output

    refs = dataset.map_batches(add_generated_column, batch_size=2, batch_format="pandas").to_arrow_refs()
    rebuilt = fake_ray.data.from_arrow_refs(refs)

    assert refs == ["arrow-ref-0"]
    assert fake_ray.data.from_arrow_refs_input == refs
    assert rebuilt.to_pandas().to_dict(orient="records") == [
        {"id": 0, "generated": 1},
        {"id": 1, "generated": 1},
    ]
    assert calls == 1


def test_fake_ray_remote_actor_methods_return_object_refs(
    fake_ray_installer: Any,
) -> None:
    fake_ray = fake_ray_installer(with_remote=True)

    class Counter:
        def __init__(self, *, start: int = 0) -> None:
            self._value = start

        def add(self, amount: int) -> int:
            self._value += amount
            return self._value

    actor = fake_ray.remote(Counter).remote(start=2)

    assert fake_ray.get(actor.add.remote(3)) == 5
    assert fake_ray.get(actor.add.remote(4)) == 9


def test_fake_ray_validates_pandas_batch_contract() -> None:
    dataset = FakeRayDataModule().dataset([lazy.pd.DataFrame({"id": [0]})])

    with pytest.raises(AssertionError, match="pandas batches"):
        dataset.map_batches(lambda batch: batch, batch_size=1, batch_format="numpy")

    with pytest.raises(AssertionError, match="positive integer"):
        dataset.map_batches(lambda batch: batch, batch_size=0, batch_format="pandas")
