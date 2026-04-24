# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients.throttle_manager import CAPACITY_POLL_INTERVAL, ThrottleDomain
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray.metrics import RayWorkerMetrics
from data_designer.integrations.ray.throttling import RayThrottleManagerProxy, create_ray_throttle_manager
from data_designer.interface.data_designer import DataDesigner


class FakeRayDataset:
    def __init__(self, blocks: list[lazy.pd.DataFrame]) -> None:
        self.blocks = blocks

    def map_batches(self, fn: Any, **kwargs: Any) -> FakeRayDataset:
        fn_kwargs = kwargs.get("fn_kwargs") or {}
        return FakeRayDataset([fn(block, **fn_kwargs) for block in self.blocks])

    def to_pandas(self) -> lazy.pd.DataFrame:
        return lazy.pd.concat(self.blocks, ignore_index=True)

    def num_blocks(self) -> int:
        return len(self.blocks)


class FakeRayDataModule:
    Dataset = FakeRayDataset

    def range(self, num_records: int) -> FakeRayDataset:
        return FakeRayDataset([lazy.pd.DataFrame({"id": list(range(num_records))})])


class FakeObjectRef:
    def __init__(self, value: Any) -> None:
        self.value = value


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


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_ray = types.ModuleType("ray")
    fake_ray.data = FakeRayDataModule()
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda: None
    fake_ray.remote = lambda cls: FakeRemoteClass(cls)
    fake_ray.get = lambda ref: ref.value
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_ray_throttle_proxy_coordinates_shared_provider_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    first_worker = create_ray_throttle_manager(fake_ray, RunConfig())
    second_worker = RayThrottleManagerProxy(first_worker._coordinator)

    first_worker.register(
        provider_name="openai",
        model_id="gpt-4.1",
        alias="writer",
        max_parallel_requests=1,
    )

    assert first_worker.try_acquire(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT) == 0.0
    assert (
        second_worker.try_acquire(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT)
        == CAPACITY_POLL_INTERVAL
    )

    first_worker.release_success(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT)

    assert second_worker.try_acquire(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT) == 0.0
    snapshot = second_worker.snapshot()
    assert snapshot["global_caps"] == [
        {
            "provider_name": "openai",
            "model_id": "gpt-4.1",
            "effective_max": 1,
            "limits_by_alias": {"writer": 1},
        }
    ]


def test_ray_backend_exposes_global_throttle_snapshot_in_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
    stub_model_providers: Any,
) -> None:
    _install_fake_ray(monkeypatch)

    def generate_batch(
        batch: lazy.pd.DataFrame,
        *,
        worker_options: Any,
        metrics_collector: Any | None = None,
        **_: Any,
    ) -> lazy.pd.DataFrame:
        throttle_manager = worker_options.throttle_manager
        assert throttle_manager is not None
        throttle_manager.register(
            provider_name="openai",
            model_id="gpt-4.1",
            alias="writer",
            max_parallel_requests=1,
        )
        throttle_manager.acquire_sync(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT)
        throttle_manager.release_success(provider_name="openai", model_id="gpt-4.1", domain=ThrottleDomain.CHAT)
        ray_backend_module._record_worker_metrics(
            metrics_collector,
            RayWorkerMetrics(total_rows=len(batch), blocks=1, elapsed_seconds=0.25),
        )
        return batch

    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)
    metrics = results.load_metrics()

    assert results.load_dataset().to_pandas().to_dict(orient="records") == [{"id": 0}, {"id": 1}]
    assert metrics.total_rows == 2
    assert metrics.throttle is not None
    assert metrics.throttle["global_caps"][0]["effective_max"] == 1
    assert metrics.throttle["domains"][0]["domain"] == ThrottleDomain.CHAT.value
