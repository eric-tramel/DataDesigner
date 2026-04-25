# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
import tempfile
import types
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fake_ray_harness import FakeRayDataModule, install_fake_ray

REAL_RAY_SMOKE_ENV = "DATA_DESIGNER_RUN_REAL_RAY_SMOKE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
RAY_UV_RUNTIME_ENV = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"


@dataclass(frozen=True)
class RaySmokePaths:
    artifact_path: Path
    managed_assets_path: Path
    runtime_path: Path


@pytest.fixture
def fake_ray_installer(monkeypatch: pytest.MonkeyPatch) -> Callable[..., types.ModuleType]:
    def _install(
        *,
        data_module: FakeRayDataModule | None = None,
        with_remote: bool = False,
        initialized: bool = True,
    ) -> types.ModuleType:
        return install_fake_ray(
            monkeypatch,
            data_module=data_module,
            with_remote=with_remote,
            initialized=initialized,
        )

    return _install


@pytest.fixture
def ray_uv_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ray reads this while importing/initializing its runtime-env hook. Set it
    # before importing Ray so local `uv run` smoke tests work in macOS sandboxes.
    if not os.environ.get(RAY_UV_RUNTIME_ENV):
        monkeypatch.setenv(RAY_UV_RUNTIME_ENV, "0")


@pytest.fixture
def real_ray(ray_uv_runtime_env: None) -> types.ModuleType:
    del ray_uv_runtime_env
    if os.environ.get(REAL_RAY_SMOKE_ENV) != "1":
        pytest.skip(f"Set {REAL_RAY_SMOKE_ENV}=1 to run real-Ray smoke tests.")
    return pytest.importorskip("ray")


@pytest.fixture
def real_ray_smoke_paths(tmp_path: Path) -> Generator[RaySmokePaths]:
    with tempfile.TemporaryDirectory(prefix="dd-ray-", dir="/tmp") as runtime_dir:
        paths = RaySmokePaths(
            artifact_path=tmp_path / "artifacts",
            managed_assets_path=tmp_path / "managed-assets",
            runtime_path=Path(runtime_dir),
        )
        paths.managed_assets_path.mkdir()
        yield paths


@pytest.fixture
def ray_sandbox_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    real_ray: types.ModuleType,
) -> None:
    node_module = importlib.import_module("ray._private.node")
    monkeypatch.setattr(
        node_module.Node,
        "_get_system_processes_for_resource_isolation",
        _sandbox_safe_system_processes,
    )
    del real_ray


@pytest.fixture
def local_ray(
    real_ray: types.ModuleType,
    real_ray_smoke_paths: RaySmokePaths,
    ray_sandbox_compatibility: None,
) -> Generator[types.ModuleType]:
    del ray_sandbox_compatibility
    real_ray.shutdown()
    real_ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=False,
        _temp_dir=str(real_ray_smoke_paths.runtime_path),
    )
    try:
        yield real_ray
    finally:
        real_ray.shutdown()


@pytest.fixture
def openai_api_key() -> str:
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        pytest.skip(f"{OPENAI_API_KEY_ENV} is required for the OpenAI-backed Ray smoke test.")
    return api_key


@pytest.fixture
def live_provider_local_ray(
    openai_api_key: str,
    local_ray: types.ModuleType,
) -> types.ModuleType:
    del openai_api_key
    return local_ray


def _sandbox_safe_system_processes(self: Any) -> str:
    all_processes = getattr(self, "all_processes", {})
    pids: list[str] = []
    for processes in all_processes.values():
        if processes:
            pids.append(str(processes[0].process.pid))
    return ",".join(pids)
