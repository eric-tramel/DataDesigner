# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from collections.abc import Callable

import pytest
from fake_ray_harness import FakeRayDataModule, install_fake_ray


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
