# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _clear_modules(monkeypatch: pytest.MonkeyPatch, prefixes: tuple[str, ...]) -> None:
    for module_name in list(sys.modules):
        if module_name in prefixes or module_name.startswith(tuple(f"{prefix}." for prefix in prefixes)):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _block_ray_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    imported: list[str] = []
    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> ModuleType:
        imported.append(name)
        if name == "ray" or name.startswith("ray."):
            raise AssertionError(f"{name} should not be imported")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return imported


def _make_ray_missing(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    imported: list[str] = []
    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> ModuleType:
        imported.append(name)
        if name == "ray" or name.startswith("ray."):
            raise ImportError("No module named 'ray'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return imported


@pytest.mark.parametrize(
    "module_name",
    [
        "data_designer.interface",
        "data_designer.interface.data_designer",
        "data_designer.config",
        "data_designer.config.config_builder",
        "data_designer.config.column_configs",
    ],
)
def test_interface_and_config_imports_do_not_import_ray(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    _clear_modules(monkeypatch, ("ray", "data_designer.interface", "data_designer.config"))
    imported = _block_ray_imports(monkeypatch)

    importlib.import_module(module_name)

    assert "ray" not in sys.modules
    assert not any(name == "ray" or name.startswith("ray.") for name in imported)


def test_ray_backend_construction_does_not_import_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_modules(monkeypatch, ("ray", "data_designer.integrations.ray"))
    imported = _block_ray_imports(monkeypatch)

    ray_module = importlib.import_module("data_designer.integrations.ray")
    backend = ray_module.RayBackend(auto_init=True)

    assert backend.auto_init is True
    assert "ray" not in sys.modules
    assert not any(name == "ray" or name.startswith("ray.") for name in imported)


def test_ray_backend_usage_without_ray_has_optional_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_modules(monkeypatch, ("ray", "data_designer.integrations.ray"))
    imported = _make_ray_missing(monkeypatch)

    ray_module = importlib.import_module("data_designer.integrations.ray")
    config_module = importlib.import_module("data_designer.config.config_builder")
    backend = ray_module.RayBackend()
    config_builder = config_module.DataDesignerConfigBuilder(model_configs=[])

    with pytest.raises(ImportError, match=r"data-designer\[ray\]"):
        backend.create(
            data_designer=object(),
            config_builder=config_builder,
            num_records=1,
            dataset_name="dataset",
        )

    assert "ray" in imported
