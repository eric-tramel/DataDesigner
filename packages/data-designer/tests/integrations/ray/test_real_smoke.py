# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Ray smoke tests for docs/assets recipe paths.

Local command:

    DATA_DESIGNER_RUN_REAL_RAY_SMOKE=1 \
        uv run --package data-designer --group dev --extra ray \
            pytest -q packages/data-designer/tests/integrations/ray/test_real_smoke.py

The OpenAI-backed test also requires OPENAI_API_KEY and skips cleanly when the
credential is absent.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.default_model_settings import (
    get_builtin_model_configs,
    get_builtin_model_providers,
    resolve_seed_default_model_settings,
)
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

REAL_RAY_SMOKE_ENV = "DATA_DESIGNER_RUN_REAL_RAY_SMOKE"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def test_real_ray_markdown_seed_recipe_arrow_refs_smoke(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    ray = _require_real_ray()
    recipe = _load_recipe("docs/assets/recipes/plugin_development/markdown_seed_reader.py")
    seed_dir = tmp_path / "markdown"
    seed_dir.mkdir()
    recipe.create_sample_markdown_files(seed_dir)
    input_df = _markdown_sections_dataframe(recipe, seed_dir)
    input_blocks = [input_df.iloc[:2].reset_index(drop=True), input_df.iloc[2:].reset_index(drop=True)]

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        input_dataset = ray.data.from_pandas(input_blocks)
        config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
        config_builder.add_column(
            ExpressionColumnConfig(
                name="section_summary",
                expr="{{ file_name }} :: {{ section_header }}",
            )
        )
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=stub_model_providers,
            secret_resolver=PlaintextResolver(),
            managed_assets_path=tmp_path / "managed-assets",
            backend=RayBackend(batch_size=2, output="arrow_refs"),
        )

        results = designer.create(config_builder, input_dataset=input_dataset)
        refs = results.output
        output_df = results.load_dataset().to_pandas()
        metrics = results.load_metrics()

        assert results.to_arrow_refs() == refs
        assert len(refs) == metrics.blocks
        assert metrics.total_rows == len(input_df)
        assert metrics.failed_blocks == 0
        assert output_df["section_summary"].notna().all()
        assert output_df["section_summary"].str.contains("::").all()
    finally:
        ray.shutdown()


def test_real_ray_text_to_python_openai_recipe_smoke(tmp_path: Path) -> None:
    ray = _require_real_ray()
    if not os.environ.get(OPENAI_API_KEY_ENV):
        pytest.skip(f"{OPENAI_API_KEY_ENV} is required for the OpenAI-backed Ray smoke test.")

    resolve_seed_default_model_settings()
    recipe = _load_recipe("docs/assets/recipes/code_generation/text_to_python.py")
    config_builder = _single_llm_text_to_python_config(recipe)
    provider = _builtin_provider("openai")

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=[provider],
            managed_assets_path=tmp_path / "managed-assets",
            backend=RayBackend(batch_size=1, output="arrow_refs"),
        )

        results = designer.create(config_builder, num_records=1)
        refs = results.output
        output_df = results.load_dataset().to_pandas()
        metrics = results.load_metrics()
        metrics_payload = metrics.to_dict()

        assert results.to_arrow_refs() == refs
        assert len(refs) == metrics.blocks
        assert metrics.total_rows == 1
        assert metrics.failed_blocks == 0
        assert output_df["instruction"].notna().all()
        assert output_df["instruction"].astype(str).str.len().min() > 0
        assert metrics_payload["model_usage"]
    finally:
        ray.shutdown()


def _require_real_ray() -> Any:
    if os.environ.get(REAL_RAY_SMOKE_ENV) != "1":
        pytest.skip(f"Set {REAL_RAY_SMOKE_ENV}=1 to run real-Ray smoke tests.")
    return pytest.importorskip("ray")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_recipe(relative_path: str) -> Any:
    recipe_path = _repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(f"ray_smoke_{recipe_path.stem}", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe from {recipe_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown_sections_dataframe(recipe: Any, seed_dir: Path) -> Any:
    records: list[dict[str, Any]] = []
    for markdown_path in sorted(seed_dir.glob("*.md")):
        sections = recipe.extract_markdown_sections(
            markdown_text=markdown_path.read_text(encoding="utf-8"),
            fallback_header=markdown_path.name,
        )
        for section_index, (section_header, section_content) in enumerate(sections):
            records.append(
                {
                    "relative_path": markdown_path.name,
                    "file_name": markdown_path.name,
                    "section_index": section_index,
                    "section_header": section_header,
                    "section_content": section_content,
                }
            )
    return lazy.pd.DataFrame(records)


def _single_llm_text_to_python_config(recipe: Any) -> DataDesignerConfigBuilder:
    source_builder = recipe.build_config(model_alias="openai-text")
    config_builder = DataDesignerConfigBuilder(model_configs=[_builtin_model_config("openai-text")])
    for column_config in source_builder.get_column_configs():
        config_builder.add_column(copy.deepcopy(column_config))
        if column_config.name == "instruction":
            break
    return config_builder


def _builtin_model_config(alias: str) -> Any:
    for model_config in get_builtin_model_configs():
        if model_config.alias != alias:
            continue
        inference_parameters = model_config.inference_parameters.model_copy(
            update={"max_parallel_requests": 1, "max_tokens": 256}
        )
        return model_config.model_copy(
            deep=True,
            update={"inference_parameters": inference_parameters, "skip_health_check": True},
        )
    raise RuntimeError(f"No built-in model config found for {alias!r}.")


def _builtin_provider(name: str) -> Any:
    for provider in get_builtin_model_providers():
        if provider.name == name:
            return provider
    raise RuntimeError(f"No built-in provider found for {name!r}.")
