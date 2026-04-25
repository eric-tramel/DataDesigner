# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Ray smoke tests for docs/assets recipe paths."""

from __future__ import annotations

import copy
import importlib
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

pytestmark = pytest.mark.ray_real_smoke


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

    _init_local_ray(ray)
    try:
        input_dataset = ray.data.from_pandas(input_blocks)
        config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
        config_builder.add_column(
            ExpressionColumnConfig(
                name="section_summary",
                expr="{{ file_name }} :: {{ section_header }}",
            )
        )
        managed_assets_path = tmp_path / "managed-assets"
        managed_assets_path.mkdir()
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=stub_model_providers,
            secret_resolver=PlaintextResolver(),
            managed_assets_path=managed_assets_path,
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


@pytest.mark.ray_live_provider
def test_real_ray_text_to_python_openai_recipe_smoke(tmp_path: Path) -> None:
    ray = _require_real_ray()
    if not os.environ.get(OPENAI_API_KEY_ENV):
        pytest.skip(f"{OPENAI_API_KEY_ENV} is required for the OpenAI-backed Ray smoke test.")

    resolve_seed_default_model_settings()
    recipe = _load_recipe("docs/assets/recipes/code_generation/text_to_python.py")
    config_builder = _single_llm_text_to_python_config(recipe)
    provider = _builtin_provider("openai")

    _init_local_ray(ray)
    try:
        managed_assets_path = tmp_path / "managed-assets"
        managed_assets_path.mkdir()
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=[provider],
            managed_assets_path=managed_assets_path,
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


def test_real_ray_streaming_out_of_core_parquet_smoke(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    ray = _require_real_ray()
    num_records = 32
    source_blocks = 8

    _init_local_ray(ray)
    try:
        input_dataset = ray.data.range(num_records, override_num_blocks=source_blocks)
        config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
        config_builder.add_column(ExpressionColumnConfig(name="ticket_key", expr="tenant-{{ id }}::ticket-{{ id }}"))
        config_builder.add_column(ExpressionColumnConfig(name="routing_key", expr="support::{{ id }}"))
        config_builder.add_column(
            ExpressionColumnConfig(
                name="audit_record",
                expr="{{ ticket_key }}::customer-{{ id }}::{{ routing_key }}",
            )
        )
        managed_assets_path = tmp_path / "managed-assets"
        managed_assets_path.mkdir()
        designer = DataDesigner(
            artifact_path=tmp_path / "artifacts",
            model_providers=stub_model_providers,
            secret_resolver=PlaintextResolver(),
            managed_assets_path=managed_assets_path,
            backend=RayBackend(
                batch_size=4,
                output="dataset",
                profile_workers=True,
                trace_enabled=True,
                max_trace_events=64,
            ),
        )

        results = designer.create(config_builder, input_dataset=input_dataset, num_records=num_records)
        parquet_dir = tmp_path / "ray-streaming-output"
        results.dataset.write_parquet(str(parquet_dir))

        persisted = ray.data.read_parquet(str(parquet_dir))
        metrics = results.load_metrics()
        analysis = results.load_analysis()
        sample_batches = persisted.iter_batches(batch_size=5, batch_format="pandas")
        sample = next(iter(sample_batches))

        assert persisted.count() == num_records
        assert metrics.total_rows == num_records
        assert metrics.blocks == source_blocks
        assert metrics.failed_blocks == 0
        assert analysis is not None
        assert analysis.total_rows == num_records
        assert analysis.worker_profiles
        assert analysis.trace_events
        assert sample["ticket_key"].notna().all()
        assert sample["routing_key"].str.contains("::").all()
        assert list(parquet_dir.glob("*.parquet"))
    finally:
        ray.shutdown()


def _require_real_ray() -> Any:
    if os.environ.get(REAL_RAY_SMOKE_ENV) != "1":
        pytest.skip(f"Set {REAL_RAY_SMOKE_ENV}=1 to run real-Ray smoke tests.")
    return pytest.importorskip("ray")


def _init_local_ray(ray: Any) -> None:
    _patch_ray_sandbox_process_discovery()
    ray.init(address="local", num_cpus=2, include_dashboard=False, ignore_reinit_error=True)


def _patch_ray_sandbox_process_discovery() -> None:
    node_module = importlib.import_module("ray._private.node")

    def _sandbox_safe_system_processes(self: Any) -> str:
        all_processes = getattr(self, "all_processes", {})
        pids: list[str] = []
        for processes in all_processes.values():
            if processes:
                pids.append(str(processes[0].process.pid))
        return ",".join(pids)

    node_module.Node._get_system_processes_for_resource_isolation = _sandbox_safe_system_processes


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
