# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Ray smoke tests for docs/assets recipe paths."""

from __future__ import annotations

import copy
import importlib.util
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
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend, RayDataContextOptions, RayExecutionResources, RayInputRepartition
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.ray_real_smoke


def test_real_ray_markdown_seed_recipe_arrow_refs_smoke(
    tmp_path: Path,
    local_ray: Any,
    real_ray_smoke_paths: Any,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    recipe = _load_recipe("docs/assets/recipes/plugin_development/markdown_seed_reader.py")
    seed_dir = tmp_path / "markdown"
    seed_dir.mkdir()
    recipe.create_sample_markdown_files(seed_dir)
    input_df = _markdown_sections_dataframe(recipe, seed_dir)
    input_blocks = [input_df.iloc[:2].reset_index(drop=True), input_df.iloc[2:].reset_index(drop=True)]

    input_dataset = local_ray.data.from_pandas(input_blocks)
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="section_summary",
            expr="{{ file_name }} :: {{ section_header }}",
        )
    )
    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
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


@pytest.mark.ray_live_provider
def test_real_ray_text_to_python_openai_recipe_smoke(
    live_provider_local_ray: Any,
    real_ray_smoke_paths: Any,
) -> None:
    resolve_seed_default_model_settings()
    recipe = _load_recipe("docs/assets/recipes/code_generation/text_to_python.py")
    config_builder = _single_llm_text_to_python_config(recipe)
    provider = _builtin_provider("openai")

    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=[provider],
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
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


def test_real_ray_streaming_out_of_core_parquet_smoke(
    tmp_path: Path,
    local_ray: Any,
    real_ray_smoke_paths: Any,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    num_records = 32
    source_blocks = 8

    input_dataset = local_ray.data.range(num_records, override_num_blocks=source_blocks)
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="ticket_key", expr="tenant-{{ id }}::ticket-{{ id }}"))
    config_builder.add_column(ExpressionColumnConfig(name="routing_key", expr="support::{{ id }}"))
    config_builder.add_column(
        ExpressionColumnConfig(
            name="audit_record",
            expr="{{ ticket_key }}::customer-{{ id }}::{{ routing_key }}",
        )
    )
    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
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

    persisted = local_ray.data.read_parquet(str(parquet_dir))
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
    assert analysis.ray_dataset_stats is not None
    assert analysis.ray_dataset_stats.stats_text is not None or analysis.ray_dataset_stats.warnings
    assert sample["ticket_key"].notna().all()
    assert sample["routing_key"].str.contains("::").all()
    assert list(parquet_dir.glob("*.parquet"))


def test_real_ray_input_dataset_repartition_smoke(
    local_ray: Any,
    real_ray_smoke_paths: Any,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    input_dataframe = lazy.pd.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})
    input_dataset = local_ray.data.from_pandas(input_dataframe)
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="x_label", expr="{{ x }}-{{ label }}"))
    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
        backend=RayBackend(
            batch_size=2,
            output="arrow_refs",
            input_repartition=RayInputRepartition(num_blocks=2),
        ),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()
    metrics = results.load_metrics()

    assert len(results.output) == 2
    assert metrics.blocks == 2
    assert sorted(output_df["x"].tolist()) == [1, 2, 3, 4]
    assert output_df["x_label"].notna().all()


def test_real_ray_local_file_seed_ingestion_smoke(
    tmp_path: Path,
    local_ray: Any,
    real_ray_smoke_paths: Any,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    seed_path = tmp_path / "seed.csv"
    lazy.pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]}).to_csv(seed_path, index=False)
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    config_builder.add_column(ExpressionColumnConfig(name="x_label", expr="{{ x }}-{{ label }}"))
    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
        backend=RayBackend(batch_size=2, output="dataset"),
    )

    results = designer.create(config_builder, num_records=5)
    output_df = results.load_dataset().to_pandas()

    assert local_ray.is_initialized()
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_real_ray_data_context_controls_smoke(
    local_ray: Any,
    real_ray_smoke_paths: Any,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    current_context = local_ray.data.DataContext.get_current()
    current_context.enable_progress_bars = True
    current_context.max_errored_blocks = 0
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="id_label", expr="row-{{ id }}"))
    designer = DataDesigner(
        artifact_path=real_ray_smoke_paths.artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=real_ray_smoke_paths.managed_assets_path,
        backend=RayBackend(
            batch_size=2,
            output="dataset",
            data_context_options=RayDataContextOptions(
                resource_limits=RayExecutionResources(cpu=1),
                enable_progress_bars=False,
                max_errored_blocks=0,
            ),
        ),
    )

    results = designer.create(config_builder, num_records=4)
    output_df = results.load_dataset().to_pandas().sort_values("id").reset_index(drop=True)
    metrics = results.load_metrics()

    assert output_df.to_dict(orient="records") == [
        {"id": 0, "id_label": "row-0"},
        {"id": 1, "id_label": "row-1"},
        {"id": 2, "id_label": "row-2"},
        {"id": 3, "id_label": "row-3"},
    ]
    assert metrics.failed_blocks == 0
    assert local_ray.data.DataContext.get_current().enable_progress_bars is True


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
