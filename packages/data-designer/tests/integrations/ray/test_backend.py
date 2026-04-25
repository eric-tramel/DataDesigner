# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from fake_ray_harness import FakeRayDataset, install_fake_ray

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig, LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import SchemaTransformProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import IndexRange, SamplingStrategy
from data_designer.config.seed_source import DirectorySeedSource, FileContentsSeedSource, LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.testing.seed_readers import LineFanoutDirectorySeedReader
from data_designer.integrations.ray import (
    RayBackend,
    RayBackendConfigurationError,
    RayBlockPlanning,
    RayDatasetCreationResults,
    RayDatasetGenerationError,
    RayDatasetMetrics,
    RayExecutionOptions,
    RayInputRepartition,
)
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray import seed_planning as ray_seed_planning
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.ray_fake


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _input_expression_config_builder(stub_model_configs: Any) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="x_label",
            expr="{{ x }}-{{ label }}",
        )
    )
    return config_builder


def _llm_config_builder(stub_model_configs: Any) -> DataDesignerConfigBuilder:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        LLMTextColumnConfig(
            name="text",
            prompt="Say hello.",
            model_alias="stub-model",
        )
    )
    return config_builder


def test_ray_backend_uses_input_dataset_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, ray_remote_args={"num_cpus": 0.5}),
    )
    designer.set_run_config(RunConfig(buffer_size=2))

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_kwargs["num_cpus"] == 0.5
    assert "fn_kwargs" not in input_dataset.map_batches_kwargs
    assert input_dataset.map_batches_kwargs["fn_constructor_kwargs"]["execution_payload"].use_input_dataset is True
    assert "ray_remote_args" not in input_dataset.map_batches_kwargs
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_override_num_blocks_for_from_scratch_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, override_num_blocks=3, read_concurrency=2),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=5)

    assert fake_ray.data.range_kwargs == {"override_num_blocks": 3, "concurrency": 2}
    assert results.load_metrics().blocks == 3


@pytest.mark.parametrize(
    ("num_records", "target_block_size", "min_blocks", "max_blocks", "expected_blocks"),
    [
        (3, 100, None, None, 1),
        (20, 4, 2, 3, 3),
    ],
)
def test_ray_backend_resolves_target_block_size_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
    num_records: int,
    target_block_size: int,
    min_blocks: int | None,
    max_blocks: int | None,
    expected_blocks: int,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=2,
            target_block_size=target_block_size,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
        ),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=num_records)

    assert fake_ray.data.range_kwargs == {"override_num_blocks": expected_blocks}
    assert results.load_metrics().blocks == expected_blocks


def test_ray_backend_rejects_block_planning_with_input_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"id": [0]})])
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(override_num_blocks=2),
    )

    with pytest.raises(RayBackendConfigurationError, match="block planning controls"):
        designer.create(stub_sampler_only_config_builder, input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_repartitions_input_dataset_by_num_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})])
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=2,
            input_repartition=RayInputRepartition(num_blocks=2, shuffle=True),
        ),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_dataset = results.load_dataset()
    output_df = output_dataset.to_pandas()

    assert input_dataset.repartition_calls == [{"num_blocks": 2, "target_num_rows_per_block": None, "shuffle": True}]
    assert output_dataset.repartition_calls == input_dataset.repartition_calls
    assert output_dataset.num_blocks() == 2
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
        {"x": 4, "label": "d", "x_label": "4-d"},
    ]


def test_ray_backend_repartitions_object_refs_by_target_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_refs = [
        lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}),
        lazy.pd.DataFrame({"x": [3], "label": ["c"]}),
    ]
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=2,
            input_repartition=RayInputRepartition(target_num_rows_per_block=2),
        ),
    )

    output_dataset = designer.create(config_builder, input_dataset=input_refs).load_dataset()

    assert fake_ray.data.from_arrow_refs_input == input_refs
    assert output_dataset.repartition_calls == [{"num_blocks": None, "target_num_rows_per_block": 2, "shuffle": False}]
    assert [len(block) for block in output_dataset.blocks] == [2, 1]


def test_ray_backend_rejects_input_repartition_without_input_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(input_repartition=RayInputRepartition(num_blocks=2)),
    )

    with pytest.raises(RayBackendConfigurationError, match="input_repartition requires input_dataset"):
        designer.create(stub_sampler_only_config_builder, num_records=4)

    assert fake_ray.data.range_kwargs is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"override_num_blocks": 0}, "positive integer"),
        ({"target_block_size": 10, "min_blocks": 4, "max_blocks": 2}, "min_blocks"),
        ({"min_blocks": 2}, "require target_block_size"),
    ],
)
def test_ray_backend_validates_invalid_block_planning_options(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayBackend(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"output": "table"}, "output"),
        ({"object_ref_format": "json"}, "object_ref_format"),
        ({"order_column": ""}, "order_column"),
        ({"max_trace_events": -1}, "max_trace_events"),
        ({"max_worker_profiles": -1}, "max_worker_profiles"),
        ({"max_throttle_snapshots": -1}, "max_throttle_snapshots"),
    ],
)
def test_ray_backend_constructor_uses_configuration_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayBackend(**kwargs)


def test_ray_backend_rejects_invalid_input_repartition_object() -> None:
    with pytest.raises(RayBackendConfigurationError, match="input_repartition"):
        RayBackend(input_repartition=object())


def test_ray_backend_defaults_bound_observability_storage() -> None:
    backend = RayBackend()

    assert backend.profile_workers is True
    assert backend.max_trace_events == 1000
    assert backend.max_worker_profiles == 1000
    assert backend.max_throttle_snapshots == 1000


@pytest.mark.parametrize("batch_size", [True, "10", 0, -1])
def test_ray_backend_rejects_invalid_batch_size(batch_size: Any) -> None:
    with pytest.raises(RayBackendConfigurationError, match="batch_size"):
        RayBackend(batch_size=batch_size)


def test_ray_backend_batch_size_none_uses_run_config_buffer_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=None),
    )
    designer.set_run_config(RunConfig(buffer_size=7))

    designer.create(_input_expression_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_kwargs["batch_size"] == 7


def test_ray_backend_threads_observability_limits_to_collector_and_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch, with_remote=True)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=1,
            max_trace_events=7,
            max_worker_profiles=8,
            max_throttle_snapshots=9,
        ),
    )

    designer.create(_input_expression_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    constructor_kwargs = input_dataset.map_batches_kwargs["fn_constructor_kwargs"]
    observability_options = constructor_kwargs["observability_options"]
    metrics_collector = constructor_kwargs["metrics_collector"]
    assert observability_options.max_worker_profiles == 8
    assert observability_options.max_throttle_snapshots == 9
    assert metrics_collector._actor._max_trace_events == 7
    assert metrics_collector._actor._max_worker_profiles == 8
    assert metrics_collector._actor._max_throttle_snapshots == 9


def test_ray_driver_planner_captures_from_scratch_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    backend = RayBackend(batch_size=2, output="arrow_refs", override_num_blocks=3)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )

    plan = ray_backend_module.RayDriverPlanner(backend=backend, ray=fake_ray).plan(
        runtime_context=designer._create_backend_runtime_context(),
        config_builder=stub_sampler_only_config_builder,
        num_records=5,
    )

    assert isinstance(plan, ray_backend_module.RayJobPlan)
    assert plan.dataset_source.kind == "range"
    assert plan.dataset_source.external_input_dataset is False
    assert plan.dataset_source.use_input_dataset is False
    assert plan.block_plan is not None
    assert plan.block_plan.planned_blocks == 3
    assert fake_ray.data.range_kwargs == {"override_num_blocks": 3}
    assert plan.worker_payload is plan.map_batches_kwargs["fn_constructor_kwargs"]["execution_payload"]
    assert plan.map_batches_kwargs["batch_size"] == 2
    assert plan.output == "arrow_refs"
    assert plan.input_blocks == 3
    assert plan.metrics_collector is None
    assert plan.throttle_manager is None


def test_ray_driver_planner_captures_preserve_order_input_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [lazy.pd.DataFrame({"x": [2], "label": ["b"]}), lazy.pd.DataFrame({"x": [1], "label": ["a"]})]
    )
    backend = RayBackend(preserve_order=True)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )

    plan = ray_backend_module.RayDriverPlanner(backend=backend, ray=fake_ray).plan(
        runtime_context=designer._create_backend_runtime_context(),
        config_builder=_input_expression_config_builder(stub_model_configs),
        num_records=2,
        input_dataset=input_dataset,
    )

    assert plan.dataset_source.kind == "input_dataset"
    assert plan.dataset_source.external_input_dataset is True
    assert plan.dataset_source.use_input_dataset is True
    assert plan.ordering_mode.hidden_order_column == ray_backend_module._RAY_INTERNAL_ROW_ID_COLUMN
    assert plan.worker_payload.hidden_order_column == ray_backend_module._RAY_INTERNAL_ROW_ID_COLUMN
    assert ray_backend_module._RAY_INTERNAL_ROW_ID_COLUMN in plan.dataset_source.dataset.to_pandas().columns
    assert plan.block_plan is None


def test_ray_driver_planner_applies_input_repartition_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}), lazy.pd.DataFrame({"x": [3], "label": ["c"]})]
    )
    backend = RayBackend(input_repartition=RayInputRepartition(num_blocks=3))
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )

    plan = ray_backend_module.RayDriverPlanner(backend=backend, ray=fake_ray).plan(
        runtime_context=designer._create_backend_runtime_context(),
        config_builder=_input_expression_config_builder(stub_model_configs),
        num_records=3,
        input_dataset=input_dataset,
    )

    assert plan.dataset_source.kind == "input_dataset"
    assert plan.dataset_source.external_input_dataset is True
    assert plan.dataset_source.dataset.repartition_calls == [
        {"num_blocks": 3, "target_num_rows_per_block": None, "shuffle": False}
    ]
    assert plan.input_blocks == 3
    assert plan.block_plan is None


def test_ray_driver_planner_captures_driver_materialized_seed_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_dataframe = lazy.pd.DataFrame({"seed_value": ["alpha"]})
    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame({"seed_value": ["unused"]})))
    config_builder.add_column(ExpressionColumnConfig(name="value_copy", expr="{{ seed_value }}"))
    backend = RayBackend(batch_size=1)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=backend,
    )

    monkeypatch.setattr(
        ray_seed_planning,
        "plan_seed_execution",
        lambda **_: ray_seed_planning.RaySeedPlan(input_dataframe=input_dataframe),
    )

    plan = ray_backend_module.RayDriverPlanner(backend=backend, ray=fake_ray).plan(
        runtime_context=designer._create_backend_runtime_context(),
        config_builder=config_builder,
        num_records=1,
    )

    assert plan.dataset_source.kind == "driver_materialized_seed"
    assert plan.dataset_source.external_input_dataset is False
    assert plan.dataset_source.use_input_dataset is True
    assert fake_ray.data.from_pandas_input is input_dataframe
    assert plan.worker_payload.use_input_dataset is True
    assert plan.block_plan is None


def test_ray_backend_rejects_grouped_block_planning_conflicts() -> None:
    with pytest.raises(RayBackendConfigurationError, match="block_planning"):
        RayBackend(block_planning=RayBlockPlanning(), override_num_blocks=1)


def test_ray_backend_rejects_grouped_execution_option_conflicts() -> None:
    with pytest.raises(RayBackendConfigurationError, match="execution_options"):
        RayBackend(execution_options=RayExecutionOptions(), num_cpus=1)


def test_ray_backend_constructor_signature_keeps_option_objects_primary() -> None:
    parameters = inspect.signature(RayBackend).parameters

    assert "block_planning" in parameters
    assert "execution_options" in parameters
    assert "batch_size" in parameters
    assert "output" in parameters
    assert "auto_init" in parameters
    assert "override_num_blocks" not in parameters
    assert "num_cpus" not in parameters


def test_ray_backend_accepts_legacy_option_kwargs_as_compatibility_shims() -> None:
    backend = RayBackend(
        override_num_blocks=3,
        read_concurrency=2,
        num_cpus=0.5,
        map_concurrency=4,
        ray_remote_args={"resources": {"token_bucket": 1}},
    )

    assert backend.block_planning == RayBlockPlanning(override_num_blocks=3, read_concurrency=2)
    assert backend.execution_options == RayExecutionOptions(
        num_cpus=0.5,
        concurrency=4,
        ray_remote_args={"resources": {"token_bucket": 1}},
    )
    assert backend.ray_remote_args == {"resources": {"token_bucket": 1}}


def test_ray_backend_rejects_unknown_legacy_option_kwargs() -> None:
    with pytest.raises(RayBackendConfigurationError, match="unsupported Ray option arguments: unknown"):
        RayBackend(unknown=1)


def test_ray_backend_propagates_execution_resource_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=1,
            num_cpus=0.5,
            num_gpus=1,
            memory=1024,
            resources={"model_client": 1},
            scheduling_strategy="SPREAD",
            map_concurrency=2,
        ),
    )

    designer.create(_input_expression_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_kwargs["num_cpus"] == 0.5
    assert input_dataset.map_batches_kwargs["num_gpus"] == 1
    assert input_dataset.map_batches_kwargs["memory"] == 1024
    assert input_dataset.map_batches_kwargs["resources"] == {"model_client": 1}
    assert input_dataset.map_batches_kwargs["scheduling_strategy"] == "SPREAD"
    assert input_dataset.map_batches_kwargs["concurrency"] == 2


def test_ray_backend_actor_pool_uses_autoscaling_strategy_and_constructor_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(
            batch_size=1,
            use_actor_pool=True,
            actor_pool_min_size=1,
            actor_pool_max_size=4,
            actor_pool_initial_size=2,
            scheduling_strategy="DEFAULT",
        ),
    )

    designer.create(_input_expression_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is not None
    assert input_dataset.map_batches_fn is ray_backend_module._RayBatchWorker
    assert "fn_kwargs" not in input_dataset.map_batches_kwargs
    assert "fn_constructor_kwargs" in input_dataset.map_batches_kwargs
    assert "concurrency" not in input_dataset.map_batches_kwargs
    assert input_dataset.map_batches_kwargs["scheduling_strategy"] == "DEFAULT"
    assert input_dataset.map_batches_kwargs["compute"].kwargs == {
        "min_size": 1,
        "max_size": 4,
        "initial_size": 2,
    }


def test_ray_backend_rejects_actor_pool_with_map_concurrency() -> None:
    with pytest.raises(RayBackendConfigurationError, match="use_actor_pool"):
        RayBackend(use_actor_pool=True, map_concurrency=2)


def test_ray_backend_driver_model_health_check_runs_once_and_workers_skip_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"id": [0]})])
    preflight_aliases: list[list[str]] = []
    worker_skip_flags: list[list[bool]] = []

    def run_driver_model_health_check(_: Any, __: DataDesignerConfigBuilder, model_aliases: list[str]) -> None:
        preflight_aliases.append(model_aliases)

    def generate_batch(batch: lazy.pd.DataFrame, *, worker: Any, **_: Any) -> Any:
        worker_skip_flags.append([model_config.skip_health_check for model_config in worker._base_config.model_configs])
        return batch

    monkeypatch.setattr(ray_backend_module, "_run_driver_model_health_check", run_driver_model_health_check)
    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    config_builder = _llm_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    designer.create(config_builder, input_dataset=input_dataset)

    assert preflight_aliases == [["stub-model"]]
    assert worker_skip_flags == [[True]]
    assert [model_config.skip_health_check for model_config in config_builder.model_configs] == [False]


def test_ray_backend_worker_model_health_checks_can_be_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"id": [0]})])
    worker_skip_flags: list[list[bool]] = []

    def generate_batch(batch: lazy.pd.DataFrame, *, worker: Any, **_: Any) -> Any:
        worker_skip_flags.append([model_config.skip_health_check for model_config in worker._base_config.model_configs])
        return batch

    monkeypatch.setattr(ray_backend_module, "_run_driver_model_health_check", lambda *_: None)
    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, worker_model_health_checks=True),
    )

    designer.create(_llm_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert worker_skip_flags == [[False]]


def test_ray_backend_driver_model_health_check_failure_stops_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"id": [0]})])

    def fail_preflight(_: Any, __: DataDesignerConfigBuilder, ___: list[str]) -> None:
        raise RayDatasetGenerationError("preflight failed")

    monkeypatch.setattr(ray_backend_module, "_run_driver_model_health_check", fail_preflight)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    with pytest.raises(RayDatasetGenerationError, match="preflight failed"):
        designer.create(_llm_config_builder(stub_model_configs), input_dataset=input_dataset)

    assert input_dataset.map_batches_kwargs is None


def test_ray_backend_preserves_input_dataset_order_across_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}),
            lazy.pd.DataFrame({"x": [3], "label": ["c"]}),
            lazy.pd.DataFrame({"x": [4, 5], "label": ["d", "e"]}),
        ]
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
        {"x": 4, "label": "d", "x_label": "4-d"},
        {"x": 5, "label": "e", "x_label": "5-e"},
    ]


def test_ray_backend_can_sort_by_explicit_order_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"row_order": [2], "x": [3], "label": ["c"]}),
            lazy.pd.DataFrame({"row_order": [0, 1], "x": [1, 2], "label": ["a", "b"]}),
        ]
    )
    config_builder = _input_expression_config_builder(stub_model_configs)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, order_column="row_order", drop_order_column=True),
    )

    results = designer.create(config_builder, input_dataset=input_dataset)
    output_df = results.load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
        {"x": 3, "label": "c", "x_label": "3-c"},
    ]


def test_ray_backend_preserve_order_does_not_require_explicit_order_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"x": [1], "label": ["a"]}),
            lazy.pd.DataFrame({"x": [2], "label": ["b"]}),
        ],
        reverse_mapped_blocks=True,
    )
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(preserve_order=True),
    )
    config_builder = _input_expression_config_builder(stub_model_configs)

    output_df = designer.create(config_builder, input_dataset=input_dataset).load_dataset().to_pandas()

    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_pandas_object_refs_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_refs = [lazy.pd.DataFrame({"x": [1], "label": ["a"]}), lazy.pd.DataFrame({"x": [2], "label": ["b"]})]
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, object_ref_format="pandas"),
    )

    results = designer.create(config_builder, input_dataset=input_refs)
    output_df = results.load_dataset().to_pandas()

    assert fake_ray.data.from_pandas_refs_input == input_refs
    assert fake_ray.data.from_arrow_refs_input is None
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_uses_arrow_object_refs_as_in_memory_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    input_refs = [
        lazy.pa.table({"x": [1], "label": ["a"]}),
        lazy.pa.table({"x": [2], "label": ["b"]}),
    ]
    config_builder = _input_expression_config_builder(stub_model_configs)

    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=1, object_ref_format="arrow"),
    )

    results = designer.create(config_builder, input_dataset=input_refs)
    output_df = results.load_dataset().to_pandas()

    assert fake_ray.data.from_arrow_refs_input == input_refs
    assert fake_ray.data.from_pandas_refs_input is None
    assert output_df.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]


def test_ray_backend_rejects_invalid_input_dataset_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(),
    )

    with pytest.raises(RayBackendConfigurationError, match="ray.data.Dataset or a sequence of Ray ObjectRefs"):
        designer.create(stub_sampler_only_config_builder, input_dataset=object())


def test_ray_backend_can_return_arrow_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="arrow_refs"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output == ["arrow-ref-0"]
    assert results.to_arrow_refs() == ["arrow-ref-0"]
    assert isinstance(results.load_dataset(), FakeRayDataset)
    assert results.load_analysis() is None
    assert results.load_metrics() == RayDatasetMetrics(
        total_rows=2, blocks=1, elapsed_seconds=results.metrics.elapsed_seconds
    )


def test_ray_backend_arrow_refs_load_dataset_uses_materialized_refs_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    generate_calls = 0

    def generate_batch(batch: lazy.pd.DataFrame, **_: Any) -> lazy.pd.DataFrame:
        nonlocal generate_calls
        generate_calls += 1
        output = batch.copy()
        output["generated"] = generate_calls
        return output

    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="arrow_refs"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output == ["arrow-ref-0"]
    assert generate_calls == 1

    dataset = results.load_dataset()
    output_df = dataset.to_pandas()

    assert fake_ray.data.from_arrow_refs_input == ["arrow-ref-0"]
    assert output_df.to_dict(orient="records") == [
        {"id": 0, "generated": 1},
        {"id": 1, "generated": 1},
    ]
    assert generate_calls == 1


def test_ray_backend_arrow_refs_dataset_rebuild_falls_back_to_items(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ray = install_fake_ray(monkeypatch)

    def fail_from_arrow_refs(refs: list[Any]) -> FakeRayDataset:
        del refs
        raise ValueError("nested schema unification failed")

    fake_ray.data.from_arrow_refs = fail_from_arrow_refs
    refs = [
        lazy.pa.table({"payload": [{"a": 1}]}),
        lazy.pa.table({"payload": [{"b": "two"}]}),
    ]

    dataset = ray_backend_module._dataset_from_arrow_refs(fake_ray, refs)

    assert fake_ray.data.from_items_input == [{"payload": {"a": 1}}, {"payload": {"b": "two"}}]
    assert dataset.to_pandas().to_dict(orient="records") == [
        {"payload": {"a": 1}},
        {"payload": {"b": "two"}},
    ]


def test_ray_backend_hydrates_filesystem_seed_reader_fanout_with_ray_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "a.txt").write_text("alpha\nbeta", encoding="utf-8")
    (seed_dir / "b.txt").write_text("gamma\ndelta", encoding="utf-8")

    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"))
    config_builder.add_column(
        ExpressionColumnConfig(
            name="line_label",
            expr="{{ relative_path }}:{{ line_index }}:{{ line }}",
        )
    )
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        seed_readers=[LineFanoutDirectorySeedReader()],
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=3),
    )

    output_df = designer.create(config_builder, num_records=3).load_dataset().to_pandas()

    assert fake_ray.data.from_pandas_input is None
    assert fake_ray.data.from_items_input == [
        {"relative_path": "a.txt"},
        {"relative_path": "b.txt"},
    ]
    assert fake_ray.data.range_kwargs is None
    assert output_df.to_dict(orient="records") == [
        {"relative_path": "a.txt", "line_index": 0, "line": "alpha", "line_label": "a.txt:0:alpha"},
        {"relative_path": "a.txt", "line_index": 1, "line": "beta", "line_label": "a.txt:1:beta"},
        {"relative_path": "b.txt", "line_index": 0, "line": "gamma", "line_label": "b.txt:0:gamma"},
    ]


def test_ray_backend_reads_local_file_seed_with_ray_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    seed_path = tmp_path / "seed.csv"
    lazy.pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]}).to_csv(seed_path, index=False)

    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    config_builder.add_column(ExpressionColumnConfig(name="value_label", expr="{{ value }}:{{ label }}"))
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    output_df = designer.create(config_builder, num_records=5).load_dataset().to_pandas()

    assert fake_ray.data.read_csv_input == str(seed_path)
    assert fake_ray.data.read_csv_kwargs == {"partitioning": None}
    assert fake_ray.data.from_pandas_input is None
    assert output_df.to_dict(orient="records") == [
        {"value": 1, "label": "a", "value_label": "1:a"},
        {"value": 2, "label": "b", "value_label": "2:b"},
        {"value": 3, "label": "c", "value_label": "3:c"},
        {"value": 1, "label": "a", "value_label": "1:a"},
        {"value": 2, "label": "b", "value_label": "2:b"},
    ]


def test_ray_backend_applies_local_file_seed_range_without_leaking_ordinal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    seed_path = tmp_path / "seed.jsonl"
    lazy.pd.DataFrame({"value": [10, 20, 30, 40], "label": ["a", "b", "c", "d"]}).to_json(
        seed_path,
        orient="records",
        lines=True,
    )

    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(
        LocalFileSeedSource(path=str(seed_path)),
        selection_strategy=IndexRange(start=1, end=2),
    )
    config_builder.add_column(ExpressionColumnConfig(name="value_label", expr="{{ value }}:{{ label }}"))
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    output_df = designer.create(config_builder, num_records=3).load_dataset().to_pandas()

    assert fake_ray.data.read_json_input == str(seed_path)
    assert fake_ray.data.read_json_kwargs == {"lines": True, "partitioning": None}
    assert fake_ray.data.from_pandas_input is None
    assert "__data_designer_ray_seed_ordinal" not in output_df.columns
    assert output_df.to_dict(orient="records") == [
        {"value": 20, "label": "b", "value_label": "20:b"},
        {"value": 30, "label": "c", "value_label": "30:c"},
        {"value": 20, "label": "b", "value_label": "20:b"},
    ]


def test_ray_backend_rejects_shuffled_ray_native_local_file_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    seed_path = tmp_path / "seed.csv"
    lazy.pd.DataFrame({"value": [1, 2]}).to_csv(seed_path, index=False)
    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(
        LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )
    config_builder.add_column(ExpressionColumnConfig(name="value_copy", expr="{{ value }}"))
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    with pytest.raises(RayBackendConfigurationError, match="local-file seed ingestion.*ordered sampling"):
        designer.create(config_builder, num_records=1)

    assert fake_ray.data.read_csv_input is None
    assert fake_ray.data.from_pandas_input is None


def test_ray_backend_reads_file_contents_seed_with_ray_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_providers: Any,
) -> None:
    fake_ray = install_fake_ray(monkeypatch)
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "b.txt").write_text("beta", encoding="utf-8")

    config_builder = DataDesignerConfigBuilder(model_configs=[])
    config_builder.with_seed_dataset(FileContentsSeedSource(path=str(seed_dir), file_pattern="*.txt"))
    config_builder.add_column(ExpressionColumnConfig(name="content_label", expr="{{ file_name }}:{{ content }}"))
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    output_df = designer.create(config_builder, num_records=3).load_dataset().to_pandas()

    assert fake_ray.data.read_binary_files_input == [
        str(seed_dir / "a.txt"),
        str(seed_dir / "b.txt"),
    ]
    assert fake_ray.data.read_binary_files_kwargs == {"include_paths": True, "partitioning": None}
    assert fake_ray.data.from_pandas_input is None
    assert output_df.to_dict(orient="records") == [
        {
            "source_kind": "file_contents",
            "source_path": str(seed_dir / "a.txt"),
            "relative_path": "a.txt",
            "file_name": "a.txt",
            "content": "alpha",
            "content_label": "a.txt:alpha",
        },
        {
            "source_kind": "file_contents",
            "source_path": str(seed_dir / "b.txt"),
            "relative_path": "b.txt",
            "file_name": "b.txt",
            "content": "beta",
            "content_label": "b.txt:beta",
        },
        {
            "source_kind": "file_contents",
            "source_path": str(seed_dir / "a.txt"),
            "relative_path": "a.txt",
            "file_name": "a.txt",
            "content": "alpha",
            "content_label": "a.txt:alpha",
        },
    ]


def test_ray_results_to_arrow_refs_wraps_materialization_failures(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    class FailingArrowDataset:
        def to_arrow_refs(self) -> list[Any]:
            raise RuntimeError("object store unavailable")

    results = RayDatasetCreationResults(
        dataset=FailingArrowDataset(),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(),
    )

    with pytest.raises(RayDatasetGenerationError, match="materialize Arrow ObjectRefs") as exc_info:
        results.to_arrow_refs()

    assert "object store unavailable" in str(exc_info.value.__cause__)


def test_ray_backend_dataset_output_wraps_dataset_and_delegates_arrow_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2, output="dataset"),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=2)

    assert results.output is results.load_dataset()
    assert isinstance(results.output, FakeRayDataset)
    assert results.to_arrow_refs() == ["arrow-ref-0"]
    assert results.load_analysis() is None
    assert results.load_metrics().total_rows == 2
    assert results.load_metrics().blocks == 1


def test_ray_backend_rejects_unsafe_processors_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    config_builder = _input_expression_config_builder(stub_model_configs)
    config_builder.add_processor(
        SchemaTransformProcessorConfig(
            name="schema-transform",
            template={"combined": "{{ x_label }}"},
        )
    )
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(),
    )

    with pytest.raises(RayBackendConfigurationError, match="Unsupported processor"):
        designer.create(config_builder, input_dataset=FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})]))


def test_ray_backend_import_is_lazy_when_ray_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = ray_backend_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray":
            raise ImportError
        return real_import_module(name)

    monkeypatch.setattr(ray_backend_module.importlib, "import_module", import_module)
    backend = RayBackend()

    with pytest.raises(ImportError, match="data-designer\\[ray\\]"):
        backend.create(
            runtime_context=object(),
            config_builder=DataDesignerConfigBuilder(model_configs=[]),
            num_records=1,
            dataset_name="dataset",
        )


def test_local_backend_ignores_missing_ray_when_no_backend_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    real_import_module = ray_backend_module.importlib.import_module

    def import_module(name: str) -> Any:
        if name == "ray":
            raise AssertionError("local DataDesigner.create() should not import ray")
        return real_import_module(name)

    monkeypatch.setattr(ray_backend_module.importlib, "import_module", import_module)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    results = designer.create(stub_sampler_only_config_builder, num_records=1)

    assert len(results.load_dataset()) == 1
    assert results.load_analysis() is not None


def test_input_dataset_requires_backend(
    tmp_path: Path,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path),
    )

    with pytest.raises(ValueError, match="input_dataset requires an execution backend"):
        designer.create(stub_sampler_only_config_builder, input_dataset=object())


def test_same_process_local_ray_local_runs_do_not_share_backend_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_model_configs: Any,
    stub_sampler_only_config_builder: Any,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch)
    input_config_builder = _input_expression_config_builder(stub_model_configs)
    input_dataset = FakeRayDataset([lazy.pd.DataFrame({"x": [1], "label": ["a"]})])
    local_before = DataDesigner(
        artifact_path=tmp_path / "local-before",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-before"),
    )
    ray_designer = DataDesigner(
        artifact_path=tmp_path / "ray",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path / "ray"),
        backend=RayBackend(batch_size=1),
    )
    local_after = DataDesigner(
        artifact_path=tmp_path / "local-after",
        model_providers=stub_model_providers,
        managed_assets_path=_managed_assets_path(tmp_path / "local-after"),
    )

    assert len(local_before.create(stub_sampler_only_config_builder, num_records=1).load_dataset()) == 1
    ray_records = ray_designer.create(input_config_builder, input_dataset=input_dataset).load_dataset().to_pandas()
    assert len(local_after.create(stub_sampler_only_config_builder, num_records=1).load_dataset()) == 1

    assert ray_records.to_dict(orient="records") == [{"x": 1, "label": "a", "x_label": "1-a"}]
    assert input_config_builder.get_seed_config() is None


def test_seed_readers_are_cloned_without_attachment_state() -> None:
    reader = DataFrameSeedReader()
    reader.attach(
        DataFrameSeedSource(df=lazy.pd.DataFrame({"x": [1]})),
        PlaintextResolver(),
    )
    assert reader.get_seed_dataset_size() == 1
    assert getattr(reader, "_duckdb_conn") is not None

    clones = ray_seed_planning.clone_seed_readers_for_worker([reader])

    assert len(clones) == 1
    assert clones[0] is not reader
    assert getattr(clones[0], "_duckdb_conn") is None
    assert not hasattr(clones[0], "source")
    assert not hasattr(clones[0], "secret_resolver")
