# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from data_designer.integrations.ray import (
    RayBackendConfigurationError,
    RayBlockPlanning,
    RayDataCheckpointConfig,
    RayDataContextOptions,
    RayExecutionOptions,
    RayExecutionResources,
    RayInputRepartition,
)
from data_designer.integrations.ray.options import resolve_ray_backend_options

pytestmark = pytest.mark.ray_fake


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"override_num_blocks": 0}, "positive integer"),
        ({"target_block_size": 10, "min_blocks": 4, "max_blocks": 2}, "min_blocks"),
        ({"min_blocks": 2}, "require target_block_size"),
    ],
)
def test_ray_block_planning_validation_uses_configuration_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayBlockPlanning(**kwargs)


def test_ray_backend_option_resolution_rejects_block_planning_conflicts() -> None:
    with pytest.raises(RayBackendConfigurationError, match="block_planning"):
        resolve_ray_backend_options(
            block_planning=RayBlockPlanning(),
            legacy_options={"override_num_blocks": 1},
        )


@pytest.mark.parametrize(
    "reserved_key",
    [
        "batch_format",
        "batch_size",
        "fn",
        "fn_args",
        "fn_constructor_args",
        "fn_constructor_kwargs",
        "fn_kwargs",
        "zero_copy_batch",
    ],
)
def test_ray_execution_options_rejects_data_designer_owned_map_batches_keys(reserved_key: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=reserved_key):
        RayExecutionOptions(ray_remote_args={reserved_key: object()})


def test_ray_execution_options_rejects_duplicate_explicit_remote_args() -> None:
    with pytest.raises(RayBackendConfigurationError, match="num_cpus"):
        RayExecutionOptions(num_cpus=0.5, ray_remote_args={"num_cpus": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ray_execution_options_rejects_non_finite_numeric_options(value: float) -> None:
    with pytest.raises(RayBackendConfigurationError, match="num_cpus.*finite"):
        RayExecutionOptions(num_cpus=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ray_execution_options_rejects_non_finite_resources(value: float) -> None:
    with pytest.raises(RayBackendConfigurationError, match=r"resources\['gpu_slice'\].*finite"):
        RayExecutionOptions(resources={"gpu_slice": value})


def test_ray_execution_options_rejects_actor_pool_with_concurrency() -> None:
    with pytest.raises(RayBackendConfigurationError, match="use_actor_pool"):
        RayExecutionOptions(use_actor_pool=True, concurrency=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_cpus": True}, "num_cpus"),
        ({"memory": 0}, "memory"),
        ({"resources": {"": 1}}, "resource names"),
        ({"concurrency": (3, 1)}, "minimum"),
        ({"actor_pool_min_size": 1}, "use_actor_pool"),
    ],
)
def test_ray_execution_options_validation_uses_configuration_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayExecutionOptions(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resource_limits": object()}, "resource_limits"),
        (
            {
                "resource_limits": RayExecutionResources(cpu=1),
                "exclude_resources": RayExecutionResources(cpu=0.5),
            },
            "cannot both set",
        ),
        ({"preserve_execution_order": "yes"}, "preserve_execution_order"),
        ({"max_errored_blocks": 1.5}, "max_errored_blocks"),
        ({"checkpoint_config": object()}, "checkpoint_config"),
    ],
)
def test_ray_data_context_options_validation_uses_configuration_error(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayDataContextOptions(**kwargs)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RayExecutionResources(cpu=True), "cpu"),
        (lambda: RayExecutionResources(memory=-1), "memory"),
        (lambda: RayDataCheckpointConfig(id_column=""), "id_column"),
        (lambda: RayDataCheckpointConfig(checkpoint_path=""), "checkpoint_path"),
        (lambda: RayDataCheckpointConfig(delete_checkpoint_on_success="no"), "delete_checkpoint_on_success"),
        (lambda: RayDataCheckpointConfig(filter_num_threads=None), "filter_num_threads"),
        (lambda: RayDataCheckpointConfig(filter_num_threads=0), "filter_num_threads"),
        (lambda: RayDataCheckpointConfig(write_num_threads=-1), "write_num_threads"),
    ],
)
def test_ray_data_context_nested_options_validate_values(factory: Any, match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        factory()


def test_ray_data_checkpoint_config_serializes_supported_context_fields() -> None:
    checkpoint_config = RayDataCheckpointConfig(
        id_column="row_id",
        checkpoint_path="s3://bucket/path",
        delete_checkpoint_on_success=False,
        filter_num_threads=2,
        write_num_threads=4,
    )

    assert checkpoint_config.to_context_value() == {
        "id_column": "row_id",
        "checkpoint_path": "s3://bucket/path",
        "delete_checkpoint_on_success": False,
        "filter_num_threads": 2,
        "write_num_threads": 4,
    }


def test_ray_backend_option_resolution_rejects_execution_option_conflicts() -> None:
    with pytest.raises(RayBackendConfigurationError, match="execution_options"):
        resolve_ray_backend_options(
            execution_options=RayExecutionOptions(),
            legacy_options={"num_cpus": 1},
        )


def test_ray_backend_option_resolution_allows_noop_legacy_values_with_option_objects() -> None:
    block_planning = RayBlockPlanning(override_num_blocks=2)
    execution_options = RayExecutionOptions(num_cpus=0.5)
    input_repartition = RayInputRepartition(num_blocks=4)
    data_context_options = RayDataContextOptions(max_errored_blocks=1)

    resolved = resolve_ray_backend_options(
        block_planning=block_planning,
        execution_options=execution_options,
        input_repartition=input_repartition,
        data_context_options=data_context_options,
        legacy_options={
            "override_num_blocks": None,
            "num_cpus": None,
            "use_actor_pool": False,
        },
    )

    assert resolved.block_planning is block_planning
    assert resolved.execution_options is execution_options
    assert resolved.input_repartition is input_repartition
    assert resolved.data_context_options is data_context_options


def test_ray_backend_option_resolution_rejects_unknown_legacy_options() -> None:
    with pytest.raises(RayBackendConfigurationError, match="unsupported Ray option arguments: unknown"):
        resolve_ray_backend_options(legacy_options={"unknown": 1})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_blocks": 0}, "num_blocks.*positive integer"),
        ({"target_num_rows_per_block": 0}, "target_num_rows_per_block.*positive integer"),
        ({"shuffle": "yes"}, "shuffle.*boolean"),
        ({"num_blocks": 2, "target_num_rows_per_block": 100}, "cannot be combined"),
        ({"target_num_rows_per_block": 100, "shuffle": True}, "shuffle can only be used"),
    ],
)
def test_ray_input_repartition_validation_uses_configuration_error(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RayBackendConfigurationError, match=match):
        RayInputRepartition(**kwargs)


def test_ray_input_repartition_resolves_repartition_kwargs() -> None:
    assert RayInputRepartition(num_blocks=3, shuffle=True).to_repartition_kwargs() == {
        "num_blocks": 3,
        "shuffle": True,
    }
    assert RayInputRepartition(target_num_rows_per_block=100).to_repartition_kwargs() == {
        "target_num_rows_per_block": 100,
    }
    assert RayInputRepartition().to_repartition_kwargs() == {}


def test_ray_backend_option_resolution_rejects_invalid_input_repartition_object() -> None:
    with pytest.raises(RayBackendConfigurationError, match="input_repartition"):
        resolve_ray_backend_options(input_repartition=object())


def test_ray_backend_option_resolution_rejects_invalid_data_context_options_object() -> None:
    with pytest.raises(RayBackendConfigurationError, match="data_context_options"):
        resolve_ray_backend_options(data_context_options=object())
