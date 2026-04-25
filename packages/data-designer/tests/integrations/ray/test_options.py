# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from data_designer.integrations.ray import RayBackendConfigurationError, RayBlockPlanning, RayExecutionOptions

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
