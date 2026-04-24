# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RayMapConcurrency = int | tuple[int, int] | tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RayResolvedBlockPlan:
    """Resolved Ray Data read planning options for from-scratch generation."""

    override_num_blocks: int | None = None
    read_concurrency: int | None = None
    planned_blocks: int | None = None

    def to_range_kwargs(self) -> dict[str, int]:
        """Return keyword arguments accepted by ``ray.data.range``."""
        kwargs: dict[str, int] = {}
        if self.override_num_blocks is not None:
            kwargs["override_num_blocks"] = self.override_num_blocks
        if self.read_concurrency is not None:
            kwargs["concurrency"] = self.read_concurrency
        return kwargs


@dataclass(frozen=True, slots=True)
class RayBlockPlanning:
    """Controls for Ray Data block creation when RayBackend creates a range dataset."""

    override_num_blocks: int | None = None
    target_block_size: int | None = None
    min_blocks: int | None = None
    max_blocks: int | None = None
    read_concurrency: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_positive_int("override_num_blocks", self.override_num_blocks)
        _validate_optional_positive_int("target_block_size", self.target_block_size)
        _validate_optional_positive_int("min_blocks", self.min_blocks)
        _validate_optional_positive_int("max_blocks", self.max_blocks)
        _validate_optional_positive_int("read_concurrency", self.read_concurrency)
        if self.override_num_blocks is not None and any(
            value is not None for value in (self.target_block_size, self.min_blocks, self.max_blocks)
        ):
            raise ValueError(
                "RayBlockPlanning override_num_blocks cannot be combined with target_block_size, "
                "min_blocks, or max_blocks."
            )
        if self.min_blocks is not None and self.max_blocks is not None and self.min_blocks > self.max_blocks:
            raise ValueError("RayBlockPlanning min_blocks must be less than or equal to max_blocks.")
        if self.target_block_size is None and (self.min_blocks is not None or self.max_blocks is not None):
            raise ValueError("RayBlockPlanning min_blocks and max_blocks require target_block_size.")

    @property
    def has_explicit_controls(self) -> bool:
        """Return whether any from-scratch read planning control was configured."""
        return any(
            value is not None
            for value in (
                self.override_num_blocks,
                self.target_block_size,
                self.min_blocks,
                self.max_blocks,
                self.read_concurrency,
            )
        )

    def resolve(self, *, num_records: int) -> RayResolvedBlockPlan:
        """Resolve configured planning controls for a concrete record count."""
        if num_records < 0:
            raise ValueError("RayBlockPlanning num_records must be non-negative.")
        if self.override_num_blocks is not None:
            return RayResolvedBlockPlan(
                override_num_blocks=self.override_num_blocks,
                read_concurrency=self.read_concurrency,
                planned_blocks=self.override_num_blocks,
            )
        if self.target_block_size is None:
            return RayResolvedBlockPlan(read_concurrency=self.read_concurrency)

        planned_blocks = max(math.ceil(max(num_records, 1) / self.target_block_size), 1)
        if self.min_blocks is not None:
            planned_blocks = max(planned_blocks, self.min_blocks)
        if self.max_blocks is not None:
            planned_blocks = min(planned_blocks, self.max_blocks)
        if num_records > 0:
            planned_blocks = min(planned_blocks, num_records)
        return RayResolvedBlockPlan(
            override_num_blocks=planned_blocks,
            read_concurrency=self.read_concurrency,
            planned_blocks=planned_blocks,
        )


@dataclass(frozen=True, slots=True)
class RayExecutionOptions:
    """Validated Ray Data ``map_batches`` execution options."""

    num_cpus: float | None = None
    num_gpus: float | None = None
    memory: float | None = None
    resources: Mapping[str, float] | None = None
    scheduling_strategy: Any | None = None
    compute: Any | None = None
    concurrency: RayMapConcurrency | None = None
    ray_remote_args_fn: Any | None = None
    ray_remote_args: Mapping[str, Any] | None = None
    use_actor_pool: bool = False
    actor_pool_min_size: int | None = None
    actor_pool_max_size: int | None = None
    actor_pool_initial_size: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_non_negative_number("num_cpus", self.num_cpus)
        _validate_optional_non_negative_number("num_gpus", self.num_gpus)
        _validate_optional_positive_number("memory", self.memory)
        _validate_resources(self.resources)
        if isinstance(self.scheduling_strategy, str) and not self.scheduling_strategy:
            raise ValueError("RayExecutionOptions scheduling_strategy must be non-empty when provided.")
        if self.compute is not None and self.use_actor_pool:
            raise ValueError("RayExecutionOptions compute cannot be combined with use_actor_pool=True.")
        if self.compute is not None and self.concurrency is not None:
            raise ValueError("RayExecutionOptions compute cannot be combined with concurrency.")
        _validate_map_concurrency(self.concurrency)
        _validate_optional_positive_int("actor_pool_min_size", self.actor_pool_min_size)
        _validate_optional_positive_int("actor_pool_max_size", self.actor_pool_max_size)
        _validate_optional_positive_int("actor_pool_initial_size", self.actor_pool_initial_size)
        if self.actor_pool_min_size is not None and not self.use_actor_pool:
            raise ValueError("RayExecutionOptions actor_pool_min_size requires use_actor_pool=True.")
        if self.actor_pool_max_size is not None and not self.use_actor_pool:
            raise ValueError("RayExecutionOptions actor_pool_max_size requires use_actor_pool=True.")
        if self.actor_pool_initial_size is not None and not self.use_actor_pool:
            raise ValueError("RayExecutionOptions actor_pool_initial_size requires use_actor_pool=True.")
        if (
            self.actor_pool_min_size is not None
            and self.actor_pool_max_size is not None
            and self.actor_pool_min_size > self.actor_pool_max_size
        ):
            raise ValueError("RayExecutionOptions actor_pool_min_size must be less than or equal to max_size.")
        if self.actor_pool_initial_size is not None:
            if self.actor_pool_min_size is not None and self.actor_pool_initial_size < self.actor_pool_min_size:
                raise ValueError("RayExecutionOptions actor_pool_initial_size must be at least actor_pool_min_size.")
            if self.actor_pool_max_size is not None and self.actor_pool_initial_size > self.actor_pool_max_size:
                raise ValueError("RayExecutionOptions actor_pool_initial_size must be at most actor_pool_max_size.")
        _validate_remote_arg_conflicts(self)

    def to_map_batches_kwargs(self, ray: Any | None = None) -> dict[str, Any]:
        """Return keyword arguments accepted by ``Dataset.map_batches``."""
        kwargs: dict[str, Any] = {}
        _set_if_not_none(kwargs, "num_cpus", self.num_cpus)
        _set_if_not_none(kwargs, "num_gpus", self.num_gpus)
        _set_if_not_none(kwargs, "memory", self.memory)
        if self.resources is not None:
            kwargs["resources"] = dict(self.resources)
        _set_if_not_none(kwargs, "scheduling_strategy", self.scheduling_strategy)
        _set_if_not_none(kwargs, "compute", self.compute)
        _set_if_not_none(kwargs, "concurrency", self.concurrency)
        _set_if_not_none(kwargs, "ray_remote_args_fn", self.ray_remote_args_fn)
        if self.ray_remote_args is not None:
            kwargs.update(dict(self.ray_remote_args))
        if self.use_actor_pool:
            if ray is None:
                raise ValueError("RayExecutionOptions use_actor_pool=True requires an imported ray module.")
            kwargs["compute"] = _create_actor_pool_strategy(ray, self)
        return kwargs


def _validate_optional_positive_int(field_name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Ray option {field_name} must be a positive integer.")


def _validate_optional_non_negative_number(field_name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Ray option {field_name} must be a non-negative number.")


def _validate_optional_positive_number(field_name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Ray option {field_name} must be a positive number.")


def _validate_resources(resources: Mapping[str, float] | None) -> None:
    if resources is None:
        return
    for name, value in resources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("RayExecutionOptions resource names must be non-empty strings.")
        _validate_optional_positive_number(f"resources[{name!r}]", value)


def _validate_map_concurrency(value: RayMapConcurrency | None) -> None:
    if value is None:
        return
    if isinstance(value, int):
        _validate_optional_positive_int("concurrency", value)
        return
    if not isinstance(value, tuple) or len(value) not in (2, 3):
        raise ValueError("RayExecutionOptions concurrency must be a positive integer or a 2/3-item tuple.")
    for index, item in enumerate(value):
        _validate_optional_positive_int(f"concurrency[{index}]", item)
    if value[0] > value[1]:
        raise ValueError("RayExecutionOptions concurrency minimum must be less than or equal to maximum.")
    if len(value) == 3 and not value[0] <= value[2] <= value[1]:
        raise ValueError("RayExecutionOptions concurrency initial size must be within the min/max range.")


def _validate_remote_arg_conflicts(options: RayExecutionOptions) -> None:
    if options.ray_remote_args is None:
        return
    explicit_fields = {
        "num_cpus": options.num_cpus,
        "num_gpus": options.num_gpus,
        "memory": options.memory,
        "resources": options.resources,
        "scheduling_strategy": options.scheduling_strategy,
        "compute": options.compute if not options.use_actor_pool else True,
        "concurrency": options.concurrency,
        "ray_remote_args_fn": options.ray_remote_args_fn,
    }
    conflicts = sorted(
        name for name, value in explicit_fields.items() if value is not None and name in options.ray_remote_args
    )
    if conflicts:
        conflict_list = ", ".join(conflicts)
        raise ValueError(
            f"RayExecutionOptions received duplicate explicit and ray_remote_args values: {conflict_list}."
        )


def _create_actor_pool_strategy(ray: Any, options: RayExecutionOptions) -> Any:
    actor_pool_strategy = getattr(ray.data, "ActorPoolStrategy", None)
    if not callable(actor_pool_strategy):
        raise ValueError("RayExecutionOptions use_actor_pool=True requires ray.data.ActorPoolStrategy.")
    kwargs: dict[str, int] = {}
    _set_if_not_none(kwargs, "min_size", options.actor_pool_min_size or 1)
    _set_if_not_none(kwargs, "max_size", options.actor_pool_max_size)
    _set_if_not_none(kwargs, "initial_size", options.actor_pool_initial_size)
    return actor_pool_strategy(**kwargs)


def _set_if_not_none(target: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        target[key] = value
