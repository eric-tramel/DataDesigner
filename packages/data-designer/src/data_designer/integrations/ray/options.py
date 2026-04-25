# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

from data_designer.integrations.ray._validation import validate_finite_number
from data_designer.integrations.ray.errors import RayBackendConfigurationError

RayMapConcurrency = int | tuple[int, int] | tuple[int, int, int]
_DATA_DESIGNER_OWNED_MAP_BATCHES_KEYS = frozenset(
    {
        "batch_format",
        "batch_size",
        "fn",
        "fn_args",
        "fn_constructor_args",
        "fn_constructor_kwargs",
        "fn_kwargs",
        "udf_modifying_row_count",
        "zero_copy_batch",
    }
)
_RAY_BACKEND_BLOCK_PLANNING_LEGACY_KWARGS = frozenset(
    {
        "override_num_blocks",
        "target_block_size",
        "min_blocks",
        "max_blocks",
        "read_concurrency",
    }
)
_RAY_BACKEND_EXECUTION_OPTIONS_LEGACY_KWARGS = frozenset(
    {
        "ray_remote_args",
        "num_cpus",
        "num_gpus",
        "memory",
        "resources",
        "scheduling_strategy",
        "compute",
        "map_concurrency",
        "ray_remote_args_fn",
        "use_actor_pool",
        "actor_pool_min_size",
        "actor_pool_max_size",
        "actor_pool_initial_size",
    }
)
_LOCAL_PROVIDER_TYPES = frozenset({"local", "local-openai", "ollama", "vllm"})
_LOCAL_PROVIDER_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


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
            raise RayBackendConfigurationError(
                "RayBlockPlanning override_num_blocks cannot be combined with target_block_size, "
                "min_blocks, or max_blocks."
            )
        if self.min_blocks is not None and self.max_blocks is not None and self.min_blocks > self.max_blocks:
            raise RayBackendConfigurationError("RayBlockPlanning min_blocks must be less than or equal to max_blocks.")
        if self.target_block_size is None and (self.min_blocks is not None or self.max_blocks is not None):
            raise RayBackendConfigurationError("RayBlockPlanning min_blocks and max_blocks require target_block_size.")

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
            raise RayBackendConfigurationError("RayBlockPlanning num_records must be non-negative.")
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
    use_actor_pool: bool | None = None
    actor_pool_min_size: int | None = None
    actor_pool_max_size: int | None = None
    actor_pool_initial_size: int | None = None
    _default_actor_pool_min_size: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_optional_non_negative_number("num_cpus", self.num_cpus)
        _validate_optional_non_negative_number("num_gpus", self.num_gpus)
        _validate_optional_positive_number("memory", self.memory)
        _validate_resources(self.resources)
        if isinstance(self.scheduling_strategy, str) and not self.scheduling_strategy:
            raise RayBackendConfigurationError(
                "RayExecutionOptions scheduling_strategy must be non-empty when provided."
            )
        if self.use_actor_pool is not None and not isinstance(self.use_actor_pool, bool):
            raise RayBackendConfigurationError("RayExecutionOptions use_actor_pool must be a boolean or None.")
        if self.compute is not None and self.use_actor_pool:
            raise RayBackendConfigurationError(
                "RayExecutionOptions compute cannot be combined with use_actor_pool=True."
            )
        if self.compute is not None and self.concurrency is not None:
            raise RayBackendConfigurationError("RayExecutionOptions compute cannot be combined with concurrency.")
        if self.use_actor_pool and self.concurrency is not None:
            raise RayBackendConfigurationError(
                "RayExecutionOptions use_actor_pool=True cannot be combined with concurrency."
            )
        _validate_map_concurrency(self.concurrency)
        _validate_optional_positive_int("actor_pool_min_size", self.actor_pool_min_size)
        _validate_optional_positive_int("actor_pool_max_size", self.actor_pool_max_size)
        _validate_optional_positive_int("actor_pool_initial_size", self.actor_pool_initial_size)
        if self.actor_pool_min_size is not None and self.use_actor_pool is not True:
            raise RayBackendConfigurationError("RayExecutionOptions actor_pool_min_size requires use_actor_pool=True.")
        if self.actor_pool_max_size is not None and self.use_actor_pool is not True:
            raise RayBackendConfigurationError("RayExecutionOptions actor_pool_max_size requires use_actor_pool=True.")
        if self.actor_pool_initial_size is not None and self.use_actor_pool is not True:
            raise RayBackendConfigurationError(
                "RayExecutionOptions actor_pool_initial_size requires use_actor_pool=True."
            )
        if (
            self.actor_pool_min_size is not None
            and self.actor_pool_max_size is not None
            and self.actor_pool_min_size > self.actor_pool_max_size
        ):
            raise RayBackendConfigurationError(
                "RayExecutionOptions actor_pool_min_size must be less than or equal to max_size."
            )
        if self.actor_pool_initial_size is not None:
            if self.actor_pool_min_size is not None and self.actor_pool_initial_size < self.actor_pool_min_size:
                raise RayBackendConfigurationError(
                    "RayExecutionOptions actor_pool_initial_size must be at least actor_pool_min_size."
                )
            if self.actor_pool_max_size is not None and self.actor_pool_initial_size > self.actor_pool_max_size:
                raise RayBackendConfigurationError(
                    "RayExecutionOptions actor_pool_initial_size must be at most actor_pool_max_size."
                )
        _validate_remote_arg_conflicts(self)

    def resolve_actor_pool_defaults(
        self,
        *,
        model_configs: list[Any],
        model_providers: list[Any],
        default_provider_name: str,
        model_aliases: list[str],
    ) -> RayExecutionOptions:
        """Return execution options with provider-aware actor-pool defaults applied."""
        if self.use_actor_pool is not None:
            return self
        policy = resolve_provider_aware_actor_pool_policy(
            model_configs=model_configs,
            model_providers=model_providers,
            default_provider_name=default_provider_name,
            model_aliases=model_aliases,
        )
        if not policy.use_actor_pool:
            return replace(self, use_actor_pool=False)
        return replace(
            self,
            use_actor_pool=True,
            actor_pool_min_size=policy.min_size,
            actor_pool_max_size=policy.max_size,
            actor_pool_initial_size=policy.initial_size,
            _default_actor_pool_min_size=False,
        )

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
                raise RayBackendConfigurationError(
                    "RayExecutionOptions use_actor_pool=True requires an imported ray module."
                )
            kwargs["compute"] = _create_actor_pool_strategy(ray, self)
        return kwargs


@dataclass(frozen=True, slots=True)
class RayActorPoolPolicy:
    """Resolved Ray actor-pool policy for provider-aware defaults."""

    use_actor_pool: bool
    min_size: int | None = None
    max_size: int | None = None
    initial_size: int | None = None


@dataclass(frozen=True, slots=True)
class RayExecutionResources:
    """Typed Ray Data execution resource controls for DataContext options."""

    cpu: float | None = None
    gpu: float | None = None
    object_store_memory: float | None = None
    memory: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_non_negative_number("cpu", self.cpu)
        _validate_optional_non_negative_number("gpu", self.gpu)
        _validate_optional_non_negative_number("object_store_memory", self.object_store_memory)
        _validate_optional_non_negative_number("memory", self.memory)

    @property
    def has_explicit_controls(self) -> bool:
        """Return whether at least one resource quantity was configured."""
        return any(
            value is not None
            for value in (
                self.cpu,
                self.gpu,
                self.object_store_memory,
                self.memory,
            )
        )

    def to_execution_resources(self, ray: Any, *, for_limits: bool) -> Any:
        """Create Ray Data ``ExecutionResources`` from this typed config."""
        execution_resources = getattr(ray.data, "ExecutionResources", None)
        if not callable(execution_resources):
            raise RayBackendConfigurationError(
                "RayDataContextOptions resource controls require ray.data.ExecutionResources."
            )
        kwargs = self.to_kwargs()
        try:
            if for_limits:
                for_limits_factory = getattr(execution_resources, "for_limits", None)
                if callable(for_limits_factory):
                    return for_limits_factory(**kwargs)
            return execution_resources(**kwargs)
        except Exception as exc:
            raise RayBackendConfigurationError(
                "RayBackend failed to create Ray Data execution resources from RayDataContextOptions."
            ) from exc

    def to_kwargs(self) -> dict[str, float]:
        """Return non-None resource keyword arguments."""
        kwargs: dict[str, float] = {}
        _set_if_not_none(kwargs, "cpu", self.cpu)
        _set_if_not_none(kwargs, "gpu", self.gpu)
        _set_if_not_none(kwargs, "object_store_memory", self.object_store_memory)
        _set_if_not_none(kwargs, "memory", self.memory)
        return kwargs


@dataclass(frozen=True, slots=True)
class RayDataCheckpointConfig:
    """Typed subset of Ray Data job-level checkpoint configuration."""

    id_column: str | None = None
    checkpoint_path: str | None = None
    delete_checkpoint_on_success: bool = True
    filter_num_threads: int = 3
    write_num_threads: int = 3

    def __post_init__(self) -> None:
        _validate_optional_non_empty_string("id_column", self.id_column)
        _validate_optional_non_empty_string("checkpoint_path", self.checkpoint_path)
        _validate_bool("delete_checkpoint_on_success", self.delete_checkpoint_on_success)
        _validate_positive_int("filter_num_threads", self.filter_num_threads)
        _validate_positive_int("write_num_threads", self.write_num_threads)

    def to_context_value(self) -> dict[str, Any]:
        """Return a value accepted by ``DataContext.checkpoint_config``."""
        value: dict[str, Any] = {
            "delete_checkpoint_on_success": self.delete_checkpoint_on_success,
            "filter_num_threads": self.filter_num_threads,
            "write_num_threads": self.write_num_threads,
        }
        _set_if_not_none(value, "id_column", self.id_column)
        _set_if_not_none(value, "checkpoint_path", self.checkpoint_path)
        return value


@dataclass(frozen=True, slots=True)
class RayDataContextOptions:
    """Validated Ray DataContext and ExecutionOptions controls.

    These options apply to Ray Datasets created by RayBackend while planning a
    job. Ray seals DataContext values into each Dataset when it is created, so
    existing ``ray.data.Dataset`` inputs must be created with their desired
    DataContext before they are passed to DataDesigner.
    """

    resource_limits: RayExecutionResources | None = None
    exclude_resources: RayExecutionResources | None = None
    preserve_execution_order: bool | None = None
    actor_locality_enabled: bool | None = None
    verbose_progress: bool | None = None
    verbose_stats_logs: bool | None = None
    enable_progress_bars: bool | None = None
    enable_operator_progress_bars: bool | None = None
    log_internal_stack_trace_to_stdout: bool | None = None
    raise_original_map_exception: bool | None = None
    max_errored_blocks: int | None = None
    checkpoint_config: RayDataCheckpointConfig | None = None

    def __post_init__(self) -> None:
        _validate_optional_resource_controls("resource_limits", self.resource_limits)
        _validate_optional_resource_controls("exclude_resources", self.exclude_resources)
        _validate_non_overlapping_resources(self.resource_limits, self.exclude_resources)
        _validate_optional_bool("preserve_execution_order", self.preserve_execution_order)
        _validate_optional_bool("actor_locality_enabled", self.actor_locality_enabled)
        _validate_optional_bool("verbose_progress", self.verbose_progress)
        _validate_optional_bool("verbose_stats_logs", self.verbose_stats_logs)
        _validate_optional_bool("enable_progress_bars", self.enable_progress_bars)
        _validate_optional_bool("enable_operator_progress_bars", self.enable_operator_progress_bars)
        _validate_optional_bool("log_internal_stack_trace_to_stdout", self.log_internal_stack_trace_to_stdout)
        _validate_optional_bool("raise_original_map_exception", self.raise_original_map_exception)
        if self.max_errored_blocks is not None and (
            isinstance(self.max_errored_blocks, bool) or not isinstance(self.max_errored_blocks, int)
        ):
            raise RayBackendConfigurationError("RayDataContextOptions max_errored_blocks must be an integer.")
        if self.checkpoint_config is not None and not isinstance(self.checkpoint_config, RayDataCheckpointConfig):
            raise RayBackendConfigurationError(
                "RayDataContextOptions checkpoint_config must be a RayDataCheckpointConfig instance."
            )

    @property
    def has_explicit_controls(self) -> bool:
        """Return whether any DataContext value should be applied."""
        return any(
            value is not None
            for value in (
                self.resource_limits,
                self.exclude_resources,
                self.preserve_execution_order,
                self.actor_locality_enabled,
                self.verbose_progress,
                self.verbose_stats_logs,
                self.enable_progress_bars,
                self.enable_operator_progress_bars,
                self.log_internal_stack_trace_to_stdout,
                self.raise_original_map_exception,
                self.max_errored_blocks,
                self.checkpoint_config,
            )
        )

    def apply_to_context(self, ray: Any, context: Any) -> None:
        """Apply configured controls to a copied Ray DataContext."""
        execution_options = getattr(context, "execution_options", None)
        if self._has_execution_option_controls and execution_options is None:
            raise RayBackendConfigurationError("RayDataContextOptions require DataContext.execution_options.")
        if self.resource_limits is not None and self.resource_limits.has_explicit_controls:
            execution_options.resource_limits = self.resource_limits.to_execution_resources(ray, for_limits=True)
        if self.exclude_resources is not None and self.exclude_resources.has_explicit_controls:
            execution_options.exclude_resources = self.exclude_resources.to_execution_resources(ray, for_limits=False)
        _set_object_attribute(
            execution_options,
            "preserve_order",
            self.preserve_execution_order,
            owner_label="Ray Data ExecutionOptions",
        )
        _set_object_attribute(
            execution_options,
            "actor_locality_enabled",
            self.actor_locality_enabled,
            owner_label="Ray Data ExecutionOptions",
        )
        _set_object_attribute(
            execution_options,
            "verbose_progress",
            self.verbose_progress,
            owner_label="Ray Data ExecutionOptions",
        )
        validate = getattr(execution_options, "validate", None)
        if callable(validate):
            try:
                validate()
            except Exception as exc:
                raise RayBackendConfigurationError(
                    "RayDataContextOptions produced invalid Ray Data ExecutionOptions."
                ) from exc

        _set_object_attribute(context, "verbose_stats_logs", self.verbose_stats_logs, owner_label="Ray DataContext")
        _set_object_attribute(context, "enable_progress_bars", self.enable_progress_bars, owner_label="Ray DataContext")
        _set_object_attribute(
            context,
            "enable_operator_progress_bars",
            self.enable_operator_progress_bars,
            owner_label="Ray DataContext",
        )
        _set_object_attribute(
            context,
            "log_internal_stack_trace_to_stdout",
            self.log_internal_stack_trace_to_stdout,
            owner_label="Ray DataContext",
        )
        _set_object_attribute(
            context,
            "raise_original_map_exception",
            self.raise_original_map_exception,
            owner_label="Ray DataContext",
        )
        _set_object_attribute(context, "max_errored_blocks", self.max_errored_blocks, owner_label="Ray DataContext")
        if self.checkpoint_config is not None:
            if not hasattr(context, "checkpoint_config"):
                raise RayBackendConfigurationError(
                    "RayDataContextOptions checkpoint_config requires DataContext.checkpoint_config."
                )
            try:
                context.checkpoint_config = self.checkpoint_config.to_context_value()
            except Exception as exc:
                raise RayBackendConfigurationError(
                    "RayBackend failed to configure Ray Data checkpointing from RayDataContextOptions."
                ) from exc

    @property
    def _has_execution_option_controls(self) -> bool:
        return any(
            value is not None
            for value in (
                self.resource_limits,
                self.exclude_resources,
                self.preserve_execution_order,
                self.actor_locality_enabled,
                self.verbose_progress,
            )
        )


@dataclass(frozen=True, slots=True)
class RayInputRepartition:
    """Controls for Ray Data repartitioning of an existing input dataset."""

    num_blocks: int | None = None
    target_num_rows_per_block: int | None = None
    shuffle: bool = False

    def __post_init__(self) -> None:
        _validate_optional_positive_int("num_blocks", self.num_blocks)
        _validate_optional_positive_int("target_num_rows_per_block", self.target_num_rows_per_block)
        if not isinstance(self.shuffle, bool):
            raise RayBackendConfigurationError("RayInputRepartition shuffle must be a boolean.")
        if self.num_blocks is not None and self.target_num_rows_per_block is not None:
            raise RayBackendConfigurationError(
                "RayInputRepartition num_blocks cannot be combined with target_num_rows_per_block."
            )
        if self.shuffle and self.target_num_rows_per_block is not None:
            raise RayBackendConfigurationError(
                "RayInputRepartition shuffle can only be used with num_blocks. "
                "Ray target_num_rows_per_block repartitioning is streaming and does not shuffle."
            )

    @property
    def has_explicit_controls(self) -> bool:
        """Return whether the input dataset should be repartitioned."""
        return self.num_blocks is not None or self.target_num_rows_per_block is not None

    def to_repartition_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``ray.data.Dataset.repartition``."""
        if self.num_blocks is not None:
            return {"num_blocks": self.num_blocks, "shuffle": self.shuffle}
        if self.target_num_rows_per_block is not None:
            return {"target_num_rows_per_block": self.target_num_rows_per_block}
        return {}


@dataclass(frozen=True, slots=True)
class RayBackendOptionResolution:
    """Resolved RayBackend option objects after applying compatibility shims."""

    block_planning: RayBlockPlanning
    execution_options: RayExecutionOptions
    input_repartition: RayInputRepartition
    data_context_options: RayDataContextOptions


def resolve_ray_backend_options(
    *,
    block_planning: RayBlockPlanning | None = None,
    execution_options: RayExecutionOptions | None = None,
    input_repartition: RayInputRepartition | None = None,
    data_context_options: RayDataContextOptions | None = None,
    legacy_options: Mapping[str, Any] | None = None,
) -> RayBackendOptionResolution:
    """Resolve RayBackend planning and execution option objects.

    Args:
        block_planning: Primary block-planning option object.
        execution_options: Primary Ray execution option object.
        input_repartition: Primary input-dataset repartition option object.
        legacy_options: Backwards-compatible constructor kwargs for individual
            block-planning or execution settings.

    Returns:
        The resolved option objects used by RayBackend.

    Raises:
        RayBackendConfigurationError: If option objects are combined with
            effective individual kwargs, or if unsupported legacy kwargs are
            present.
    """
    remaining_options = dict(legacy_options or {})
    block_planning_kwargs = _pop_legacy_options(
        remaining_options,
        _RAY_BACKEND_BLOCK_PLANNING_LEGACY_KWARGS,
    )
    execution_options_kwargs = _pop_legacy_options(
        remaining_options,
        _RAY_BACKEND_EXECUTION_OPTIONS_LEGACY_KWARGS,
    )
    if remaining_options:
        unsupported_options = ", ".join(sorted(remaining_options))
        raise RayBackendConfigurationError(
            "RayBackend received unsupported Ray option arguments: "
            f"{unsupported_options}. Use block_planning=RayBlockPlanning(...) or "
            "execution_options=RayExecutionOptions(...) for Ray planning and execution controls."
        )
    return RayBackendOptionResolution(
        block_planning=_resolve_ray_block_planning(block_planning, block_planning_kwargs),
        execution_options=_resolve_ray_execution_options(execution_options, execution_options_kwargs),
        input_repartition=_resolve_ray_input_repartition(input_repartition),
        data_context_options=_resolve_ray_data_context_options(data_context_options),
    )


def create_ray_data_context(ray: Any, options: RayDataContextOptions) -> Any | None:
    """Return a copied Ray DataContext with backend options applied."""
    if not options.has_explicit_controls:
        return None
    data_context_cls = getattr(ray.data, "DataContext", None)
    get_current = getattr(data_context_cls, "get_current", None)
    if not callable(get_current):
        raise RayBackendConfigurationError("RayDataContextOptions require ray.data.DataContext.get_current().")
    try:
        current_context = get_current()
        copy_context = getattr(current_context, "copy", None)
        data_context = copy_context() if callable(copy_context) else copy.deepcopy(current_context)
        options.apply_to_context(ray, data_context)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayBackendConfigurationError("RayBackend failed to configure Ray DataContext options.") from exc
    return data_context


def _pop_legacy_options(source: dict[str, Any], option_names: frozenset[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for option_name in option_names:
        if option_name in source:
            options[option_name] = source.pop(option_name)
    return options


def _resolve_ray_block_planning(
    block_planning: RayBlockPlanning | None,
    legacy_options: Mapping[str, Any],
) -> RayBlockPlanning:
    if block_planning is not None and any(value is not None for value in legacy_options.values()):
        raise RayBackendConfigurationError(
            "RayBackend block_planning cannot be combined with individual block planning arguments."
        )
    if block_planning is not None:
        return block_planning
    return RayBlockPlanning(
        override_num_blocks=legacy_options.get("override_num_blocks"),
        target_block_size=legacy_options.get("target_block_size"),
        min_blocks=legacy_options.get("min_blocks"),
        max_blocks=legacy_options.get("max_blocks"),
        read_concurrency=legacy_options.get("read_concurrency"),
    )


def _resolve_ray_execution_options(
    execution_options: RayExecutionOptions | None,
    legacy_options: Mapping[str, Any],
) -> RayExecutionOptions:
    if execution_options is not None and _has_effective_execution_legacy_options(legacy_options):
        raise RayBackendConfigurationError(
            "RayBackend execution_options cannot be combined with individual Ray execution arguments."
        )
    if execution_options is not None:
        return execution_options
    use_actor_pool = legacy_options["use_actor_pool"] if "use_actor_pool" in legacy_options else None
    return RayExecutionOptions(
        num_cpus=legacy_options.get("num_cpus"),
        num_gpus=legacy_options.get("num_gpus"),
        memory=legacy_options.get("memory"),
        resources=legacy_options.get("resources"),
        scheduling_strategy=legacy_options.get("scheduling_strategy"),
        compute=legacy_options.get("compute"),
        concurrency=legacy_options.get("map_concurrency"),
        ray_remote_args_fn=legacy_options.get("ray_remote_args_fn"),
        ray_remote_args=legacy_options.get("ray_remote_args"),
        use_actor_pool=use_actor_pool,
        actor_pool_min_size=legacy_options.get("actor_pool_min_size"),
        actor_pool_max_size=legacy_options.get("actor_pool_max_size"),
        actor_pool_initial_size=legacy_options.get("actor_pool_initial_size"),
    )


def _resolve_ray_input_repartition(input_repartition: RayInputRepartition | None) -> RayInputRepartition:
    if input_repartition is None:
        return RayInputRepartition()
    if not isinstance(input_repartition, RayInputRepartition):
        raise RayBackendConfigurationError("RayBackend input_repartition must be a RayInputRepartition instance.")
    return input_repartition


def _resolve_ray_data_context_options(data_context_options: RayDataContextOptions | None) -> RayDataContextOptions:
    if data_context_options is None:
        return RayDataContextOptions()
    if not isinstance(data_context_options, RayDataContextOptions):
        raise RayBackendConfigurationError("RayBackend data_context_options must be a RayDataContextOptions instance.")
    return data_context_options


def _has_effective_execution_legacy_options(legacy_options: Mapping[str, Any]) -> bool:
    for name, value in legacy_options.items():
        if name == "use_actor_pool":
            if value:
                return True
            continue
        if value is not None:
            return True
    return False


def resolve_provider_aware_actor_pool_policy(
    *,
    model_configs: list[Any],
    model_providers: list[Any],
    default_provider_name: str,
    model_aliases: list[str],
) -> RayActorPoolPolicy:
    """Resolve default Ray actor-pool sizing from referenced model providers.

    External provider APIs get a bounded actor pool so Ray does not create more
    model-client workers than the provider throttle budget can use. Local
    OpenAI-compatible endpoints are left in task mode by default because their
    best actor count depends on GPU placement and per-actor resource requests.
    """
    referenced_aliases = set(model_aliases)
    if not referenced_aliases:
        return RayActorPoolPolicy(use_actor_pool=False)

    providers_by_name = {provider.name: provider for provider in model_providers}
    external_caps: dict[tuple[str, str], int] = {}
    for model_config in model_configs:
        if model_config.alias not in referenced_aliases:
            continue
        provider_name = model_config.provider or default_provider_name
        provider = providers_by_name.get(provider_name)
        if _is_local_model_provider(provider):
            continue
        max_parallel_requests = int(model_config.inference_parameters.max_parallel_requests)
        throttle_key = (provider_name, model_config.model)
        external_caps[throttle_key] = min(external_caps.get(throttle_key, max_parallel_requests), max_parallel_requests)

    if not external_caps:
        return RayActorPoolPolicy(use_actor_pool=False)
    max_size = max(sum(external_caps.values()), 1)
    return RayActorPoolPolicy(use_actor_pool=True, max_size=max_size)


def _is_local_model_provider(provider: Any | None) -> bool:
    if provider is None:
        return False
    provider_type = getattr(provider, "provider_type", "")
    if isinstance(provider_type, str) and provider_type.lower() in _LOCAL_PROVIDER_TYPES:
        return True
    endpoint = getattr(provider, "endpoint", "")
    if not isinstance(endpoint, str) or not endpoint:
        return False
    parsed = urlparse(endpoint)
    hostname = parsed.hostname
    if hostname is None:
        return endpoint.startswith("unix:")
    return hostname.lower() in _LOCAL_PROVIDER_HOSTS


def _validate_optional_positive_int(field_name: str, value: int | None) -> None:
    if value is None:
        return
    _validate_positive_int(field_name, value)


def _validate_positive_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a positive integer.")


def _validate_optional_non_negative_number(field_name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a non-negative number.")
    validate_finite_number(field_name, value, error_type=RayBackendConfigurationError, error_label="Ray option")
    if value < 0:
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a non-negative number.")


def _validate_optional_positive_number(field_name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a positive number.")
    validate_finite_number(field_name, value, error_type=RayBackendConfigurationError, error_label="Ray option")
    if value <= 0:
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a positive number.")


def _validate_optional_bool(field_name: str, value: bool | None) -> None:
    if value is not None:
        _validate_bool(field_name, value)


def _validate_bool(field_name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a boolean.")


def _validate_optional_non_empty_string(field_name: str, value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise RayBackendConfigurationError(f"Ray option {field_name} must be a non-empty string.")


def _validate_optional_resource_controls(
    field_name: str,
    value: RayExecutionResources | None,
) -> None:
    if value is not None and not isinstance(value, RayExecutionResources):
        raise RayBackendConfigurationError(f"RayDataContextOptions {field_name} must be a RayExecutionResources.")


def _validate_non_overlapping_resources(
    resource_limits: RayExecutionResources | None,
    exclude_resources: RayExecutionResources | None,
) -> None:
    if resource_limits is None or exclude_resources is None:
        return
    overlapping_fields = [
        field_name
        for field_name in ("cpu", "gpu", "object_store_memory", "memory")
        if getattr(resource_limits, field_name) is not None and getattr(exclude_resources, field_name) is not None
    ]
    if overlapping_fields:
        field_list = ", ".join(overlapping_fields)
        raise RayBackendConfigurationError(
            "RayDataContextOptions resource_limits and exclude_resources cannot both set "
            f"the same resource(s): {field_list}."
        )


def _validate_resources(resources: Mapping[str, float] | None) -> None:
    if resources is None:
        return
    for name, value in resources.items():
        if not isinstance(name, str) or not name:
            raise RayBackendConfigurationError("RayExecutionOptions resource names must be non-empty strings.")
        _validate_optional_positive_number(f"resources[{name!r}]", value)


def _validate_map_concurrency(value: RayMapConcurrency | None) -> None:
    if value is None:
        return
    if isinstance(value, int):
        _validate_optional_positive_int("concurrency", value)
        return
    if not isinstance(value, tuple) or len(value) not in (2, 3):
        raise RayBackendConfigurationError(
            "RayExecutionOptions concurrency must be a positive integer or a 2/3-item tuple."
        )
    for index, item in enumerate(value):
        _validate_optional_positive_int(f"concurrency[{index}]", item)
    if value[0] > value[1]:
        raise RayBackendConfigurationError(
            "RayExecutionOptions concurrency minimum must be less than or equal to maximum."
        )
    if len(value) == 3 and not value[0] <= value[2] <= value[1]:
        raise RayBackendConfigurationError(
            "RayExecutionOptions concurrency initial size must be within the min/max range."
        )


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
        raise RayBackendConfigurationError(
            f"RayExecutionOptions received duplicate explicit and ray_remote_args values: {conflict_list}."
        )
    reserved_keys = sorted(_DATA_DESIGNER_OWNED_MAP_BATCHES_KEYS.intersection(options.ray_remote_args))
    if reserved_keys:
        reserved_list = ", ".join(reserved_keys)
        raise RayBackendConfigurationError(
            "RayExecutionOptions ray_remote_args cannot override DataDesigner-owned map_batches arguments: "
            f"{reserved_list}."
        )


def _create_actor_pool_strategy(ray: Any, options: RayExecutionOptions) -> Any:
    actor_pool_strategy = getattr(ray.data, "ActorPoolStrategy", None)
    if not callable(actor_pool_strategy):
        raise RayBackendConfigurationError(
            "RayExecutionOptions use_actor_pool=True requires ray.data.ActorPoolStrategy."
        )
    kwargs: dict[str, int] = {}
    min_size = options.actor_pool_min_size
    if min_size is None and options._default_actor_pool_min_size:
        min_size = 1
    _set_if_not_none(kwargs, "min_size", min_size)
    _set_if_not_none(kwargs, "max_size", options.actor_pool_max_size)
    _set_if_not_none(kwargs, "initial_size", options.actor_pool_initial_size)
    return actor_pool_strategy(**kwargs)


def _set_if_not_none(target: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        target[key] = value


def _set_object_attribute(target: Any, name: str, value: Any | None, *, owner_label: str) -> None:
    if value is None:
        return
    if not hasattr(target, name):
        raise RayBackendConfigurationError(f"{owner_label} does not support option {name!r}.")
    try:
        setattr(target, name, value)
    except Exception as exc:
        raise RayBackendConfigurationError(f"RayBackend failed to set {owner_label} option {name!r}.") from exc
