# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import os
import pickle
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

from pydantic import PrivateAttr

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy, SeedConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import create_resource_provider
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    FileSystemSeedReader,
    SeedReader,
    SeedReaderRegistry,
)
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.storage.media_storage import StorageMode
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray.metrics import (
    RayDatasetMetrics,
    RayWorkerMetrics,
    aggregate_ray_metrics,
    normalize_ray_worker_metrics,
)
from data_designer.integrations.ray.observability import (
    RayDatasetAnalysis,
    RayThrottleSnapshot,
    RayTraceEvent,
    RayWorkerProfile,
    normalize_ray_throttle_snapshot,
    normalize_ray_trace_event,
    normalize_ray_worker_profile,
)
from data_designer.integrations.ray.options import RayBlockPlanning, RayExecutionOptions, RayMapConcurrency
from data_designer.integrations.ray.processor_policy import validate_ray_safe_processors
from data_designer.integrations.ray.throttling import create_ray_throttle_manager

if TYPE_CHECKING:
    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider


RayOutputMode = Literal["dataset", "arrow_refs"]
RayObjectRefInputFormat = Literal["arrow", "pandas"]
_RAY_RANGE_ID_COLUMN = "id"
_RAY_INTERNAL_ROW_ID_COLUMN = "__data_designer_ray_row_id"


@dataclass(frozen=True)
class _RayWorkerOptions:
    model_providers: list[ModelProvider]
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: list[SeedReader]
    managed_assets_path: str
    person_reader: PersonReader | None
    mcp_providers: list[MCPProviderT]
    run_config: RunConfig
    throttle_manager: Any | None = None


@dataclass(frozen=True)
class _RaySeedWindow:
    start: int
    size: int


@dataclass(frozen=True)
class _RayExecutionPayload:
    config_json: str
    worker_options: _RayWorkerOptions
    use_input_dataset: bool
    seed_window: _RaySeedWindow | None = None
    seed_config: SeedConfig | None = None
    hidden_order_column: str | None = None


@dataclass(frozen=True)
class _RayObservabilityOptions:
    profile_workers: bool = False
    trace_enabled: bool = False


class RayDatasetCreationResults:
    """Results wrapper for Ray-resident Data Designer outputs."""

    def __init__(
        self,
        *,
        dataset: Any,
        config_builder: DataDesignerConfigBuilder,
        metrics: RayDatasetMetrics,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        throttle_manager: Any | None = None,
        output: Any | None = None,
        observability_options: _RayObservabilityOptions | None = None,
    ) -> None:
        self.dataset = dataset
        self._config_builder = config_builder
        self._driver_metrics = metrics
        self._metrics_cache: RayDatasetMetrics | None = None
        self._worker_metrics_cache: list[RayWorkerMetrics] | None = None
        self._analysis_cache: RayDatasetAnalysis | None = None
        self._ray = ray
        self._metrics_collector = metrics_collector
        self._throttle_manager = throttle_manager
        self._output = output
        self._observability_options = observability_options or _RayObservabilityOptions()

    def load_dataset(self) -> Any:
        """Return the Ray Dataset without materializing it on the driver."""
        return self.dataset

    def load_analysis(self) -> RayDatasetAnalysis | None:
        """Return Ray-native worker profiles and bounded traces when available."""
        return self.load_observability()

    def load_metrics(self) -> RayDatasetMetrics:
        """Return driver-visible Ray execution metrics."""
        if self._metrics_cache is not None:
            return self._metrics_cache
        if self._ray is None or self._metrics_collector is None:
            return self._driver_metrics
        try:
            worker_metrics_payloads = self._load_worker_metrics_payloads()
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to load worker metrics.") from exc
        if not worker_metrics_payloads:
            return self._driver_metrics
        worker_metrics = aggregate_ray_metrics(worker_metrics_payloads)
        self._metrics_cache = _merge_driver_and_worker_metrics(
            self._driver_metrics,
            worker_metrics,
            throttle_metrics=self._load_throttle_metrics(),
        )
        return self._metrics_cache

    def load_worker_metrics(self) -> list[RayWorkerMetrics]:
        """Return per-worker metrics payloads before dataset-level aggregation."""
        if self._ray is None or self._metrics_collector is None:
            return []
        try:
            return list(self._load_worker_metrics_payloads())
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to load worker metrics.") from exc

    def load_observability(self) -> RayDatasetAnalysis | None:
        """Return bounded Ray-native profiles, traces, and worker-local throttle snapshots."""
        if self._analysis_cache is not None:
            return self._analysis_cache
        if self._ray is None or self._metrics_collector is None:
            return None
        try:
            payload = self._load_observability_payload()
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to load Ray observability artifacts.") from exc
        if not _has_observability_payload(payload):
            return None

        metrics = self.load_metrics()
        worker_profiles = [
            normalize_ray_worker_profile(profile) for profile in payload.get("worker_profiles", []) or []
        ]
        trace_events = [normalize_ray_trace_event(event) for event in payload.get("trace_events", []) or []]
        throttle_snapshots = [
            normalize_ray_throttle_snapshot(snapshot) for snapshot in payload.get("throttle_snapshots", []) or []
        ]
        self._analysis_cache = RayDatasetAnalysis(
            total_rows=metrics.total_rows,
            blocks=metrics.blocks,
            failed_blocks=metrics.failed_blocks,
            worker_profiles=worker_profiles,
            trace_events=trace_events,
            trace_events_dropped=int(payload.get("trace_events_dropped", 0) or 0),
            throttle_snapshots=throttle_snapshots,
        )
        return self._analysis_cache

    @property
    def metrics(self) -> RayDatasetMetrics:
        """Return the latest available Ray execution metrics."""
        return self.load_metrics()

    def _materialize_dataset_for_metrics(self) -> None:
        materialize = getattr(self.dataset, "materialize", None)
        if not callable(materialize):
            return
        self.dataset = materialize()

    def _load_worker_metrics_payloads(self) -> list[RayWorkerMetrics]:
        if self._worker_metrics_cache is not None:
            return self._worker_metrics_cache
        payloads = self._ray.get(self._metrics_collector.snapshot.remote())
        if not payloads:
            self._materialize_dataset_for_metrics()
            payloads = self._ray.get(self._metrics_collector.snapshot.remote())
        self._worker_metrics_cache = [normalize_ray_worker_metrics(payload) for payload in payloads]
        return self._worker_metrics_cache

    def _load_observability_payload(self) -> dict[str, Any]:
        observability_snapshot = getattr(self._metrics_collector, "observability_snapshot", None)
        if observability_snapshot is None:
            return {}
        payload = self._ray.get(observability_snapshot.remote())
        if not _has_observability_payload(payload):
            self._load_worker_metrics_payloads()
            payload = self._ray.get(observability_snapshot.remote())
        return dict(payload)

    def _load_throttle_metrics(self) -> dict[str, Any] | None:
        if self._throttle_manager is None:
            return None
        snapshot = getattr(self._throttle_manager, "snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot()

    def to_arrow_refs(self) -> list[Any]:
        """Return Ray ObjectRefs containing PyArrow tables, one per Ray block."""
        if self._output is not None:
            return self._output
        try:
            return self.dataset.to_arrow_refs()
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to materialize Arrow ObjectRefs.") from exc

    @property
    def output(self) -> Any:
        """Backend-selected output object."""
        return self._output if self._output is not None else self.dataset


class RayBackend:
    """Ray Data execution backend for in-memory Data Designer jobs.

    The backend maps Data Designer generation over Ray Data blocks and returns
    Ray-resident outputs. Ray is imported lazily so base Data Designer installs
    do not require the optional dependency. input_dataset may be a Ray Dataset
    or a sequence of ObjectRefs containing Arrow tables or pandas DataFrames.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        output: RayOutputMode = "dataset",
        object_ref_format: RayObjectRefInputFormat = "arrow",
        auto_init: bool = False,
        zero_copy_batch: bool = True,
        ray_remote_args: dict[str, Any] | None = None,
        block_planning: RayBlockPlanning | None = None,
        override_num_blocks: int | None = None,
        target_block_size: int | None = None,
        min_blocks: int | None = None,
        max_blocks: int | None = None,
        read_concurrency: int | None = None,
        execution_options: RayExecutionOptions | None = None,
        num_cpus: float | None = None,
        num_gpus: float | None = None,
        memory: float | None = None,
        resources: dict[str, float] | None = None,
        scheduling_strategy: Any | None = None,
        compute: Any | None = None,
        map_concurrency: RayMapConcurrency | None = None,
        ray_remote_args_fn: Any | None = None,
        use_actor_pool: bool = False,
        actor_pool_min_size: int | None = None,
        actor_pool_max_size: int | None = None,
        actor_pool_initial_size: int | None = None,
        preflight_model_health_check: bool = True,
        worker_model_health_checks: bool = False,
        order_column: str | None = None,
        drop_order_column: bool = False,
        preserve_order: bool = False,
        keep_internal_order_column: bool = False,
        allow_unsafe_processors: bool = False,
        global_provider_throttling: bool = True,
        profile_workers: bool = True,
        trace_enabled: bool = False,
        max_trace_events: int = 1000,
    ) -> None:
        if output not in ("dataset", "arrow_refs"):
            raise ValueError("RayBackend output must be 'dataset' or 'arrow_refs'.")
        if object_ref_format not in ("arrow", "pandas"):
            raise ValueError("RayBackend object_ref_format must be 'arrow' or 'pandas'.")
        if order_column is not None and order_column == "":
            raise ValueError("RayBackend order_column must be a non-empty string when provided.")
        if max_trace_events < 0:
            raise ValueError("RayBackend max_trace_events must be non-negative.")
        self.block_planning = _resolve_block_planning(
            block_planning,
            override_num_blocks=override_num_blocks,
            target_block_size=target_block_size,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            read_concurrency=read_concurrency,
        )
        self.execution_options = _resolve_execution_options(
            execution_options,
            ray_remote_args=ray_remote_args,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            resources=resources,
            scheduling_strategy=scheduling_strategy,
            compute=compute,
            map_concurrency=map_concurrency,
            ray_remote_args_fn=ray_remote_args_fn,
            use_actor_pool=use_actor_pool,
            actor_pool_min_size=actor_pool_min_size,
            actor_pool_max_size=actor_pool_max_size,
            actor_pool_initial_size=actor_pool_initial_size,
        )
        self.batch_size = batch_size
        self.output = output
        self.object_ref_format = object_ref_format
        self.auto_init = auto_init
        self.zero_copy_batch = zero_copy_batch
        self.ray_remote_args = ray_remote_args
        self.preflight_model_health_check = preflight_model_health_check
        self.worker_model_health_checks = worker_model_health_checks
        self.order_column = order_column
        self.drop_order_column = drop_order_column
        self.preserve_order = preserve_order
        self.keep_internal_order_column = keep_internal_order_column
        self.allow_unsafe_processors = allow_unsafe_processors
        self.global_provider_throttling = global_provider_throttling
        self.profile_workers = profile_workers
        self.trace_enabled = trace_enabled
        self.max_trace_events = max_trace_events

    def create(
        self,
        *,
        data_designer: Any,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> RayDatasetCreationResults:
        del dataset_name
        start_time = time.perf_counter()
        ray = _import_ray()
        if not ray.is_initialized():
            if not self.auto_init:
                raise RayBackendConfigurationError(
                    "Ray is not initialized. Call ray.init(...) before using RayBackend, "
                    "or construct RayBackend(auto_init=True)."
                )
            ray.init()

        external_input_dataset = input_dataset is not None
        use_input_dataset = external_input_dataset
        seed_config = config_builder.get_seed_config()
        if use_input_dataset and seed_config is not None:
            raise RayBackendConfigurationError(
                "RayBackend input_dataset is used as the seed dataset; remove the existing seed config."
            )
        if use_input_dataset and self.block_planning.has_explicit_controls:
            raise RayBackendConfigurationError(
                "RayBackend block planning controls apply only when RayBackend creates a from-scratch "
                "range dataset. Remove override_num_blocks, target_block_size, min_blocks, max_blocks, "
                "and read_concurrency when passing input_dataset."
            )
        seed_window: _RaySeedWindow | None = None
        if not use_input_dataset and seed_config is not None:
            if _should_materialize_seed_on_driver(data_designer=data_designer, seed_config=seed_config):
                input_dataset = ray.data.from_pandas(
                    _materialize_seed_dataframe(
                        data_designer=data_designer,
                        seed_config=seed_config,
                        num_records=num_records,
                    )
                )
                use_input_dataset = True
            else:
                seed_window = _preflight_seed_window(
                    data_designer=data_designer,
                    seed_config=seed_config,
                    num_records=num_records,
                )
        if not self.allow_unsafe_processors:
            validate_ray_safe_processors(config_builder)

        model_aliases = _model_health_check_aliases(config_builder)
        if self.preflight_model_health_check:
            _run_driver_model_health_check(data_designer, config_builder, model_aliases)

        block_plan = self.block_planning.resolve(num_records=num_records) if input_dataset is None else None
        dataset = self._resolve_input_dataset(
            ray,
            input_dataset=input_dataset,
            num_records=num_records,
            block_plan=block_plan,
        )
        hidden_order_column = _RAY_INTERNAL_ROW_ID_COLUMN if self.preserve_order and self.order_column is None else None
        if hidden_order_column is not None and use_input_dataset:
            dataset = _attach_hidden_row_id_column(ray, dataset, hidden_order_column=hidden_order_column)
        input_blocks = _get_num_blocks(dataset)
        metrics_collector = _create_metrics_collector(ray, max_trace_events=self.max_trace_events)
        batch_size = self.batch_size or data_designer._run_config.buffer_size
        observability_options = _RayObservabilityOptions(
            profile_workers=self.profile_workers,
            trace_enabled=self.trace_enabled,
        )
        throttle_manager = (
            create_ray_throttle_manager(ray, data_designer._run_config)
            if self.global_provider_throttling
            and config_builder.model_configs
            and callable(getattr(ray, "remote", None))
            else None
        )
        worker_config_builder = _clone_config_builder_for_worker(
            config_builder,
            worker_model_health_checks=self.worker_model_health_checks,
        )
        worker_seed_readers = (
            [DataFrameSeedReader()]
            if use_input_dataset
            else _clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values())
        )
        worker_options = _RayWorkerOptions(
            model_providers=list(data_designer._model_providers),
            default_provider_name=data_designer._model_provider_registry.get_default_provider_name(),
            secret_resolver=data_designer._secret_resolver,
            seed_readers=worker_seed_readers,
            managed_assets_path=str(data_designer._managed_assets_path),
            person_reader=data_designer._person_reader,
            mcp_providers=list(data_designer._mcp_providers),
            run_config=data_designer._run_config,
            throttle_manager=throttle_manager,
        )
        execution_payload = _compile_ray_execution_payload(
            config_builder=worker_config_builder,
            worker_options=worker_options,
            use_input_dataset=use_input_dataset,
            seed_window=seed_window,
            hidden_order_column=hidden_order_column,
        )

        map_batches_kwargs: dict[str, Any] = {
            "fn_constructor_kwargs": {
                "execution_payload": execution_payload,
                "metrics_collector": metrics_collector,
                "observability_options": observability_options,
            },
            "batch_size": batch_size,
            "batch_format": "pandas",
            "zero_copy_batch": self.zero_copy_batch,
        }
        map_batches_kwargs.update(self.execution_options.to_map_batches_kwargs(ray))

        try:
            mapped = dataset.map_batches(_RayBatchWorker, **map_batches_kwargs)
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError(
                "RayBackend failed while constructing the Ray map_batches execution plan."
            ) from exc

        try:
            mapped = self._apply_ordering(mapped, hidden_order_column=hidden_order_column)
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed while applying Ray output ordering.") from exc

        try:
            output = mapped.to_arrow_refs() if self.output == "arrow_refs" else None
            result_dataset = _dataset_from_arrow_refs(ray, output) if output is not None else mapped
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError(
                "RayBackend failed while materializing Ray output blocks for output='arrow_refs'."
            ) from exc

        output_blocks = len(output) if output is not None else _get_num_blocks(mapped)
        metrics = RayDatasetMetrics(
            total_rows=num_records if not external_input_dataset else 0,
            blocks=output_blocks or input_blocks or (block_plan.planned_blocks if block_plan is not None else 0) or 0,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        return RayDatasetCreationResults(
            dataset=result_dataset,
            config_builder=config_builder,
            metrics=metrics,
            ray=ray,
            metrics_collector=metrics_collector,
            throttle_manager=throttle_manager,
            output=output,
            observability_options=observability_options,
        )

    def _resolve_input_dataset(
        self,
        ray: Any,
        *,
        input_dataset: Any | None,
        num_records: int,
        block_plan: Any | None,
    ) -> Any:
        if input_dataset is None:
            range_kwargs = block_plan.to_range_kwargs() if block_plan is not None else {}
            return ray.data.range(num_records, **range_kwargs)
        if hasattr(input_dataset, "map_batches"):
            return input_dataset
        if isinstance(input_dataset, (list, tuple)):
            refs = list(input_dataset)
            if self.object_ref_format == "pandas":
                return ray.data.from_pandas_refs(refs)
            return ray.data.from_arrow_refs(refs)
        raise TypeError(
            "RayBackend input_dataset must be a ray.data.Dataset or a sequence of Ray ObjectRefs "
            "containing PyArrow tables or pandas DataFrames."
        )

    def _apply_ordering(self, dataset: Any, *, hidden_order_column: str | None) -> Any:
        order_column = self.order_column or hidden_order_column
        if order_column is None:
            return dataset
        ordered = dataset.sort(order_column)
        if hidden_order_column is not None and order_column == hidden_order_column:
            if self.keep_internal_order_column:
                return ordered
            return ordered.drop_columns([hidden_order_column])
        if self.drop_order_column:
            return ordered.drop_columns([order_column])
        return ordered


def _resolve_block_planning(
    block_planning: RayBlockPlanning | None,
    *,
    override_num_blocks: int | None,
    target_block_size: int | None,
    min_blocks: int | None,
    max_blocks: int | None,
    read_concurrency: int | None,
) -> RayBlockPlanning:
    if block_planning is not None and any(
        value is not None
        for value in (override_num_blocks, target_block_size, min_blocks, max_blocks, read_concurrency)
    ):
        raise ValueError("RayBackend block_planning cannot be combined with individual block planning arguments.")
    if block_planning is not None:
        return block_planning
    return RayBlockPlanning(
        override_num_blocks=override_num_blocks,
        target_block_size=target_block_size,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        read_concurrency=read_concurrency,
    )


def _resolve_execution_options(
    execution_options: RayExecutionOptions | None,
    *,
    ray_remote_args: dict[str, Any] | None,
    num_cpus: float | None,
    num_gpus: float | None,
    memory: float | None,
    resources: dict[str, float] | None,
    scheduling_strategy: Any | None,
    compute: Any | None,
    map_concurrency: RayMapConcurrency | None,
    ray_remote_args_fn: Any | None,
    use_actor_pool: bool,
    actor_pool_min_size: int | None,
    actor_pool_max_size: int | None,
    actor_pool_initial_size: int | None,
) -> RayExecutionOptions:
    explicit_values = (
        ray_remote_args,
        num_cpus,
        num_gpus,
        memory,
        resources,
        scheduling_strategy,
        compute,
        map_concurrency,
        ray_remote_args_fn,
        actor_pool_min_size,
        actor_pool_max_size,
        actor_pool_initial_size,
    )
    if execution_options is not None and (use_actor_pool or any(value is not None for value in explicit_values)):
        raise ValueError("RayBackend execution_options cannot be combined with individual Ray execution arguments.")
    if execution_options is not None:
        return execution_options
    return RayExecutionOptions(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory=memory,
        resources=resources,
        scheduling_strategy=scheduling_strategy,
        compute=compute,
        concurrency=map_concurrency,
        ray_remote_args_fn=ray_remote_args_fn,
        ray_remote_args=ray_remote_args,
        use_actor_pool=use_actor_pool,
        actor_pool_min_size=actor_pool_min_size,
        actor_pool_max_size=actor_pool_max_size,
        actor_pool_initial_size=actor_pool_initial_size,
    )


def _model_health_check_aliases(config_builder: DataDesignerConfigBuilder) -> list[str]:
    model_aliases: set[str] = set()
    for config in config_builder.get_column_configs():
        if column_type_is_model_generated(config.column_type):
            model_alias = getattr(config, "model_alias", None)
            if model_alias:
                model_aliases.add(model_alias)
        if isinstance(config, CustomColumnConfig) and config.model_aliases:
            model_aliases.update(config.model_aliases)
    return sorted(model_aliases)


def _run_driver_model_health_check(
    data_designer: Any,
    config_builder: DataDesignerConfigBuilder,
    model_aliases: list[str],
) -> None:
    if not model_aliases:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="data-designer-ray-preflight-") as artifact_dir:
            ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
            resource_provider = create_resource_provider(
                artifact_storage=ArtifactStorage(artifact_path=artifact_dir, dataset_name="ray-preflight"),
                model_configs=config_builder.model_configs,
                secret_resolver=data_designer._secret_resolver,
                model_provider_registry=data_designer._model_provider_registry,
                seed_reader_registry=data_designer._seed_reader_registry,
                person_reader=data_designer._person_reader
                or create_person_reader(str(data_designer._managed_assets_path)),
                run_config=data_designer._run_config,
                mcp_providers=list(data_designer._mcp_providers),
                tool_configs=config_builder.tool_configs,
            )
            resource_provider.model_registry.run_health_check(model_aliases)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend model health-check preflight failed.") from exc


def _clone_config_builder_for_worker(
    config_builder: DataDesignerConfigBuilder,
    *,
    worker_model_health_checks: bool,
) -> DataDesignerConfigBuilder:
    worker_config_builder = copy.deepcopy(config_builder)
    if worker_model_health_checks:
        return worker_config_builder
    worker_config_builder._model_configs = [
        model_config.model_copy(update={"skip_health_check": True})
        for model_config in worker_config_builder.model_configs
    ]
    return worker_config_builder


def _preflight_seed_window(
    *,
    data_designer: Any,
    seed_config: SeedConfig,
    num_records: int,
) -> _RaySeedWindow:
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend seed configs without input_dataset currently support only ordered sampling. "
            "Pass the seed data as input_dataset or use the local backend for shuffled seed sampling."
        )

    try:
        seed_reader = SeedReaderRegistry(
            readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values())
        ).get_reader(seed_config.source, data_designer._secret_resolver)
        seed_dataset_size = seed_reader.get_seed_dataset_size()
        index_range = _resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=seed_dataset_size)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayBackendConfigurationError(
            "RayBackend failed to preflight the seed config for partition offsets."
        ) from exc

    if num_records > index_range.size:
        raise RayBackendConfigurationError(
            "RayBackend seed configs without input_dataset require num_records to fit inside the selected seed "
            f"range. Requested {num_records} records from {index_range.size} selected seed rows."
        )
    return _RaySeedWindow(start=index_range.start, size=index_range.size)


def _should_materialize_seed_on_driver(*, data_designer: Any, seed_config: SeedConfig) -> bool:
    """Return whether Ray should treat a seed source as a materialized input dataset."""
    try:
        seed_reader = SeedReaderRegistry(
            readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values())
        ).get_reader(seed_config.source, data_designer._secret_resolver)
    except Exception as exc:
        raise RayBackendConfigurationError("RayBackend failed to inspect the seed reader for Ray planning.") from exc
    return isinstance(seed_reader, FileSystemSeedReader)


def _materialize_seed_dataframe(
    *,
    data_designer: Any,
    seed_config: SeedConfig,
    num_records: int,
) -> Any:
    if seed_config.sampling_strategy != SamplingStrategy.ORDERED:
        raise RayBackendConfigurationError(
            "RayBackend driver-materialized filesystem seed configs currently support only ordered sampling. "
            "Pass the seed data as input_dataset or use the local backend for shuffled seed sampling."
        )

    try:
        seed_reader = SeedReaderRegistry(
            readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values())
        ).get_reader(seed_config.source, data_designer._secret_resolver)
        seed_dataset_size = seed_reader.get_seed_dataset_size()
        index_range = _resolve_seed_config_index_range(seed_config=seed_config, seed_dataset_size=seed_dataset_size)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayBackendConfigurationError("RayBackend failed to materialize the seed config for Ray input.") from exc

    batch_reader = seed_reader.create_batch_reader(batch_size=num_records, index_range=index_range, shuffle=False)
    output = lazy.pd.DataFrame()
    empty_batch_reads = 0
    while len(output) < num_records:
        try:
            seed_batch = batch_reader.read_next_batch().to_pandas().reset_index(drop=True)
        except StopIteration:
            batch_reader = seed_reader.create_batch_reader(
                batch_size=num_records,
                index_range=index_range,
                shuffle=False,
            )
            continue

        if len(seed_batch) == 0:
            empty_batch_reads += 1
            if empty_batch_reads > max(num_records * 2, 1):
                raise RayBackendConfigurationError(
                    "RayBackend filesystem seed materialization did not produce any rows after repeated reads."
                )
            continue
        output = lazy.pd.concat([output, seed_batch], ignore_index=True)

    return output.iloc[:num_records].reset_index(drop=True)


def _resolve_seed_config_index_range(*, seed_config: SeedConfig, seed_dataset_size: int) -> IndexRange:
    if seed_dataset_size <= 0:
        raise RayBackendConfigurationError("RayBackend seed config resolved to an empty seed dataset.")
    if seed_config.selection_strategy is None:
        return IndexRange(start=0, end=seed_dataset_size - 1)
    if isinstance(seed_config.selection_strategy, IndexRange):
        if seed_config.selection_strategy.end >= seed_dataset_size:
            raise RayBackendConfigurationError(
                "RayBackend seed config selection_strategy is out of bounds for the seed dataset. "
                f"Selection end={seed_config.selection_strategy.end}, seed rows={seed_dataset_size}."
            )
        return seed_config.selection_strategy
    if isinstance(seed_config.selection_strategy, PartitionBlock):
        if seed_config.selection_strategy.num_partitions > seed_dataset_size:
            raise RayBackendConfigurationError(
                "RayBackend seed config selection_strategy is out of bounds for the seed dataset. "
                f"Partition count={seed_config.selection_strategy.num_partitions}, seed rows={seed_dataset_size}."
            )
        return seed_config.selection_strategy.to_index_range(seed_dataset_size)
    raise RayBackendConfigurationError(
        "RayBackend seed config uses an unsupported selection_strategy for partitioned execution."
    )


def _get_num_blocks(dataset: Any) -> int | None:
    num_blocks = getattr(dataset, "num_blocks", None)
    if not callable(num_blocks):
        return None
    try:
        value = num_blocks()
    except Exception:
        return None
    return int(value) if value is not None else None


def _count_dataset_rows(dataset: Any) -> int:
    count = getattr(dataset, "count", None)
    if not callable(count):
        raise RayBackendConfigurationError(
            "RayBackend preserve_order=True for input datasets requires ray.data.Dataset.count()."
        )
    try:
        return int(count())
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend failed to count input rows for hidden order preservation.") from exc


def _attach_hidden_row_id_column(ray: Any, dataset: Any, *, hidden_order_column: str) -> Any:
    row_count = _count_dataset_rows(dataset)
    try:
        order_dataset = ray.data.range(row_count).map_batches(
            _rename_range_id_column,
            fn_kwargs={"hidden_order_column": hidden_order_column},
            batch_format="pandas",
        )
        zip_dataset = getattr(dataset, "zip", None)
        if not callable(zip_dataset):
            raise RayBackendConfigurationError(
                "RayBackend preserve_order=True for input datasets requires ray.data.Dataset.zip()."
            )
        return zip_dataset(order_dataset)
    except RayBackendConfigurationError:
        raise
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend failed to attach hidden row ids to the input dataset.") from exc


def _rename_range_id_column(batch: Any, *, hidden_order_column: str) -> Any:
    dataframe = _coerce_pandas_dataframe(batch)
    if _RAY_RANGE_ID_COLUMN not in dataframe.columns:
        raise RayDatasetGenerationError(
            f"RayBackend expected ray.data.range() batches to include {_RAY_RANGE_ID_COLUMN!r}."
        )
    return lazy.pd.DataFrame({hidden_order_column: dataframe[_RAY_RANGE_ID_COLUMN].tolist()})


def _dataset_from_arrow_refs(ray: Any, refs: list[Any]) -> Any:
    from_arrow_refs = getattr(ray.data, "from_arrow_refs", None)
    if not callable(from_arrow_refs):
        raise RayDatasetGenerationError(
            "RayBackend output='arrow_refs' requires ray.data.from_arrow_refs to expose a dataset "
            "backed by the materialized Arrow ObjectRefs."
        )
    try:
        return from_arrow_refs(refs)
    except Exception as exc:
        try:
            return _dataset_from_arrow_refs_via_items(ray, refs)
        except Exception:
            raise RayDatasetGenerationError(
                "RayBackend failed to rebuild a Ray Dataset from materialized Arrow ObjectRefs."
            ) from exc


def _dataset_from_arrow_refs_via_items(ray: Any, refs: list[Any]) -> Any:
    from_items = getattr(ray.data, "from_items", None)
    if not callable(from_items):
        raise RayDatasetGenerationError("RayBackend fallback requires ray.data.from_items.")
    records: list[dict[str, Any]] = []
    for table in ray.get(refs):
        to_pylist = getattr(table, "to_pylist", None)
        if callable(to_pylist):
            records.extend(to_pylist())
            continue
        to_pandas = getattr(table, "to_pandas", None)
        if callable(to_pandas):
            records.extend(to_pandas().to_dict(orient="records"))
            continue
        raise RayDatasetGenerationError("RayBackend Arrow ObjectRef fallback expected PyArrow tables with to_pylist().")
    kwargs = {"override_num_blocks": len(refs)} if refs else {}
    return from_items(records, **kwargs)


def _compile_ray_execution_payload(
    *,
    config_builder: DataDesignerConfigBuilder,
    worker_options: _RayWorkerOptions,
    use_input_dataset: bool,
    seed_window: _RaySeedWindow | None = None,
    hidden_order_column: str | None = None,
) -> _RayExecutionPayload:
    data_designer_config = config_builder.build()
    payload_seed_config = data_designer_config.seed_config if seed_window is not None else None
    if seed_window is not None:
        data_designer_config.seed_config = None
    config_json = data_designer_config.to_json(indent=None)
    if config_json is None:
        raise RayBackendConfigurationError("RayBackend failed to serialize the Data Designer configuration.")
    payload = _RayExecutionPayload(
        config_json=config_json,
        worker_options=worker_options,
        use_input_dataset=use_input_dataset,
        seed_window=seed_window,
        seed_config=payload_seed_config,
        hidden_order_column=hidden_order_column,
    )
    _validate_worker_payload_serializable(payload)
    return payload


def _validate_worker_payload_serializable(payload: _RayExecutionPayload) -> None:
    try:
        serializer = importlib.import_module("cloudpickle")
    except ImportError:
        serializer = pickle
    try:
        serializer.dumps(payload)
    except Exception as exc:
        if payload.worker_options.throttle_manager is not None:
            worker_options = replace(payload.worker_options, throttle_manager=None)
            payload_without_throttle = replace(payload, worker_options=worker_options)
            try:
                serializer.dumps(payload_without_throttle)
            except Exception:
                pass
            else:
                return
        raise RayBackendConfigurationError(
            "RayBackend worker payload is not serializable. Check model providers, seed readers, "
            "secret resolvers, and MCP providers for process-safe state before launching Ray workers."
        ) from exc


class _RayMetricsCollector:
    def __init__(self, max_trace_events: int = 1000) -> None:
        self._payloads: list[dict[str, Any]] = []
        self._worker_profiles: list[dict[str, Any]] = []
        self._trace_events: list[dict[str, Any]] = []
        self._trace_events_dropped = 0
        self._throttle_snapshots: list[dict[str, Any]] = []
        self._max_trace_events = max_trace_events

    def record(self, payload: dict[str, Any]) -> None:
        self._payloads.append(payload)

    def record_observability(self, payload: dict[str, Any]) -> None:
        profile = payload.get("worker_profile")
        if isinstance(profile, dict):
            self._worker_profiles.append(profile)
        for event in payload.get("trace_events", []) or []:
            if len(self._trace_events) >= self._max_trace_events:
                self._trace_events_dropped += 1
                continue
            if isinstance(event, dict):
                self._trace_events.append(event)
        for snapshot in payload.get("throttle_snapshots", []) or []:
            if isinstance(snapshot, dict):
                self._throttle_snapshots.append(snapshot)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._payloads)

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "worker_profiles": list(self._worker_profiles),
            "trace_events": list(self._trace_events),
            "trace_events_dropped": self._trace_events_dropped,
            "throttle_snapshots": list(self._throttle_snapshots),
        }


def _create_metrics_collector(ray: Any, *, max_trace_events: int = 1000) -> Any | None:
    remote = getattr(ray, "remote", None)
    if not callable(remote):
        return None
    return remote(_RayMetricsCollector).remote(max_trace_events=max_trace_events)


def _merge_driver_and_worker_metrics(
    driver_metrics: RayDatasetMetrics,
    worker_metrics: RayDatasetMetrics,
    *,
    throttle_metrics: dict[str, Any] | None = None,
) -> RayDatasetMetrics:
    return RayDatasetMetrics(
        total_rows=worker_metrics.total_rows,
        blocks=worker_metrics.blocks,
        failed_blocks=worker_metrics.failed_blocks,
        elapsed_seconds=worker_metrics.elapsed_seconds,
        model_usage=worker_metrics.model_usage or driver_metrics.model_usage,
        throttle=throttle_metrics or worker_metrics.throttle or driver_metrics.throttle,
    )


def _clone_seed_readers_for_worker(readers: Iterable[SeedReader]) -> list[SeedReader]:
    clones = [_clone_seed_reader_for_worker(reader) for reader in readers]
    seed_types = {reader.get_seed_type() for reader in clones}
    dataframe_reader = DataFrameSeedReader()
    if dataframe_reader.get_seed_type() not in seed_types:
        clones.append(dataframe_reader)
    return clones


def _clone_seed_reader_for_worker(reader: SeedReader) -> SeedReader:
    clone = copy.copy(reader)
    clone._reset_attachment_state()
    for attr in ("source", "secret_resolver"):
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone


class _InMemoryPreviewArtifactStorage(ArtifactStorage):
    """ArtifactStorage implementation for Ray preview batches.

    Ray workers use ``DatasetBuilder.build_preview()``, which returns the main
    dataset in memory. Processor preview artifacts may still be written through
    the storage interface, so this class keeps those frames in memory and avoids
    per-block temporary filesystem setup.
    """

    _frames: dict[tuple[str, str, str], Any] = PrivateAttr(default_factory=dict)
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, *, dataset_name: str = "ray-block") -> None:
        super().__init__(artifact_path=Path.cwd(), dataset_name=dataset_name)
        self.set_media_storage_mode(StorageMode.DATAFRAME)

    def clear(self) -> None:
        self._frames.clear()
        self._metadata.clear()

    def load_dataset(self, batch_stage: BatchStage = BatchStage.FINAL_RESULT) -> Any:
        frames = [self._copy_dataframe(frame) for _, frame in self._iter_stage_frames(batch_stage)]
        if not frames:
            return lazy.pd.DataFrame()
        return lazy.pd.concat(frames, ignore_index=True)

    def load_processor_dataset(self, processor_name: str) -> Any:
        frames = [
            self._copy_dataframe(frame)
            for (stage, subfolder, parquet_file_name), frame in self._frames.items()
            if stage == BatchStage.PROCESSORS_OUTPUTS.value
            and (subfolder == processor_name or (not subfolder and Path(parquet_file_name).stem == processor_name))
        ]
        if not frames:
            return lazy.pd.DataFrame()
        return lazy.pd.concat(frames, ignore_index=True)

    def list_processor_names(self) -> list[str]:
        names = {
            subfolder or Path(parquet_file_name).stem
            for (stage, subfolder, parquet_file_name) in self._frames
            if stage == BatchStage.PROCESSORS_OUTPUTS.value
        }
        return sorted(names)

    def load_dataset_with_dropped_columns(self) -> Any:
        dataset = self.load_dataset()
        dropped = self.load_dataset(BatchStage.DROPPED_COLUMNS)
        if len(dropped) == 0:
            return dataset
        return lazy.pd.concat([dataset, dropped], axis=1)

    def move_partial_result_to_final_file_path(self, batch_number: int) -> Path:
        partial_file_name = self.create_batch_file_path(batch_number, BatchStage.PARTIAL_RESULT).name
        partial_key = (BatchStage.PARTIAL_RESULT.value, "", partial_file_name)
        final_key = (BatchStage.FINAL_RESULT.value, "", partial_file_name)
        frame = self._frames.pop(partial_key)
        self._frames[final_key] = frame
        return self.create_batch_file_path(batch_number, BatchStage.FINAL_RESULT)

    def write_batch_to_parquet_file(
        self,
        batch_number: int,
        dataframe: Any,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        file_path = self.create_batch_file_path(batch_number, batch_stage)
        self.write_parquet_file(file_path.name, dataframe, batch_stage, subfolder=subfolder)
        return file_path

    def write_parquet_file(
        self,
        parquet_file_name: str,
        dataframe: Any,
        batch_stage: BatchStage,
        subfolder: str | None = None,
    ) -> Path:
        subfolder = subfolder or ""
        stage = self._stage_value(batch_stage)
        self._frames[(stage, subfolder, parquet_file_name)] = self._copy_dataframe(dataframe)
        stage_path = self._get_stage_path(batch_stage)
        return stage_path / subfolder / parquet_file_name if subfolder else stage_path / parquet_file_name

    def get_parquet_file_paths(self) -> list[str]:
        return [
            str(Path(self.final_dataset_folder_name) / parquet_file_name)
            for (stage, _, parquet_file_name) in sorted(self._frames)
            if stage == BatchStage.FINAL_RESULT.value
        ]

    def get_processor_file_paths(self) -> dict[str, list[str]]:
        processor_files: dict[str, list[str]] = {}
        for stage, subfolder, parquet_file_name in sorted(self._frames):
            if stage != BatchStage.PROCESSORS_OUTPUTS.value:
                continue
            processor_name = subfolder or Path(parquet_file_name).stem
            path_parts = [self.processors_outputs_folder_name]
            if subfolder:
                path_parts.append(subfolder)
            path_parts.append(parquet_file_name)
            processor_files.setdefault(processor_name, []).append(str(Path(*path_parts)))
        return processor_files

    def get_file_paths(self) -> dict[str, list[str] | dict[str, list[str]]]:
        file_paths: dict[str, list[str] | dict[str, list[str]]] = {
            "parquet-files": self.get_parquet_file_paths(),
        }
        processor_file_paths = self.get_processor_file_paths()
        if processor_file_paths:
            file_paths["processor-files"] = processor_file_paths
        return file_paths

    def read_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def write_metadata(self, metadata: dict[str, Any]) -> Path:
        self._metadata = dict(metadata)
        return self.metadata_file_path

    def update_metadata(self, updates: dict[str, Any]) -> Path:
        self._metadata.update(updates)
        return self.metadata_file_path

    def _iter_stage_frames(self, batch_stage: BatchStage) -> list[tuple[tuple[str, str, str], Any]]:
        stage = self._stage_value(batch_stage)
        return sorted((key, frame) for key, frame in self._frames.items() if key[0] == stage)

    def _stage_value(self, batch_stage: BatchStage) -> str:
        return BatchStage(batch_stage).value

    def _copy_dataframe(self, dataframe: Any) -> Any:
        copy_method = getattr(dataframe, "copy", None)
        if not callable(copy_method):
            return dataframe
        try:
            return copy_method(deep=True)
        except TypeError:
            return copy_method()


class _RayBatchWorker:
    def __init__(
        self,
        *,
        execution_payload: _RayExecutionPayload,
        metrics_collector: Any | None = None,
        observability_options: _RayObservabilityOptions | None = None,
    ) -> None:
        os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
        self._execution_payload: _RayExecutionPayload = execution_payload
        self._metrics_collector: Any | None = metrics_collector
        self._observability_options: _RayObservabilityOptions = observability_options or _RayObservabilityOptions()
        self._base_config: DataDesignerConfig = DataDesignerConfig.model_validate_json(execution_payload.config_json)
        self._artifact_storage: _InMemoryPreviewArtifactStorage = _InMemoryPreviewArtifactStorage()
        worker_options = execution_payload.worker_options
        run_config = copy.deepcopy(worker_options.run_config)
        if self._observability_options.trace_enabled:
            run_config.async_trace = True
        self._seed_reader_registry: SeedReaderRegistry = SeedReaderRegistry(
            readers=copy.deepcopy(worker_options.seed_readers)
        )
        dataframe_seed_reader = DataFrameSeedReader()
        if (
            execution_payload.use_input_dataset
            and dataframe_seed_reader.get_seed_type() not in self._seed_reader_registry._readers
        ):
            self._seed_reader_registry.add_reader(dataframe_seed_reader)
        self._resource_provider: Any = create_resource_provider(
            artifact_storage=self._artifact_storage,
            model_configs=self._base_config.model_configs or [],
            secret_resolver=worker_options.secret_resolver,
            model_provider_registry=resolve_model_provider_registry(
                worker_options.model_providers,
                worker_options.default_provider_name,
            ),
            seed_reader_registry=self._seed_reader_registry,
            person_reader=worker_options.person_reader or create_person_reader(worker_options.managed_assets_path),
            seed_dataset_source=None,
            run_config=run_config,
            mcp_providers=worker_options.mcp_providers,
            tool_configs=self._base_config.tool_configs or [],
            throttle_manager=worker_options.throttle_manager,
        )

    def __call__(self, batch: Any) -> Any:
        return _generate_batch(batch, worker=self, metrics_collector=self._metrics_collector)

    def close(self) -> None:
        self._release_block_state()

    def generate_batch(self, batch: Any) -> Any:
        start_time = time.perf_counter()
        block_id = uuid.uuid4().hex
        observability_options = self._observability_options
        worker_context = _get_ray_worker_context()
        trace_events: list[RayTraceEvent] = []
        dataframe = _coerce_pandas_dataframe(batch)
        num_records = len(dataframe)
        if observability_options.trace_enabled:
            trace_events.append(
                _create_trace_event(
                    block_id,
                    "block_started",
                    start_time,
                    row_count=num_records,
                    worker_context=worker_context,
                    details={"input_columns": [str(column) for column in dataframe.columns]},
                )
            )
        if num_records == 0:
            elapsed_seconds = time.perf_counter() - start_time
            profile = (
                _profile_worker_output(dataframe, block_id=block_id, model_usage=None)
                if observability_options.profile_workers
                else None
            )
            if observability_options.trace_enabled:
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "block_completed",
                        start_time,
                        row_count=0,
                        worker_context=worker_context,
                        details={"empty_batch": True},
                    )
                )
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(total_rows=0, blocks=1, elapsed_seconds=elapsed_seconds, block_id=block_id),
            )
            _record_worker_observability(
                self._metrics_collector,
                worker_profile=profile,
                trace_events=trace_events,
                throttle_snapshots=[],
            )
            return dataframe

        try:
            order_values = _get_hidden_order_values(
                dataframe,
                hidden_order_column=self._execution_payload.hidden_order_column,
            )
            block_config = self._create_block_config(dataframe)
            self._attach_seed_reader(block_config)
            model_usage_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
            builder = DatasetBuilder(
                data_designer_config=block_config,
                resource_provider=self._resource_provider,
                use_async=True,
            )
            if observability_options.trace_enabled:
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "generation_started",
                        start_time,
                        row_count=num_records,
                        worker_context=worker_context,
                    )
                )
            raw_dataset = builder.build_preview(num_records=num_records)
            output = builder.process_preview(raw_dataset)
            if self._execution_payload.hidden_order_column is not None:
                output = _append_hidden_order_column(
                    output,
                    order_values=order_values,
                    hidden_order_column=self._execution_payload.hidden_order_column,
                )
            elapsed_seconds = time.perf_counter() - start_time
            model_usage = self._get_model_usage_delta(model_usage_snapshot, elapsed_seconds)
            if observability_options.trace_enabled:
                trace_events.extend(
                    _task_traces_to_events(
                        builder.task_traces,
                        block_id=block_id,
                        worker_context=worker_context,
                    )
                )
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "block_completed",
                        start_time,
                        row_count=len(output),
                        worker_context=worker_context,
                        details={"output_columns": [str(column) for column in output.columns]},
                    )
                )
            throttle_snapshots = _snapshot_worker_throttle(self._resource_provider.model_registry.throttle_manager)
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(
                    total_rows=len(output),
                    blocks=1,
                    elapsed_seconds=elapsed_seconds,
                    model_usage=model_usage,
                    block_id=block_id,
                ),
            )
            _record_worker_observability(
                self._metrics_collector,
                worker_profile=(
                    _profile_worker_output(output, block_id=block_id, model_usage=model_usage)
                    if observability_options.profile_workers
                    else None
                ),
                trace_events=trace_events,
                throttle_snapshots=throttle_snapshots,
            )
            return output
        except Exception as exc:
            if observability_options.trace_enabled:
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "block_failed",
                        start_time,
                        row_count=num_records,
                        worker_context=worker_context,
                        details={"error_type": type(exc).__name__, "error": str(exc)},
                    )
                )
            _record_worker_metrics(
                self._metrics_collector,
                RayWorkerMetrics(
                    total_rows=0,
                    blocks=1,
                    failed_blocks=1,
                    elapsed_seconds=time.perf_counter() - start_time,
                    block_id=block_id,
                ),
            )
            _record_worker_observability(
                self._metrics_collector,
                worker_profile=None,
                trace_events=trace_events,
                throttle_snapshots=[],
            )
            raise RayDatasetGenerationError(_format_worker_failure_message(dataframe=dataframe, exc=exc)) from exc
        finally:
            self._release_block_state()

    def _create_block_config(self, dataframe: Any) -> DataDesignerConfig:
        block_config = DataDesignerConfig.model_validate_json(self._execution_payload.config_json)
        if self._execution_payload.use_input_dataset:
            block_config.seed_config = SeedConfig(source=DataFrameSeedSource(df=_strip_internal_columns(dataframe)))
        elif self._execution_payload.seed_window is not None:
            if self._execution_payload.seed_config is None:
                raise RayDatasetGenerationError("RayBackend seed window was provided without a seed config.")
            block_config.seed_config = SeedConfig(
                source=DataFrameSeedSource(
                    df=_read_seed_window_dataframe(
                        seed_config=self._execution_payload.seed_config,
                        worker_options=self._execution_payload.worker_options,
                        dataframe=dataframe,
                        seed_window=self._execution_payload.seed_window,
                    )
                )
            )
        return block_config

    def _attach_seed_reader(self, block_config: DataDesignerConfig) -> None:
        if block_config.seed_config is None:
            self._resource_provider.seed_reader = None
            return
        self._resource_provider.seed_reader = self._seed_reader_registry.get_reader(
            block_config.seed_config.source,
            self._execution_payload.worker_options.secret_resolver,
        )

    def _get_model_usage_delta(
        self,
        model_usage_snapshot: Any,
        elapsed_seconds: float,
    ) -> dict[str, dict[str, Any]] | None:
        usage_deltas = self._resource_provider.model_registry.get_usage_deltas(model_usage_snapshot)
        if not usage_deltas:
            return None
        return {
            model_name: usage_stats.get_usage_stats(total_time_elapsed=elapsed_seconds)
            for model_name, usage_stats in usage_deltas.items()
        }

    def _release_block_state(self) -> None:
        self._resource_provider.seed_reader = None
        self._artifact_storage.clear()
        _detach_seed_readers(self._seed_reader_registry._readers.values())


def _detach_seed_readers(readers: Iterable[SeedReader]) -> None:
    for reader in readers:
        reader._reset_attachment_state()
        for attr in ("source", "secret_resolver"):
            if hasattr(reader, attr):
                delattr(reader, attr)


def _generate_batch(
    batch: Any,
    *,
    worker: _RayBatchWorker | None = None,
    config_builder: DataDesignerConfigBuilder | None = None,
    worker_options: _RayWorkerOptions | None = None,
    use_input_dataset: bool | None = None,
    seed_window: _RaySeedWindow | None = None,
    hidden_order_column: str | None = None,
    metrics_collector: Any | None = None,
    observability_options: _RayObservabilityOptions | None = None,
) -> Any:
    if worker is not None:
        return worker.generate_batch(batch)
    if config_builder is None or worker_options is None or use_input_dataset is None:
        raise TypeError("_generate_batch requires a worker or legacy config_builder/worker_options arguments.")
    execution_payload = _compile_ray_execution_payload(
        config_builder=config_builder,
        worker_options=worker_options,
        use_input_dataset=use_input_dataset,
        seed_window=seed_window,
        hidden_order_column=hidden_order_column,
    )
    return _RayBatchWorker(
        execution_payload=execution_payload,
        metrics_collector=metrics_collector,
        observability_options=observability_options,
    )(batch)


def _strip_internal_columns(dataframe: Any) -> Any:
    columns_to_drop = [column for column in (_RAY_INTERNAL_ROW_ID_COLUMN,) if column in dataframe.columns]
    if not columns_to_drop:
        return dataframe.copy()
    return dataframe.drop(columns=columns_to_drop).copy()


def _get_hidden_order_values(dataframe: Any, *, hidden_order_column: str | None) -> list[int]:
    if hidden_order_column is None:
        return []
    if hidden_order_column in dataframe.columns:
        values = dataframe[hidden_order_column].tolist()
    elif _RAY_RANGE_ID_COLUMN in dataframe.columns:
        values = dataframe[_RAY_RANGE_ID_COLUMN].tolist()
    else:
        raise RayDatasetGenerationError(
            "RayBackend preserve_order=True could not find the hidden row-id source column in a Ray worker batch."
        )
    return [int(value) for value in values]


def _append_hidden_order_column(
    output: Any,
    *,
    order_values: list[int],
    hidden_order_column: str,
) -> Any:
    if len(output) != len(order_values):
        raise RayDatasetGenerationError(
            "RayBackend preserve_order=True requires each Ray worker batch to emit one output row for each "
            f"input row. Input rows={len(order_values)}, output rows={len(output)}."
        )
    output = output.copy()
    output[hidden_order_column] = order_values
    return output


def _read_seed_window_dataframe(
    *,
    seed_config: SeedConfig,
    worker_options: _RayWorkerOptions,
    dataframe: Any,
    seed_window: _RaySeedWindow,
) -> Any:
    row_offsets = _get_contiguous_range_offsets(dataframe)
    index_range = IndexRange(
        start=seed_window.start + row_offsets[0],
        end=seed_window.start + row_offsets[-1],
    )
    if row_offsets[-1] >= seed_window.size:
        raise RayDatasetGenerationError(
            "RayBackend seed partition offset exceeded the selected seed range. "
            f"Offset={row_offsets[-1]}, selected seed rows={seed_window.size}."
        )

    seed_reader = SeedReaderRegistry(readers=copy.deepcopy(worker_options.seed_readers)).get_reader(
        seed_config.source,
        worker_options.secret_resolver,
    )
    batch_reader = seed_reader.create_batch_reader(
        batch_size=len(row_offsets),
        index_range=index_range,
        shuffle=False,
    )
    seed_dataframe = batch_reader.read_next_batch().to_pandas().reset_index(drop=True)
    if len(seed_dataframe) != len(row_offsets):
        raise RayDatasetGenerationError(
            "RayBackend seed reader returned an unexpected row count for a partition offset. "
            f"Expected {len(row_offsets)} rows, got {len(seed_dataframe)}."
        )
    return seed_dataframe


def _get_contiguous_range_offsets(dataframe: Any) -> list[int]:
    if _RAY_RANGE_ID_COLUMN not in dataframe.columns:
        raise RayDatasetGenerationError(
            f"RayBackend seed partition offsets require {_RAY_RANGE_ID_COLUMN!r} from ray.data.range()."
        )
    row_offsets = [int(value) for value in dataframe[_RAY_RANGE_ID_COLUMN].tolist()]
    if not row_offsets:
        raise RayDatasetGenerationError("RayBackend seed partition offsets received an empty Ray worker batch.")
    expected_offsets = list(range(row_offsets[0], row_offsets[0] + len(row_offsets)))
    if row_offsets != expected_offsets:
        raise RayDatasetGenerationError(
            "RayBackend seed partition offsets require contiguous ray.data.range() batches. "
            f"Received offsets {row_offsets!r}."
        )
    return row_offsets


def _format_worker_failure_message(*, dataframe: Any, exc: Exception) -> str:
    context_parts = [f"{len(dataframe)} input row(s)"]
    if _RAY_INTERNAL_ROW_ID_COLUMN in dataframe.columns:
        row_ids = [int(value) for value in dataframe[_RAY_INTERNAL_ROW_ID_COLUMN].tolist()]
        if row_ids:
            context_parts.append(f"logical rows {min(row_ids)}-{max(row_ids)}")
    elif _RAY_RANGE_ID_COLUMN in dataframe.columns:
        row_ids = [int(value) for value in dataframe[_RAY_RANGE_ID_COLUMN].tolist()]
        if row_ids:
            context_parts.append(f"range rows {min(row_ids)}-{max(row_ids)}")
    task_context = _get_ray_task_context()
    if task_context:
        context_parts.append(task_context)
    return f"RayBackend worker failed while generating a Ray block ({', '.join(context_parts)}): {exc}"


def _get_ray_task_context() -> str | None:
    try:
        ray = importlib.import_module("ray")
        get_runtime_context = getattr(ray, "get_runtime_context", None)
        if not callable(get_runtime_context):
            return None
        runtime_context = get_runtime_context()
    except Exception:
        return None

    context_values: list[str] = []
    for attr, method_name in (
        ("task_id", "get_task_id"),
        ("actor_id", "get_actor_id"),
        ("worker_id", "get_worker_id"),
    ):
        try:
            method = getattr(runtime_context, method_name, None)
            value = method() if callable(method) else getattr(runtime_context, attr, None)
        except Exception:
            continue
        if value is None:
            continue
        context_values.append(f"{attr}={value}")
    return ", ".join(context_values) if context_values else None


def _record_worker_observability(
    metrics_collector: Any | None,
    *,
    worker_profile: RayWorkerProfile | None,
    trace_events: list[RayTraceEvent],
    throttle_snapshots: list[RayThrottleSnapshot],
) -> None:
    if metrics_collector is None:
        return
    payload = {
        "worker_profile": worker_profile.to_dict() if worker_profile is not None else None,
        "trace_events": [event.to_dict() for event in trace_events],
        "throttle_snapshots": [snapshot.to_dict() for snapshot in throttle_snapshots],
    }
    importlib.import_module("ray").get(metrics_collector.record_observability.remote(payload))


def _record_worker_metrics(metrics_collector: Any | None, metrics: RayWorkerMetrics) -> None:
    if metrics_collector is None:
        return
    importlib.import_module("ray").get(metrics_collector.record.remote(metrics.to_dict()))


def _profile_worker_output(
    output: Any, *, block_id: str, model_usage: dict[str, dict[str, Any]] | None
) -> RayWorkerProfile:
    warnings: list[str] = []
    try:
        columns = [str(column) for column in output.columns]
        column_dtypes = {str(column): str(dtype) for column, dtype in output.dtypes.items()}
        non_null_counts = {str(column): int(value) for column, value in output.notna().sum().to_dict().items()}
        null_counts = {str(column): int(value) for column, value in output.isna().sum().to_dict().items()}
        memory_usage_bytes = int(output.memory_usage(deep=True).sum())
    except Exception as exc:
        columns = []
        column_dtypes = {}
        non_null_counts = {}
        null_counts = {}
        memory_usage_bytes = None
        warnings.append(f"Failed to profile Ray worker output: {type(exc).__name__}: {exc}")
    return RayWorkerProfile(
        block_id=block_id,
        total_rows=len(output),
        columns=columns,
        column_dtypes=column_dtypes,
        non_null_counts=non_null_counts,
        null_counts=null_counts,
        memory_usage_bytes=memory_usage_bytes,
        model_usage=model_usage,
        warnings=warnings,
    )


def _snapshot_worker_throttle(throttle_manager: Any | None) -> list[RayThrottleSnapshot]:
    if throttle_manager is None:
        return []
    domains = getattr(throttle_manager, "_domains", None)
    if not isinstance(domains, dict):
        return []
    get_effective_max = getattr(throttle_manager, "get_effective_max", None)
    snapshots: list[RayThrottleSnapshot] = []
    for key, state in domains.items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        provider_name, model_id, domain = key
        if not all(isinstance(value, str) for value in (provider_name, model_id, domain)):
            continue
        effective_max = get_effective_max(provider_name, model_id) if callable(get_effective_max) else None
        snapshots.append(
            RayThrottleSnapshot(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                current_limit=_safe_int_attr(state, "current_limit"),
                effective_max=effective_max,
                in_flight=_safe_int_attr(state, "in_flight"),
                waiters=_safe_int_attr(state, "waiters"),
                rate_limit_ceiling=_safe_int_attr(state, "rate_limit_ceiling"),
                consecutive_rate_limits=_safe_int_attr(state, "consecutive_429s"),
            )
        )
    return snapshots


def _task_traces_to_events(
    task_traces: list[Any],
    *,
    block_id: str,
    worker_context: dict[str, Any],
) -> list[RayTraceEvent]:
    events: list[RayTraceEvent] = []
    for task_trace in task_traces:
        dispatched_at = _safe_float_attr(task_trace, "dispatched_at")
        slot_acquired_at = _safe_float_attr(task_trace, "slot_acquired_at")
        completed_at = _safe_float_attr(task_trace, "completed_at")
        elapsed_seconds = max(completed_at - dispatched_at, 0.0) if completed_at > 0 and dispatched_at > 0 else 0.0
        wait_seconds = max(slot_acquired_at - dispatched_at, 0.0) if slot_acquired_at > 0 and dispatched_at > 0 else 0.0
        run_seconds = max(completed_at - slot_acquired_at, 0.0) if completed_at > 0 and slot_acquired_at > 0 else 0.0
        events.append(
            RayTraceEvent(
                block_id=block_id,
                event_type="engine_task",
                timestamp_seconds=time.time(),
                elapsed_seconds=elapsed_seconds,
                row_count=0,
                worker_hostname=worker_context["worker_hostname"],
                worker_pid=worker_context["worker_pid"],
                ray_task_id=worker_context.get("ray_task_id"),
                ray_node_id=worker_context.get("ray_node_id"),
                details={
                    "column": getattr(task_trace, "column", None),
                    "row_group": getattr(task_trace, "row_group", None),
                    "row_index": getattr(task_trace, "row_index", None),
                    "task_type": getattr(task_trace, "task_type", None),
                    "status": getattr(task_trace, "status", None),
                    "error": getattr(task_trace, "error", None),
                    "queue_wait_seconds": wait_seconds,
                    "run_seconds": run_seconds,
                },
            )
        )
    return events


def _create_trace_event(
    block_id: str,
    event_type: str,
    start_time: float,
    *,
    row_count: int,
    worker_context: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> RayTraceEvent:
    return RayTraceEvent(
        block_id=block_id,
        event_type=event_type,
        timestamp_seconds=time.time(),
        elapsed_seconds=time.perf_counter() - start_time,
        row_count=row_count,
        worker_hostname=worker_context["worker_hostname"],
        worker_pid=worker_context["worker_pid"],
        ray_task_id=worker_context.get("ray_task_id"),
        ray_node_id=worker_context.get("ray_node_id"),
        details=details,
    )


def _get_ray_worker_context() -> dict[str, Any]:
    context: dict[str, Any] = {
        "worker_hostname": socket.gethostname(),
        "worker_pid": os.getpid(),
        "ray_task_id": None,
        "ray_node_id": None,
    }
    try:
        ray = importlib.import_module("ray")
        get_runtime_context = getattr(ray, "get_runtime_context", None)
        runtime_context = get_runtime_context() if callable(get_runtime_context) else None
    except Exception:
        runtime_context = None
    if runtime_context is None:
        return context
    context["ray_task_id"] = _runtime_context_value(runtime_context, "get_task_id")
    context["ray_node_id"] = _runtime_context_value(runtime_context, "get_node_id")
    return context


def _runtime_context_value(runtime_context: Any, method_name: str) -> str | None:
    method = getattr(runtime_context, method_name, None)
    if not callable(method):
        return None
    try:
        value = method()
    except Exception:
        return None
    return str(value) if value is not None else None


def _safe_int_attr(value: Any, attr_name: str) -> int:
    attr = getattr(value, attr_name, 0)
    return attr if isinstance(attr, int) and not isinstance(attr, bool) and attr >= 0 else 0


def _safe_float_attr(value: Any, attr_name: str) -> float:
    attr = getattr(value, attr_name, 0.0)
    return float(attr) if isinstance(attr, (int, float)) and not isinstance(attr, bool) and attr >= 0 else 0.0


def _has_observability_payload(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("worker_profiles")
        or payload.get("trace_events")
        or payload.get("trace_events_dropped")
        or payload.get("throttle_snapshots")
    )


def _coerce_pandas_dataframe(batch: Any) -> Any:
    if isinstance(batch, lazy.pd.DataFrame):
        return batch
    if isinstance(batch, dict):
        return lazy.pd.DataFrame(batch)
    return lazy.pd.DataFrame(batch)


def _import_ray() -> Any:
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise ImportError("RayBackend requires Ray. Install Data Designer with `data-designer[ray]`.") from exc
