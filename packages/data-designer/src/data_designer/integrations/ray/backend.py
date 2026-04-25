# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.dataset_builders.block_execution import execute_dataset_block
from data_designer.engine.storage.artifact_storage import SDG_CONFIG_FILENAME, ArtifactStorage
from data_designer.integrations.ray import seed_planning as ray_seed_planning
from data_designer.integrations.ray.artifact_output import DataDesignerRayDatasink
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray.metrics import RayDatasetMetrics, RayWorkerMetrics
from data_designer.integrations.ray.observability import RayDatasetAnalysis
from data_designer.integrations.ray.observability_collection import (
    _create_metrics_collector,
    _RayObservabilityOptions,
)
from data_designer.integrations.ray.options import (
    RayBlockPlanning,
    RayExecutionOptions,
    RayInputRepartition,
    resolve_ray_backend_options,
)
from data_designer.integrations.ray.processor_policy import (
    validate_no_ray_after_generation_processors,
    validate_ray_safe_processors,
)
from data_designer.integrations.ray.results import RayResultArtifacts
from data_designer.integrations.ray.throttling import create_ray_throttle_manager
from data_designer.integrations.ray.worker_pipeline import (
    _RAY_INTERNAL_ROW_ID_COLUMN,
    _coerce_pandas_dataframe,
    _compile_ray_execution_payload,
    _RayExecutionPayload,
    _RayWorkerGenerationPipeline,
    _RayWorkerOptions,
)
from data_designer.interface.backends import BackendRuntimeContext

RayOutputMode = Literal["dataset", "arrow_refs"]
RayObjectRefInputFormat = Literal["arrow", "pandas"]
RayDatasetSourceKind = Literal[
    "range",
    "input_dataset",
    "object_refs",
    "driver_materialized_seed",
    "ray_native_seed",
    "seed_window",
]


@dataclass(frozen=True)
class RayDatasetSource:
    """Resolved Ray Dataset input for a planned job."""

    kind: RayDatasetSourceKind
    dataset: Any
    external_input_dataset: bool
    use_input_dataset: bool


@dataclass(frozen=True)
class RayOrderingMode:
    """Resolved ordering behavior for Ray output blocks."""

    order_column: str | None = None
    hidden_order_column: str | None = None
    drop_order_column: bool = False
    keep_internal_order_column: bool = False

    @property
    def requires_hidden_row_ids(self) -> bool:
        """Return whether input rows need a driver-attached hidden row id."""
        return self.hidden_order_column is not None

    def apply(self, dataset: Any) -> Any:
        """Apply resolved output ordering to a Ray Dataset."""
        order_column = self.order_column or self.hidden_order_column
        if order_column is None:
            return dataset
        ordered = dataset.sort(order_column)
        if self.hidden_order_column is not None and order_column == self.hidden_order_column:
            if self.keep_internal_order_column:
                return ordered
            return ordered.drop_columns([self.hidden_order_column])
        if self.drop_order_column:
            return ordered.drop_columns([order_column])
        return ordered


@dataclass(frozen=True)
class RayJobPlan:
    """Driver-resolved Ray execution plan.

    The plan owns all state needed after driver planning so execution does not
    re-read backend attributes or runtime context.
    """

    dataset_source: RayDatasetSource
    block_plan: Any | None
    worker_payload: _RayExecutionPayload
    map_batches_kwargs: dict[str, Any]
    ordering_mode: RayOrderingMode
    metrics_collector: Any | None
    throttle_manager: Any | None
    observability_options: _RayObservabilityOptions
    output: RayOutputMode
    num_records: int
    input_blocks: int | None


class RayDatasetCreationResults:
    """Public results facade for Ray-resident Data Designer outputs."""

    def __init__(
        self,
        *,
        dataset: Any,
        config_builder: DataDesignerConfigBuilder,
        metrics: RayDatasetMetrics,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        throttle_manager: Any | None = None,
        artifact_storage: ArtifactStorage | None = None,
        output: Any | None = None,
        observability_options: _RayObservabilityOptions | None = None,
    ) -> None:
        del config_builder, observability_options
        self.artifact_storage = artifact_storage
        self._artifacts = RayResultArtifacts(
            dataset=dataset,
            metrics=metrics,
            ray=ray,
            metrics_collector=metrics_collector,
            throttle_manager=throttle_manager,
            output=output,
        )

    @property
    def dataset(self) -> Any:
        """Return the Ray Dataset reference without Ray actor reads or driver materialization."""
        return self._artifacts.dataset

    @dataset.setter
    def dataset(self, value: Any) -> None:
        self._artifacts.dataset = value

    def load_dataset(self) -> Any:
        """Return the Ray Dataset without Ray actor reads or driver materialization."""
        return self._artifacts.load_dataset()

    def load_analysis(self) -> RayDatasetAnalysis | None:
        """Return Ray-native analysis.

        This may read the Ray metrics actor and can materialize the Ray Dataset
        once if worker metrics or observability payloads have not been emitted
        yet.
        """
        return self.load_observability()

    def load_metrics(self) -> RayDatasetMetrics:
        """Return driver-visible Ray execution metrics.

        This may read the Ray metrics actor. If the first metrics snapshot is
        empty, it explicitly materializes the Ray Dataset once and reads the
        actor again so lazy Ray execution has a chance to emit worker metrics.
        """
        return self._artifacts.load_metrics()

    def load_worker_metrics(self) -> list[RayWorkerMetrics]:
        """Return per-worker metrics payloads before dataset-level aggregation.

        This may read the Ray metrics actor. If the first metrics snapshot is
        empty, it explicitly materializes the Ray Dataset once and reads the
        actor again so lazy Ray execution has a chance to emit worker metrics.
        """
        return self._artifacts.load_worker_metrics()

    def load_observability(self) -> RayDatasetAnalysis | None:
        """Return bounded Ray-native profiles, traces, and worker-local throttle snapshots.

        This may read the Ray metrics actor. If observability is missing before
        worker execution, it asks the metrics loader to run its explicit
        materialization fallback before reading observability again.
        """
        return self._artifacts.load_observability()

    @property
    def metrics(self) -> RayDatasetMetrics:
        """Return the latest available Ray execution metrics.

        This has the same Ray actor read and materialization side effects as
        ``load_metrics()``.
        """
        return self.load_metrics()

    def to_arrow_refs(self) -> list[Any]:
        """Return Ray ObjectRefs containing PyArrow tables, one per Ray block.

        If the backend already selected ``output="arrow_refs"``, this returns
        the cached refs. Otherwise it calls ``Ray Dataset.to_arrow_refs()``,
        which materializes Ray output blocks.
        """
        return self._artifacts.to_arrow_refs()

    @property
    def output(self) -> Any:
        """Backend-selected output object without additional Ray actor reads.

        For dataset output this returns the Ray Dataset reference. For
        ``output="arrow_refs"`` this returns the refs materialized during
        backend execution.
        """
        return self._artifacts.output

    def load_processor_dataset(self, processor_name: str) -> Any:
        """Load a persisted processor output dataset as pandas."""
        if self.artifact_storage is None:
            raise RuntimeError("RayBackend was not configured to write artifacts.")
        return self.artifact_storage.load_processor_dataset(processor_name)


class RayBackend:
    """Ray Data execution backend for in-memory Data Designer jobs.

    The backend maps Data Designer generation over Ray Data blocks and returns
    Ray-resident outputs. When ``write_artifacts`` is enabled, the mapped Ray
    Dataset is also written through a Ray Datasink-compatible artifact writer
    using the standard Data Designer artifact layout. Ray is imported lazily so
    base Data Designer installs do not require the optional dependency.
    input_dataset may be a Ray Dataset or a sequence of ObjectRefs containing
    Arrow tables or pandas DataFrames.

    Prefer ``block_planning=RayBlockPlanning(...)`` and
    ``execution_options=RayExecutionOptions(...)`` for Ray-specific controls.
    Individual Ray planning and execution kwargs are still accepted as
    backwards-compatible shims.

    Observability artifacts are bounded on the metrics actor. ``profile_workers=True``
    computes one per-block worker profile before the collector applies
    ``max_worker_profiles``; trace events and throttle snapshots are bounded by
    ``max_trace_events`` and ``max_throttle_snapshots``. Keep worker profiling
    disabled for jobs where profiling overhead matters more than diagnostics.
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        output: RayOutputMode = "dataset",
        object_ref_format: RayObjectRefInputFormat = "arrow",
        auto_init: bool = False,
        zero_copy_batch: bool = True,
        block_planning: RayBlockPlanning | None = None,
        execution_options: RayExecutionOptions | None = None,
        input_repartition: RayInputRepartition | None = None,
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
        max_worker_profiles: int = 1000,
        max_throttle_snapshots: int = 1000,
        write_artifacts: bool = False,
        artifact_write_concurrency: int | None = None,
        artifact_write_ray_remote_args: dict[str, Any] | None = None,
        distributed_artifact_writes: bool = True,
        **legacy_options: Any,
    ) -> None:
        if output not in ("dataset", "arrow_refs"):
            raise RayBackendConfigurationError("RayBackend output must be 'dataset' or 'arrow_refs'.")
        if object_ref_format not in ("arrow", "pandas"):
            raise RayBackendConfigurationError("RayBackend object_ref_format must be 'arrow' or 'pandas'.")
        _validate_ray_backend_batch_size(batch_size)
        if order_column is not None and order_column == "":
            raise RayBackendConfigurationError("RayBackend order_column must be a non-empty string when provided.")
        _validate_ray_backend_non_negative_int("max_trace_events", max_trace_events)
        _validate_ray_backend_non_negative_int("max_worker_profiles", max_worker_profiles)
        _validate_ray_backend_non_negative_int("max_throttle_snapshots", max_throttle_snapshots)
        resolved_options = resolve_ray_backend_options(
            block_planning=block_planning,
            execution_options=execution_options,
            input_repartition=input_repartition,
            legacy_options=legacy_options,
        )
        self.block_planning = resolved_options.block_planning
        self.execution_options = resolved_options.execution_options
        self.input_repartition = resolved_options.input_repartition
        self.batch_size = batch_size
        self.output = output
        self.object_ref_format = object_ref_format
        self.auto_init = auto_init
        self.zero_copy_batch = zero_copy_batch
        self.ray_remote_args = self.execution_options.ray_remote_args
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
        self.max_worker_profiles = max_worker_profiles
        self.max_throttle_snapshots = max_throttle_snapshots
        self.write_artifacts = write_artifacts
        self.artifact_write_concurrency = artifact_write_concurrency
        self.artifact_write_ray_remote_args = artifact_write_ray_remote_args
        self.distributed_artifact_writes = distributed_artifact_writes

    def create(
        self,
        *,
        runtime_context: BackendRuntimeContext,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> RayDatasetCreationResults:
        start_time = time.perf_counter()
        ray = _import_ray()
        if not ray.is_initialized():
            if not self.auto_init:
                raise RayBackendConfigurationError(
                    "Ray is not initialized. Call ray.init(...) before using RayBackend, "
                    "or construct RayBackend(auto_init=True)."
                )
            ray.init()

        plan = RayDriverPlanner(backend=self, ray=ray).plan(
            runtime_context=runtime_context,
            config_builder=config_builder,
            num_records=num_records,
            input_dataset=input_dataset,
        )
        result_dataset, mapped, output = self._execute_plan(ray, plan, materialize_output=not self.write_artifacts)
        artifact_storage = None
        if self.write_artifacts:
            artifact_storage = self._write_artifacts(
                ray=ray,
                mapped=mapped,
                runtime_context=runtime_context,
                config_builder=config_builder,
                num_records=num_records,
                dataset_name=dataset_name,
                batch_size=int(plan.map_batches_kwargs["batch_size"]),
            )
            result_dataset = ray.data.read_parquet(str(artifact_storage.final_dataset_path))
            output = result_dataset.to_arrow_refs() if plan.output == "arrow_refs" else None
        metrics = _create_driver_metrics(
            plan,
            mapped=mapped,
            output=output,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        return RayDatasetCreationResults(
            dataset=result_dataset,
            config_builder=config_builder,
            metrics=metrics,
            ray=ray,
            metrics_collector=plan.metrics_collector,
            throttle_manager=plan.throttle_manager,
            artifact_storage=artifact_storage,
            output=output,
            observability_options=plan.observability_options,
        )

    def _execute_plan(
        self, ray: Any, plan: RayJobPlan, *, materialize_output: bool = True
    ) -> tuple[Any, Any, Any | None]:
        try:
            mapped = plan.dataset_source.dataset.map_batches(_RayBatchWorker, **plan.map_batches_kwargs)
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError(
                "RayBackend failed while constructing the Ray map_batches execution plan."
            ) from exc

        try:
            mapped = plan.ordering_mode.apply(mapped)
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed while applying Ray output ordering.") from exc

        if not materialize_output:
            return mapped, mapped, None

        try:
            output = mapped.to_arrow_refs() if plan.output == "arrow_refs" else None
            result_dataset = _dataset_from_arrow_refs(ray, output) if output is not None else mapped
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError(
                "RayBackend failed while materializing Ray output blocks for output='arrow_refs'."
            ) from exc
        return result_dataset, mapped, output

    def _write_artifacts(
        self,
        *,
        ray: Any,
        mapped: Any,
        runtime_context: BackendRuntimeContext,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        batch_size: int,
    ) -> ArtifactStorage:
        del ray
        ArtifactStorage.mkdir_if_needed(runtime_context.artifact_path)
        artifact_storage = ArtifactStorage(artifact_path=runtime_context.artifact_path, dataset_name=dataset_name)
        ArtifactStorage.mkdir_if_needed(artifact_storage.base_dataset_path)
        config_builder.get_builder_config().to_json(artifact_storage.base_dataset_path / SDG_CONFIG_FILENAME)

        datasink = DataDesignerRayDatasink(
            base_dataset_path=artifact_storage.base_dataset_path,
            dataset_name=artifact_storage.resolved_dataset_name,
            target_num_records=num_records,
            buffer_size=batch_size,
            min_rows_per_write=batch_size,
            supports_distributed_writes=self.distributed_artifact_writes,
        )
        try:
            mapped.write_datasink(
                datasink,
                ray_remote_args=self.artifact_write_ray_remote_args,
                concurrency=self.artifact_write_concurrency,
            )
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed while writing Data Designer artifacts.") from exc
        return artifact_storage

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
        raise RayBackendConfigurationError(
            "RayBackend input_dataset must be a ray.data.Dataset or a sequence of Ray ObjectRefs "
            "containing PyArrow tables or pandas DataFrames."
        )


class RayDriverPlanner:
    """Resolve driver-side Ray backend state into a job plan."""

    def __init__(self, *, backend: RayBackend, ray: Any) -> None:
        self._backend = backend
        self._ray = ray

    def plan(
        self,
        *,
        runtime_context: BackendRuntimeContext,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        input_dataset: Any | None = None,
    ) -> RayJobPlan:
        external_input_dataset = input_dataset is not None
        use_input_dataset = external_input_dataset
        dataset_source_kind = _initial_dataset_source_kind(input_dataset)
        seed_config = config_builder.get_seed_config()
        if use_input_dataset and seed_config is not None:
            raise RayBackendConfigurationError(
                "RayBackend input_dataset is used as the seed dataset; remove the existing seed config."
            )
        if use_input_dataset and self._backend.block_planning.has_explicit_controls:
            raise RayBackendConfigurationError(
                "RayBackend block planning controls apply only when RayBackend creates a from-scratch "
                "range dataset. Remove override_num_blocks, target_block_size, min_blocks, max_blocks, "
                "and read_concurrency when passing input_dataset."
            )
        if not external_input_dataset and self._backend.input_repartition.has_explicit_controls:
            raise RayBackendConfigurationError(
                "RayBackend input_repartition requires input_dataset. "
                "Use RayBlockPlanning for from-scratch range datasets."
            )

        seed_window: ray_seed_planning.RaySeedWindow | None = None
        if not use_input_dataset and seed_config is not None:
            seed_plan = ray_seed_planning.plan_seed_execution(
                runtime_context=runtime_context,
                seed_config=seed_config,
                num_records=num_records,
                ray=self._ray,
            )
            if seed_plan.input_dataframe is not None:
                input_dataset = self._ray.data.from_pandas(seed_plan.input_dataframe)
                use_input_dataset = True
                dataset_source_kind = "driver_materialized_seed"
            elif seed_plan.input_dataset is not None:
                input_dataset = seed_plan.input_dataset
                use_input_dataset = True
                dataset_source_kind = "ray_native_seed"
            else:
                seed_window = seed_plan.seed_window
                dataset_source_kind = "seed_window"

        validate_no_ray_after_generation_processors(config_builder)
        if not self._backend.allow_unsafe_processors:
            validate_ray_safe_processors(config_builder, allow_dataset_artifacts=self._backend.write_artifacts)

        model_aliases = _model_health_check_aliases(config_builder)
        if self._backend.preflight_model_health_check:
            _run_driver_model_health_check(runtime_context, config_builder, model_aliases)

        block_plan = self._backend.block_planning.resolve(num_records=num_records) if input_dataset is None else None
        dataset = self._backend._resolve_input_dataset(
            self._ray,
            input_dataset=input_dataset,
            num_records=num_records,
            block_plan=block_plan,
        )
        if external_input_dataset:
            dataset = _repartition_input_dataset(dataset, self._backend.input_repartition)
        ordering_mode = RayOrderingMode(
            order_column=self._backend.order_column,
            hidden_order_column=(
                _RAY_INTERNAL_ROW_ID_COLUMN
                if self._backend.preserve_order and self._backend.order_column is None
                else None
            ),
            drop_order_column=self._backend.drop_order_column,
            keep_internal_order_column=self._backend.keep_internal_order_column,
        )
        if ordering_mode.requires_hidden_row_ids and use_input_dataset:
            dataset = _attach_hidden_row_id_column(
                self._ray,
                dataset,
                hidden_order_column=_RAY_INTERNAL_ROW_ID_COLUMN,
            )

        metrics_collector = _create_metrics_collector(
            self._ray,
            max_trace_events=self._backend.max_trace_events,
            max_worker_profiles=self._backend.max_worker_profiles,
            max_throttle_snapshots=self._backend.max_throttle_snapshots,
        )
        observability_options = _RayObservabilityOptions(
            profile_workers=self._backend.profile_workers,
            trace_enabled=self._backend.trace_enabled,
            max_worker_profiles=self._backend.max_worker_profiles,
            max_throttle_snapshots=self._backend.max_throttle_snapshots,
        )
        throttle_manager = self._create_throttle_manager(runtime_context, config_builder)
        worker_config_builder = _clone_config_builder_for_worker(
            config_builder,
            worker_model_health_checks=self._backend.worker_model_health_checks,
        )
        worker_seed_readers = (
            ray_seed_planning.input_dataset_seed_readers()
            if use_input_dataset
            else ray_seed_planning.clone_seed_readers_for_worker(runtime_context.seed_readers)
        )
        worker_options = _RayWorkerOptions(
            model_providers=list(runtime_context.model_providers),
            default_provider_name=runtime_context.default_provider_name,
            secret_resolver=runtime_context.secret_resolver,
            seed_readers=worker_seed_readers,
            managed_assets_path=str(runtime_context.managed_assets_path),
            person_reader=runtime_context.person_reader,
            mcp_providers=list(runtime_context.mcp_providers),
            run_config=runtime_context.run_config,
            throttle_manager=throttle_manager,
        )
        execution_payload = _compile_ray_execution_payload(
            config_builder=worker_config_builder,
            worker_options=worker_options,
            use_input_dataset=use_input_dataset,
            seed_window=seed_window,
            hidden_order_column=ordering_mode.hidden_order_column,
            capture_artifacts=self._backend.write_artifacts,
        )

        map_batches_kwargs: dict[str, Any] = {
            "fn_constructor_kwargs": {
                "execution_payload": execution_payload,
                "metrics_collector": metrics_collector,
                "observability_options": observability_options,
            },
            "batch_size": self._backend.batch_size
            if self._backend.batch_size is not None
            else runtime_context.run_config.buffer_size,
            "batch_format": "pandas",
            "zero_copy_batch": self._backend.zero_copy_batch,
        }
        map_batches_kwargs.update(self._backend.execution_options.to_map_batches_kwargs(self._ray))

        return RayJobPlan(
            dataset_source=RayDatasetSource(
                kind=dataset_source_kind,
                dataset=dataset,
                external_input_dataset=external_input_dataset,
                use_input_dataset=use_input_dataset,
            ),
            block_plan=block_plan,
            worker_payload=execution_payload,
            map_batches_kwargs=map_batches_kwargs,
            ordering_mode=ordering_mode,
            metrics_collector=metrics_collector,
            throttle_manager=throttle_manager,
            observability_options=observability_options,
            output=self._backend.output,
            num_records=num_records,
            input_blocks=_get_num_blocks(dataset),
        )

    def _create_throttle_manager(
        self,
        runtime_context: BackendRuntimeContext,
        config_builder: DataDesignerConfigBuilder,
    ) -> Any | None:
        if not self._backend.global_provider_throttling:
            return None
        if not config_builder.model_configs:
            return None
        if not callable(getattr(self._ray, "remote", None)):
            return None
        return create_ray_throttle_manager(self._ray, runtime_context.run_config)


def _initial_dataset_source_kind(input_dataset: Any | None) -> RayDatasetSourceKind:
    if input_dataset is None:
        return "range"
    if hasattr(input_dataset, "map_batches"):
        return "input_dataset"
    return "object_refs"


def _repartition_input_dataset(dataset: Any, input_repartition: RayInputRepartition) -> Any:
    repartition_kwargs = input_repartition.to_repartition_kwargs()
    if not repartition_kwargs:
        return dataset
    repartition = getattr(dataset, "repartition", None)
    if not callable(repartition):
        raise RayBackendConfigurationError("RayBackend input_repartition requires ray.data.Dataset.repartition().")
    try:
        return repartition(**repartition_kwargs)
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend failed to repartition the input dataset.") from exc


def _create_driver_metrics(
    plan: RayJobPlan,
    *,
    mapped: Any,
    output: Any | None,
    elapsed_seconds: float,
) -> RayDatasetMetrics:
    output_blocks = len(output) if output is not None else _get_num_blocks(mapped)
    planned_blocks = plan.block_plan.planned_blocks if plan.block_plan is not None else 0
    return RayDatasetMetrics(
        total_rows=0 if plan.dataset_source.external_input_dataset else plan.num_records,
        blocks=output_blocks or plan.input_blocks or planned_blocks or 0,
        elapsed_seconds=elapsed_seconds,
    )


def _validate_ray_backend_batch_size(batch_size: int | None) -> None:
    if batch_size is None:
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RayBackendConfigurationError("RayBackend batch_size must be a positive integer or None.")


def _validate_ray_backend_non_negative_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RayBackendConfigurationError(f"RayBackend {field_name} must be a non-negative integer.")


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
    runtime_context: BackendRuntimeContext,
    config_builder: DataDesignerConfigBuilder,
    model_aliases: list[str],
) -> None:
    if not model_aliases:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="data-designer-ray-preflight-") as artifact_dir:
            ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
            resource_provider = runtime_context.create_resource_provider(
                artifact_storage=ArtifactStorage(artifact_path=artifact_dir, dataset_name="ray-preflight"),
                model_configs=config_builder.model_configs,
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
    if ray_seed_planning.RAY_RANGE_ID_COLUMN not in dataframe.columns:
        raise RayDatasetGenerationError(
            f"RayBackend expected ray.data.range() batches to include {ray_seed_planning.RAY_RANGE_ID_COLUMN!r}."
        )
    return lazy.pd.DataFrame({hidden_order_column: dataframe[ray_seed_planning.RAY_RANGE_ID_COLUMN].tolist()})


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


class _RayBatchWorker:
    def __init__(
        self,
        *,
        execution_payload: _RayExecutionPayload,
        metrics_collector: Any | None = None,
        observability_options: _RayObservabilityOptions | None = None,
    ) -> None:
        self._execution_payload: _RayExecutionPayload = execution_payload
        self._metrics_collector: Any | None = metrics_collector
        self._observability_options: _RayObservabilityOptions = observability_options or _RayObservabilityOptions()
        self._base_config: DataDesignerConfig = DataDesignerConfig.model_validate_json(execution_payload.config_json)
        self._pipeline = _RayWorkerGenerationPipeline(
            execution_payload=execution_payload,
            metrics_collector=metrics_collector,
            observability_options=self._observability_options,
            execute_block=execute_dataset_block,
        )
        self._worker_options: _RayWorkerOptions = self._pipeline.worker_options

    def __call__(self, batch: Any) -> Any:
        return _generate_batch(batch, worker=self, metrics_collector=self._metrics_collector)

    def close(self) -> None:
        return None

    def generate_batch(self, batch: Any) -> Any:
        worker_batch = self._pipeline.prepare_batch(batch)
        batch_observer = self._pipeline.begin_observability(worker_batch)
        if worker_batch.num_records == 0:
            return self._pipeline.complete_empty_batch(worker_batch, batch_observer)
        return self._pipeline.generate_non_empty_batch(worker_batch, batch_observer)


def _generate_batch(
    batch: Any,
    *,
    worker: _RayBatchWorker | None = None,
    config_builder: DataDesignerConfigBuilder | None = None,
    worker_options: _RayWorkerOptions | None = None,
    use_input_dataset: bool | None = None,
    seed_window: ray_seed_planning.RaySeedWindow | None = None,
    hidden_order_column: str | None = None,
    capture_artifacts: bool = False,
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
        capture_artifacts=capture_artifacts,
    )
    return _RayBatchWorker(
        execution_payload=execution_payload,
        metrics_collector=metrics_collector,
        observability_options=observability_options,
    )(batch)


def _import_ray() -> Any:
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        raise ImportError("RayBackend requires Ray. Install Data Designer with `data-designer[ray]`.") from exc
