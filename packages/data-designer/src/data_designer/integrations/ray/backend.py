# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import ProcessorType
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.model_provider import resolve_model_provider_registry
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import create_resource_provider
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage
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

if TYPE_CHECKING:
    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider


RayOutputMode = Literal["dataset", "arrow_refs"]
RayObjectRefInputFormat = Literal["arrow", "pandas"]


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
        self._metrics_cache = _merge_driver_and_worker_metrics(self._driver_metrics, worker_metrics)
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

    def _materialize_dataset_for_metrics(self) -> None:
        materialize = getattr(self.dataset, "materialize", None)
        if not callable(materialize):
            return
        self.dataset = materialize()

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
        order_column: str | None = None,
        drop_order_column: bool = False,
        preserve_order: bool = False,
        allow_unsafe_processors: bool = False,
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
        self.batch_size = batch_size
        self.output = output
        self.object_ref_format = object_ref_format
        self.auto_init = auto_init
        self.zero_copy_batch = zero_copy_batch
        self.ray_remote_args = ray_remote_args
        self.order_column = order_column
        self.drop_order_column = drop_order_column
        self.preserve_order = preserve_order
        self.allow_unsafe_processors = allow_unsafe_processors
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

        use_input_dataset = input_dataset is not None
        if use_input_dataset and config_builder.get_seed_config() is not None:
            raise RayBackendConfigurationError(
                "RayBackend input_dataset is used as the seed dataset; remove the existing seed config."
            )
        if not use_input_dataset and config_builder.get_seed_config() is not None:
            raise RayBackendConfigurationError(
                "RayBackend does not yet support seed configs without input_dataset because partition offsets "
                "are not available. Pass seed data as input_dataset, or use the local backend until "
                "RayBackend partition-offset support exists."
            )
        if self.preserve_order and self.order_column is None:
            raise RayBackendConfigurationError(
                "RayBackend preserve_order=True requires order_column in this experimental backend. "
                "Automatic hidden row-id injection is tracked as follow-up hardening work."
            )
        if not self.allow_unsafe_processors:
            _validate_ray_safe_processors(config_builder)

        dataset = self._resolve_input_dataset(ray, input_dataset=input_dataset, num_records=num_records)
        input_blocks = _get_num_blocks(dataset)
        metrics_collector = _create_metrics_collector(ray, max_trace_events=self.max_trace_events)
        batch_size = self.batch_size or data_designer._run_config.buffer_size
        observability_options = _RayObservabilityOptions(
            profile_workers=self.profile_workers,
            trace_enabled=self.trace_enabled,
        )
        worker_options = _RayWorkerOptions(
            model_providers=list(data_designer._model_providers),
            default_provider_name=data_designer._model_provider_registry.get_default_provider_name(),
            secret_resolver=data_designer._secret_resolver,
            seed_readers=_clone_seed_readers_for_worker(data_designer._seed_reader_registry._readers.values()),
            managed_assets_path=str(data_designer._managed_assets_path),
            person_reader=data_designer._person_reader,
            mcp_providers=list(data_designer._mcp_providers),
            run_config=data_designer._run_config,
        )

        map_batches_kwargs: dict[str, Any] = {
            "fn_kwargs": {
                "config_builder": config_builder,
                "worker_options": worker_options,
                "use_input_dataset": use_input_dataset,
                "metrics_collector": metrics_collector,
                "observability_options": observability_options,
            },
            "batch_size": batch_size,
            "batch_format": "pandas",
            "zero_copy_batch": self.zero_copy_batch,
        }
        if self.ray_remote_args is not None:
            map_batches_kwargs.update(self.ray_remote_args)

        try:
            mapped = dataset.map_batches(_generate_batch, **map_batches_kwargs)
            mapped = self._apply_ordering(mapped)
            output = mapped.to_arrow_refs() if self.output == "arrow_refs" else None
            result_dataset = _dataset_from_arrow_refs(ray, output) if output is not None else mapped
        except RayDatasetGenerationError:
            raise
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed while constructing the Ray execution plan.") from exc

        output_blocks = len(output) if output is not None else _get_num_blocks(mapped)
        metrics = RayDatasetMetrics(
            total_rows=num_records if input_dataset is None else 0,
            blocks=output_blocks or input_blocks or 0,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        return RayDatasetCreationResults(
            dataset=result_dataset,
            config_builder=config_builder,
            metrics=metrics,
            ray=ray,
            metrics_collector=metrics_collector,
            output=output,
            observability_options=observability_options,
        )

    def _resolve_input_dataset(self, ray: Any, *, input_dataset: Any | None, num_records: int) -> Any:
        if input_dataset is None:
            return ray.data.range(num_records)
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

    def _apply_ordering(self, dataset: Any) -> Any:
        if self.order_column is None:
            return dataset
        ordered = dataset.sort(self.order_column)
        if self.drop_order_column:
            return ordered.drop_columns([self.order_column])
        return ordered


def _validate_ray_safe_processors(config_builder: DataDesignerConfigBuilder) -> None:
    unsafe_processors = [
        processor
        for processor in config_builder.get_processor_configs()
        if processor.processor_type != ProcessorType.DROP_COLUMNS
    ]
    if not unsafe_processors:
        return
    processor_names = ", ".join(f"{processor.name} ({processor.processor_type})" for processor in unsafe_processors)
    raise RayBackendConfigurationError(
        "RayBackend currently supports only distributed-safe processors. "
        f"Unsupported processor(s): {processor_names}. "
        "Pass allow_unsafe_processors=True to bypass this experimental guard."
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


def _dataset_from_arrow_refs(ray: Any, refs: list[Any]) -> Any:
    from_arrow_refs = getattr(ray.data, "from_arrow_refs", None)
    if not callable(from_arrow_refs):
        raise RayDatasetGenerationError(
            "RayBackend output='arrow_refs' requires ray.data.from_arrow_refs to expose a dataset "
            "backed by the materialized Arrow ObjectRefs."
        )
    return from_arrow_refs(refs)


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
    driver_metrics: RayDatasetMetrics, worker_metrics: RayDatasetMetrics
) -> RayDatasetMetrics:
    return RayDatasetMetrics(
        total_rows=worker_metrics.total_rows,
        blocks=worker_metrics.blocks,
        failed_blocks=worker_metrics.failed_blocks,
        elapsed_seconds=worker_metrics.elapsed_seconds,
        model_usage=worker_metrics.model_usage or driver_metrics.model_usage,
    )


def _clone_seed_readers_for_worker(readers: Iterable[SeedReader]) -> list[SeedReader]:
    return [_clone_seed_reader_for_worker(reader) for reader in readers]


def _clone_seed_reader_for_worker(reader: SeedReader) -> SeedReader:
    clone = copy.copy(reader)
    clone._reset_attachment_state()
    for attr in ("source", "secret_resolver"):
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone


def _generate_batch(
    batch: Any,
    *,
    config_builder: DataDesignerConfigBuilder,
    worker_options: _RayWorkerOptions,
    use_input_dataset: bool,
    metrics_collector: Any | None = None,
    observability_options: _RayObservabilityOptions | None = None,
) -> Any:
    start_time = time.perf_counter()
    block_id = uuid.uuid4().hex
    observability_options = observability_options or _RayObservabilityOptions()
    worker_context = _get_ray_worker_context()
    trace_events: list[RayTraceEvent] = []
    os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
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
            metrics_collector,
            RayWorkerMetrics(total_rows=0, blocks=1, elapsed_seconds=elapsed_seconds, block_id=block_id),
        )
        _record_worker_observability(
            metrics_collector,
            worker_profile=profile,
            trace_events=trace_events,
            throttle_snapshots=[],
        )
        return dataframe

    try:
        block_builder = copy.deepcopy(config_builder)
        if use_input_dataset:
            block_builder.with_seed_dataset(DataFrameSeedSource(df=dataframe.copy()))

        with tempfile.TemporaryDirectory(prefix="data-designer-ray-") as artifact_dir:
            if observability_options.trace_enabled:
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "resource_setup_started",
                        start_time,
                        row_count=num_records,
                        worker_context=worker_context,
                    )
                )
            ArtifactStorage.mkdir_if_needed(Path(artifact_dir))
            seed_readers = copy.deepcopy(worker_options.seed_readers)
            run_config = copy.deepcopy(worker_options.run_config)
            if observability_options.trace_enabled:
                run_config.async_trace = True
            resource_provider = create_resource_provider(
                artifact_storage=ArtifactStorage(artifact_path=artifact_dir, dataset_name="ray-block"),
                model_configs=block_builder.model_configs,
                secret_resolver=worker_options.secret_resolver,
                model_provider_registry=resolve_model_provider_registry(
                    worker_options.model_providers,
                    worker_options.default_provider_name,
                ),
                seed_reader_registry=SeedReaderRegistry(readers=seed_readers),
                person_reader=worker_options.person_reader or create_person_reader(worker_options.managed_assets_path),
                seed_dataset_source=(
                    block_builder.get_seed_config().source if block_builder.get_seed_config() is not None else None
                ),
                run_config=run_config,
                mcp_providers=worker_options.mcp_providers,
                tool_configs=block_builder.tool_configs,
            )
            builder = DatasetBuilder(
                data_designer_config=block_builder.build(),
                resource_provider=resource_provider,
                use_async=True,
            )
            if observability_options.trace_enabled:
                trace_events.append(
                    _create_trace_event(
                        block_id,
                        "resource_setup_completed",
                        start_time,
                        row_count=num_records,
                        worker_context=worker_context,
                    )
                )
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
            elapsed_seconds = time.perf_counter() - start_time
            model_usage = resource_provider.model_registry.get_model_usage_stats(elapsed_seconds)
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
            throttle_snapshots = _snapshot_worker_throttle(resource_provider.model_registry.throttle_manager)
            _record_worker_metrics(
                metrics_collector,
                RayWorkerMetrics(
                    total_rows=len(output),
                    blocks=1,
                    elapsed_seconds=elapsed_seconds,
                    model_usage=model_usage,
                    block_id=block_id,
                ),
            )
            _record_worker_observability(
                metrics_collector,
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
            metrics_collector,
            RayWorkerMetrics(
                total_rows=0,
                blocks=1,
                failed_blocks=1,
                elapsed_seconds=time.perf_counter() - start_time,
                block_id=block_id,
            ),
        )
        _record_worker_observability(
            metrics_collector,
            worker_profile=None,
            trace_events=trace_events,
            throttle_snapshots=[],
        )
        raise


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
