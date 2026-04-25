# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import os
import pickle
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.seed import SeedConfig
from data_designer.engine.dataset_builders.block_execution import (
    BlockExecutionOptions,
    BlockExecutionResult,
    execute_dataset_block,
)
from data_designer.engine.resources.person_reader import PersonReader
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.integrations.ray import seed_planning as ray_seed_planning
from data_designer.integrations.ray.artifact_output import append_ray_artifact_columns
from data_designer.integrations.ray.errors import RayBackendConfigurationError, RayDatasetGenerationError
from data_designer.integrations.ray.metrics import RayWorkerMetrics
from data_designer.integrations.ray.observability import RayTraceEvent
from data_designer.integrations.ray.observability_collection import (
    _create_trace_event,
    _get_ray_worker_context,
    _profile_worker_output,
    _RayObservabilityOptions,
    _record_worker_metrics,
    _record_worker_observability,
    _snapshot_worker_throttle,
    _task_traces_to_events,
)

if TYPE_CHECKING:
    from data_designer.config.mcp import MCPProviderT
    from data_designer.config.models import ModelProvider


_RAY_INTERNAL_ROW_ID_COLUMN = "__data_designer_ray_row_id"
_ExecuteDatasetBlock = Callable[..., BlockExecutionResult]


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
class _RayExecutionPayload:
    config_json: str
    worker_options: _RayWorkerOptions
    use_input_dataset: bool
    seed_window: ray_seed_planning.RaySeedWindow | None = None
    seed_config: SeedConfig | None = None
    hidden_order_column: str | None = None
    capture_artifacts: bool = False


@dataclass(frozen=True)
class _RayWorkerBatch:
    dataframe: Any
    num_records: int
    block_id: str
    start_time: float
    worker_context: dict[str, Any]


@dataclass(frozen=True)
class _RayWorkerGenerationResult:
    dataframe: Any
    block_result: BlockExecutionResult
    model_usage: dict[str, dict[str, Any]] | None


class _RayWorkerAllRowsDroppedError(Exception):
    def __init__(self, block_result: BlockExecutionResult) -> None:
        super().__init__(
            f"all input rows were dropped during block execution (input_rows={block_result.input_rows}, output_rows=0)"
        )
        self.block_result = block_result


class _RayWorkerGenerationPipeline:
    """Execute one Ray worker batch through named generation phases."""

    def __init__(
        self,
        *,
        execution_payload: _RayExecutionPayload,
        metrics_collector: Any | None = None,
        observability_options: _RayObservabilityOptions | None = None,
        execute_block: _ExecuteDatasetBlock | None = None,
    ) -> None:
        os.environ["DATA_DESIGNER_ASYNC_ENGINE"] = "1"
        self._execution_payload = execution_payload
        self._observability_options = observability_options or _RayObservabilityOptions()
        self._worker_options = _prepare_worker_options(
            execution_payload,
            observability_options=self._observability_options,
        )
        self._execute_block = execute_block or execute_dataset_block
        self._observer = _RayWorkerObserver(
            metrics_collector=metrics_collector,
            observability_options=self._observability_options,
        )

    @property
    def worker_options(self) -> _RayWorkerOptions:
        return self._worker_options

    def generate_batch(self, batch: Any) -> Any:
        worker_batch = self.prepare_batch(batch)
        batch_observer = self.begin_observability(worker_batch)
        if worker_batch.num_records == 0:
            return self.complete_empty_batch(worker_batch, batch_observer)
        return self.generate_non_empty_batch(worker_batch, batch_observer)

    def begin_observability(self, worker_batch: _RayWorkerBatch) -> _RayWorkerBatchObserver:
        return self._observer.begin_batch(worker_batch)

    def generate_non_empty_batch(
        self,
        worker_batch: _RayWorkerBatch,
        batch_observer: _RayWorkerBatchObserver,
    ) -> Any:
        try:
            batch_observer.record_generation_started(worker_batch)
            generation_result = self.generate_rows(worker_batch)
            return self.complete_successful_batch(worker_batch, generation_result, batch_observer)
        except _RayWorkerAllRowsDroppedError as exc:
            batch_observer.record_all_rows_dropped(worker_batch, exc.block_result)
            raise RayDatasetGenerationError(
                _format_worker_failure_message(dataframe=worker_batch.dataframe, exc=exc)
            ) from None
        except Exception as exc:
            batch_observer.record_failure(worker_batch, exc)
            if isinstance(exc, RayDatasetGenerationError):
                raise
            raise RayDatasetGenerationError(
                _format_worker_failure_message(dataframe=worker_batch.dataframe, exc=exc)
            ) from exc

    def prepare_batch(self, batch: Any) -> _RayWorkerBatch:
        dataframe = _coerce_pandas_dataframe(batch)
        return _RayWorkerBatch(
            dataframe=dataframe,
            num_records=len(dataframe),
            block_id=uuid.uuid4().hex,
            start_time=time.perf_counter(),
            worker_context=_get_ray_worker_context(),
        )

    def generate_rows(self, worker_batch: _RayWorkerBatch) -> _RayWorkerGenerationResult:
        order_values = _get_hidden_order_values(
            worker_batch.dataframe,
            hidden_order_column=self._execution_payload.hidden_order_column,
        )
        block_config = self.create_block_config(worker_batch.dataframe)
        input_frame = (
            _strip_internal_columns(worker_batch.dataframe) if self._execution_payload.use_input_dataset else None
        )
        block_result = self._execute_block(
            data_designer_config=block_config,
            runtime_context=self._worker_options,
            input_frame=input_frame,
            num_records=worker_batch.num_records,
            options=BlockExecutionOptions(use_async=True, dataset_name="ray-block"),
        )
        if block_result.all_rows_dropped:
            raise _RayWorkerAllRowsDroppedError(block_result)

        output = block_result.dataframe
        if self._execution_payload.hidden_order_column is not None:
            output = _append_hidden_order_column(
                output,
                order_values=order_values,
                hidden_order_column=self._execution_payload.hidden_order_column,
            )
        return _RayWorkerGenerationResult(
            dataframe=output,
            block_result=block_result,
            model_usage=block_result.model_usage_stats or None,
        )

    def complete_empty_batch(self, worker_batch: _RayWorkerBatch, batch_observer: _RayWorkerBatchObserver) -> Any:
        batch_observer.record_empty(worker_batch)
        return worker_batch.dataframe

    def complete_successful_batch(
        self,
        worker_batch: _RayWorkerBatch,
        generation_result: _RayWorkerGenerationResult,
        batch_observer: _RayWorkerBatchObserver,
    ) -> Any:
        batch_observer.record_success(
            worker_batch,
            generation_result,
            throttle_manager=self._worker_options.throttle_manager,
        )
        if self._execution_payload.capture_artifacts:
            return append_ray_artifact_columns(generation_result.dataframe, generation_result.block_result)
        return generation_result.dataframe

    def create_block_config(self, dataframe: Any) -> DataDesignerConfig:
        block_config = DataDesignerConfig.model_validate_json(self._execution_payload.config_json)
        if self._execution_payload.seed_window is not None:
            if self._execution_payload.seed_config is None:
                raise RayDatasetGenerationError("RayBackend seed window was provided without a seed config.")
            block_config.seed_config = ray_seed_planning.create_seed_config_for_window(
                seed_config=self._execution_payload.seed_config,
                seed_readers=self._worker_options.seed_readers,
                secret_resolver=self._worker_options.secret_resolver,
                dataframe=dataframe,
                seed_window=self._execution_payload.seed_window,
            )
        return block_config


class _RayWorkerObserver:
    def __init__(
        self,
        *,
        metrics_collector: Any | None,
        observability_options: _RayObservabilityOptions,
    ) -> None:
        self._metrics_collector = metrics_collector
        self._observability_options = observability_options

    def begin_batch(self, worker_batch: _RayWorkerBatch) -> _RayWorkerBatchObserver:
        batch_observer = _RayWorkerBatchObserver(
            metrics_collector=self._metrics_collector,
            observability_options=self._observability_options,
        )
        batch_observer.record_block_started(worker_batch)
        return batch_observer


class _RayWorkerBatchObserver:
    def __init__(
        self,
        *,
        metrics_collector: Any | None,
        observability_options: _RayObservabilityOptions,
    ) -> None:
        self._metrics_collector = metrics_collector
        self._observability_options = observability_options
        self._trace_events: list[RayTraceEvent] = []

    def record_block_started(self, worker_batch: _RayWorkerBatch) -> None:
        if not self._observability_options.trace_enabled:
            return
        self._trace_events.append(
            _create_trace_event(
                worker_batch.block_id,
                "block_started",
                worker_batch.start_time,
                row_count=worker_batch.num_records,
                worker_context=worker_batch.worker_context,
                details={"input_columns": [str(column) for column in worker_batch.dataframe.columns]},
            )
        )

    def record_generation_started(self, worker_batch: _RayWorkerBatch) -> None:
        if not self._observability_options.trace_enabled:
            return
        self._trace_events.append(
            _create_trace_event(
                worker_batch.block_id,
                "generation_started",
                worker_batch.start_time,
                row_count=worker_batch.num_records,
                worker_context=worker_batch.worker_context,
            )
        )

    def record_empty(self, worker_batch: _RayWorkerBatch) -> None:
        elapsed_seconds = time.perf_counter() - worker_batch.start_time
        if self._observability_options.trace_enabled:
            self._trace_events.append(
                _create_trace_event(
                    worker_batch.block_id,
                    "block_completed",
                    worker_batch.start_time,
                    row_count=0,
                    worker_context=worker_batch.worker_context,
                    details={"empty_batch": True},
                )
            )
        _record_worker_metrics(
            self._metrics_collector,
            RayWorkerMetrics(
                total_rows=0,
                input_rows=0,
                output_rows=0,
                empty_input=True,
                blocks=1,
                elapsed_seconds=elapsed_seconds,
                block_id=worker_batch.block_id,
            ),
        )
        _record_worker_observability(
            self._metrics_collector,
            worker_profile=(
                _profile_worker_output(worker_batch.dataframe, block_id=worker_batch.block_id, model_usage=None)
                if self._observability_options.profile_workers
                else None
            ),
            trace_events=self._trace_events,
            throttle_snapshots=[],
        )

    def record_success(
        self,
        worker_batch: _RayWorkerBatch,
        generation_result: _RayWorkerGenerationResult,
        *,
        throttle_manager: Any | None,
    ) -> None:
        elapsed_seconds = time.perf_counter() - worker_batch.start_time
        block_result = generation_result.block_result
        if self._observability_options.trace_enabled:
            self._trace_events.extend(
                _task_traces_to_events(
                    block_result.task_traces,
                    block_id=worker_batch.block_id,
                    worker_context=worker_batch.worker_context,
                )
            )
            self._trace_events.append(
                _create_trace_event(
                    worker_batch.block_id,
                    "block_completed",
                    worker_batch.start_time,
                    row_count=len(generation_result.dataframe),
                    worker_context=worker_batch.worker_context,
                    details={
                        "output_columns": [str(column) for column in generation_result.dataframe.columns],
                        "input_rows": block_result.input_rows,
                        "output_rows": block_result.output_rows,
                        "dropped_rows": block_result.dropped_rows,
                    },
                )
            )
        _record_worker_metrics(
            self._metrics_collector,
            _metrics_from_block_result(
                block_result,
                elapsed_seconds=elapsed_seconds,
                block_id=worker_batch.block_id,
            ),
        )
        _record_worker_observability(
            self._metrics_collector,
            worker_profile=(
                _profile_worker_output(
                    generation_result.dataframe,
                    block_id=worker_batch.block_id,
                    model_usage=generation_result.model_usage,
                )
                if self._observability_options.profile_workers
                else None
            ),
            trace_events=self._trace_events,
            throttle_snapshots=_snapshot_worker_throttle(throttle_manager),
        )

    def record_all_rows_dropped(self, worker_batch: _RayWorkerBatch, block_result: BlockExecutionResult) -> None:
        elapsed_seconds = time.perf_counter() - worker_batch.start_time
        if self._observability_options.trace_enabled:
            self._trace_events.append(
                _create_trace_event(
                    worker_batch.block_id,
                    "block_failed",
                    worker_batch.start_time,
                    row_count=worker_batch.num_records,
                    worker_context=worker_batch.worker_context,
                    details={
                        "error_type": "AllRowsDropped",
                        "error": "all input rows were dropped",
                        "input_rows": block_result.input_rows,
                        "output_rows": block_result.output_rows,
                        "dropped_rows": block_result.dropped_rows,
                    },
                )
            )
        _record_worker_metrics(
            self._metrics_collector,
            _metrics_from_block_result(
                block_result,
                elapsed_seconds=elapsed_seconds,
                block_id=worker_batch.block_id,
                failed_blocks=1,
            ),
        )
        _record_worker_observability(
            self._metrics_collector,
            worker_profile=None,
            trace_events=self._trace_events,
            throttle_snapshots=[],
        )

    def record_failure(self, worker_batch: _RayWorkerBatch, exc: Exception) -> None:
        if self._observability_options.trace_enabled:
            self._trace_events.append(
                _create_trace_event(
                    worker_batch.block_id,
                    "block_failed",
                    worker_batch.start_time,
                    row_count=worker_batch.num_records,
                    worker_context=worker_batch.worker_context,
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )
        _record_worker_metrics(
            self._metrics_collector,
            RayWorkerMetrics(
                total_rows=0,
                input_rows=worker_batch.num_records,
                output_rows=0,
                blocks=1,
                failed_blocks=1,
                elapsed_seconds=time.perf_counter() - worker_batch.start_time,
                block_id=worker_batch.block_id,
            ),
        )
        _record_worker_observability(
            self._metrics_collector,
            worker_profile=None,
            trace_events=self._trace_events,
            throttle_snapshots=[],
        )


def _prepare_worker_options(
    execution_payload: _RayExecutionPayload,
    *,
    observability_options: _RayObservabilityOptions,
) -> _RayWorkerOptions:
    worker_options = execution_payload.worker_options
    run_config = copy.deepcopy(worker_options.run_config)
    if observability_options.trace_enabled:
        run_config.async_trace = True
    seed_readers = copy.deepcopy(worker_options.seed_readers)
    if execution_payload.use_input_dataset:
        seed_readers = ray_seed_planning.ensure_dataframe_seed_reader(seed_readers)
    return replace(
        worker_options,
        seed_readers=seed_readers,
        run_config=run_config,
    )


def _compile_ray_execution_payload(
    *,
    config_builder: DataDesignerConfigBuilder,
    worker_options: _RayWorkerOptions,
    use_input_dataset: bool,
    seed_window: ray_seed_planning.RaySeedWindow | None = None,
    hidden_order_column: str | None = None,
    capture_artifacts: bool = False,
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
        capture_artifacts=capture_artifacts,
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
    elif ray_seed_planning.RAY_RANGE_ID_COLUMN in dataframe.columns:
        values = dataframe[ray_seed_planning.RAY_RANGE_ID_COLUMN].tolist()
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


def _metrics_from_block_result(
    block_result: BlockExecutionResult,
    *,
    elapsed_seconds: float,
    block_id: str,
    failed_blocks: int = 0,
) -> RayWorkerMetrics:
    return RayWorkerMetrics(
        total_rows=block_result.output_rows,
        input_rows=block_result.input_rows,
        output_rows=block_result.output_rows,
        dropped_rows=block_result.dropped_rows,
        all_rows_dropped=block_result.all_rows_dropped,
        partial_rows_dropped=block_result.partial_rows_dropped,
        blocks=1,
        failed_blocks=failed_blocks,
        elapsed_seconds=elapsed_seconds,
        model_usage=block_result.model_usage_stats or None,
        block_id=block_id,
    )


def _format_worker_failure_message(*, dataframe: Any, exc: Exception) -> str:
    context_parts = [f"{len(dataframe)} input row(s)"]
    if _RAY_INTERNAL_ROW_ID_COLUMN in dataframe.columns:
        row_ids = [int(value) for value in dataframe[_RAY_INTERNAL_ROW_ID_COLUMN].tolist()]
        if row_ids:
            context_parts.append(f"logical rows {min(row_ids)}-{max(row_ids)}")
    elif ray_seed_planning.RAY_RANGE_ID_COLUMN in dataframe.columns:
        row_ids = [int(value) for value in dataframe[ray_seed_planning.RAY_RANGE_ID_COLUMN].tolist()]
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


def _coerce_pandas_dataframe(batch: Any) -> Any:
    if isinstance(batch, lazy.pd.DataFrame):
        return batch
    if isinstance(batch, dict):
        return lazy.pd.DataFrame(batch)
    return lazy.pd.DataFrame(batch)
