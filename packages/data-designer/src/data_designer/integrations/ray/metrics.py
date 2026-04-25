# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

from data_designer.integrations.ray._validation import validate_finite_number
from data_designer.integrations.ray.errors import RayMetricsError

ModelUsageSummary = dict[str, dict[str, Any]]
RayWorkerMetricsPayload: TypeAlias = "RayWorkerMetrics | Mapping[str, Any]"
_DATASET_METRICS_ONLY_FIELDS = frozenset(
    {
        "all_rows_dropped_blocks",
        "partial_rows_dropped_blocks",
        "empty_input_blocks",
        "worker_elapsed_seconds",
        "throttle",
    }
)


@dataclass(frozen=True, slots=True)
class RayWorkerMetrics:
    """Serializable metrics emitted by one Ray worker or processed Ray block."""

    total_rows: int = 0
    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    all_rows_dropped: bool = False
    partial_rows_dropped: bool = False
    empty_input: bool = False
    blocks: int = 1
    failed_blocks: int = 0
    elapsed_seconds: float = 0.0
    model_usage: ModelUsageSummary | None = None
    block_id: str | None = None

    def __post_init__(self) -> None:
        if self.output_rows == 0 and self.total_rows > 0:
            object.__setattr__(self, "output_rows", self.total_rows)
        _validate_non_negative_int("total_rows", self.total_rows)
        _validate_non_negative_int("input_rows", self.input_rows)
        _validate_non_negative_int("output_rows", self.output_rows)
        _validate_non_negative_int("dropped_rows", self.dropped_rows)
        _validate_bool("all_rows_dropped", self.all_rows_dropped)
        _validate_bool("partial_rows_dropped", self.partial_rows_dropped)
        _validate_bool("empty_input", self.empty_input)
        _validate_non_negative_int("blocks", self.blocks)
        _validate_non_negative_int("failed_blocks", self.failed_blocks)
        _validate_failed_blocks_not_greater_than_blocks(blocks=self.blocks, failed_blocks=self.failed_blocks)
        _validate_non_negative_float("elapsed_seconds", self.elapsed_seconds)
        if self.block_id is not None and not isinstance(self.block_id, str):
            raise RayMetricsError("Ray metrics field 'block_id' must be a string when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RayDatasetMetrics:
    """Serializable summary metrics for RayDatasetCreationResults.

    total_rows is populated only when the backend knows the count without
    forcing Ray Dataset materialization on the driver.
    elapsed_seconds is driver-observed wall-clock duration. worker_elapsed_seconds
    is the cumulative elapsed duration reported by worker/block payloads.
    Aggregated model usage rates use worker_elapsed_seconds when available.
    """

    total_rows: int = 0
    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    all_rows_dropped_blocks: int = 0
    partial_rows_dropped_blocks: int = 0
    empty_input_blocks: int = 0
    blocks: int = 0
    failed_blocks: int = 0
    elapsed_seconds: float = 0.0
    worker_elapsed_seconds: float = 0.0
    model_usage: ModelUsageSummary | None = None
    throttle: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.output_rows == 0 and self.total_rows > 0:
            object.__setattr__(self, "output_rows", self.total_rows)
        _validate_non_negative_int("total_rows", self.total_rows)
        _validate_non_negative_int("input_rows", self.input_rows)
        _validate_non_negative_int("output_rows", self.output_rows)
        _validate_non_negative_int("dropped_rows", self.dropped_rows)
        _validate_non_negative_int("all_rows_dropped_blocks", self.all_rows_dropped_blocks)
        _validate_non_negative_int("partial_rows_dropped_blocks", self.partial_rows_dropped_blocks)
        _validate_non_negative_int("empty_input_blocks", self.empty_input_blocks)
        _validate_non_negative_int("blocks", self.blocks)
        _validate_non_negative_int("failed_blocks", self.failed_blocks)
        _validate_failed_blocks_not_greater_than_blocks(blocks=self.blocks, failed_blocks=self.failed_blocks)
        _validate_non_negative_float("elapsed_seconds", self.elapsed_seconds)
        _validate_non_negative_float("worker_elapsed_seconds", self.worker_elapsed_seconds)

    @property
    def successful_blocks(self) -> int:
        """Return completed block count after failed blocks are subtracted."""
        return self.blocks - self.failed_blocks

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class _MetricsAccumulator:
    total_rows: int = 0
    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    all_rows_dropped_blocks: int = 0
    partial_rows_dropped_blocks: int = 0
    empty_input_blocks: int = 0
    blocks: int = 0
    failed_blocks: int = 0
    worker_elapsed_seconds: float = 0.0
    model_usage: ModelUsageSummary = field(default_factory=dict)


def aggregate_ray_metrics(
    worker_metrics: Iterable[RayWorkerMetricsPayload],
    *,
    elapsed_seconds: float = 0.0,
) -> RayDatasetMetrics:
    """Aggregate worker or block metrics into a dataset-level summary.

    The helper accepts plain mappings so Ray workers can return JSON-like
    payloads without importing Data Designer integration classes on the driver.
    Model usage is expected to follow Data Designer's model usage summary shape;
    token and request rates are recomputed from aggregate counters using
    cumulative worker elapsed time when worker timings are available.
    """
    accumulator = _MetricsAccumulator()
    for payload in worker_metrics:
        metrics = normalize_ray_worker_metrics(payload)
        accumulator.total_rows += metrics.total_rows
        accumulator.input_rows += metrics.input_rows
        accumulator.output_rows += metrics.output_rows
        accumulator.dropped_rows += metrics.dropped_rows
        accumulator.all_rows_dropped_blocks += int(metrics.all_rows_dropped)
        accumulator.partial_rows_dropped_blocks += int(metrics.partial_rows_dropped)
        accumulator.empty_input_blocks += int(metrics.empty_input)
        accumulator.blocks += metrics.blocks
        accumulator.failed_blocks += metrics.failed_blocks
        accumulator.worker_elapsed_seconds += metrics.elapsed_seconds
        if metrics.model_usage:
            _merge_model_usage(accumulator.model_usage, metrics.model_usage)

    model_usage = accumulator.model_usage or None
    if model_usage is not None:
        _recompute_model_usage_rates(
            model_usage,
            _model_usage_rate_elapsed_seconds(
                elapsed_seconds=elapsed_seconds,
                worker_elapsed_seconds=accumulator.worker_elapsed_seconds,
            ),
        )

    return RayDatasetMetrics(
        total_rows=accumulator.total_rows,
        input_rows=accumulator.input_rows,
        output_rows=accumulator.output_rows,
        dropped_rows=accumulator.dropped_rows,
        all_rows_dropped_blocks=accumulator.all_rows_dropped_blocks,
        partial_rows_dropped_blocks=accumulator.partial_rows_dropped_blocks,
        empty_input_blocks=accumulator.empty_input_blocks,
        blocks=accumulator.blocks,
        failed_blocks=accumulator.failed_blocks,
        elapsed_seconds=elapsed_seconds,
        worker_elapsed_seconds=accumulator.worker_elapsed_seconds,
        model_usage=model_usage,
    )


def normalize_ray_worker_metrics(payload: RayWorkerMetricsPayload) -> RayWorkerMetrics:
    """Normalize a dataclass or mapping payload into RayWorkerMetrics."""
    if isinstance(payload, RayDatasetMetrics):
        raise RayMetricsError(
            "RayDatasetMetrics payloads are dataset-level summaries and cannot be normalized as worker metrics."
        )
    if isinstance(payload, RayWorkerMetrics):
        return payload
    if not isinstance(payload, Mapping):
        raise RayMetricsError(
            f"Ray metrics payload must be a mapping or RayWorkerMetrics dataclass, got {type(payload)!r}."
        )
    _reject_dataset_metrics_mapping(payload)

    return RayWorkerMetrics(
        total_rows=_coerce_int(payload, "total_rows", default=0),
        input_rows=_coerce_int(payload, "input_rows", default=0),
        output_rows=_coerce_int(payload, "output_rows", default=_coerce_int(payload, "total_rows", default=0)),
        dropped_rows=_coerce_int(payload, "dropped_rows", default=0),
        all_rows_dropped=_coerce_bool(payload, "all_rows_dropped", default=False),
        partial_rows_dropped=_coerce_bool(payload, "partial_rows_dropped", default=False),
        empty_input=_coerce_bool(payload, "empty_input", default=False),
        blocks=_coerce_int(payload, "blocks", default=1),
        failed_blocks=_coerce_int(payload, "failed_blocks", default=0),
        elapsed_seconds=_coerce_float(payload, "elapsed_seconds", default=0.0),
        model_usage=_coerce_model_usage(payload.get("model_usage")),
        block_id=_coerce_optional_str(payload.get("block_id")),
    )


def _coerce_int(payload: Mapping[str, Any], field_name: str, *, default: int) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be an integer.")
    return value


def _coerce_float(payload: Mapping[str, Any], field_name: str, *, default: float) -> float:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label="Ray metrics field",
    )
    return float(value)


def _coerce_bool(payload: Mapping[str, Any], field_name: str, *, default: bool) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be a boolean.")
    return value


def _coerce_model_usage(value: Any) -> ModelUsageSummary | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RayMetricsError("Ray metrics field 'model_usage' must be a mapping when provided.")

    model_usage: ModelUsageSummary = {}
    for model_name, stats in value.items():
        if not isinstance(model_name, str):
            raise RayMetricsError("Ray metrics model usage keys must be strings.")
        if not isinstance(stats, Mapping):
            raise RayMetricsError(f"Ray metrics model usage for {model_name!r} must be a mapping.")
        model_usage[model_name] = dict(stats)
    return model_usage


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RayMetricsError("Ray metrics field 'block_id' must be a string when provided.")
    return value


def _validate_non_negative_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be an integer.")
    if value < 0:
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be non-negative.")


def _validate_non_negative_float(field_name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label="Ray metrics field",
    )
    if value < 0:
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be non-negative.")


def _validate_bool(field_name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise RayMetricsError(f"Ray metrics field {field_name!r} must be a boolean.")


def _validate_failed_blocks_not_greater_than_blocks(*, blocks: int, failed_blocks: int) -> None:
    if failed_blocks > blocks:
        raise RayMetricsError("Ray metrics field 'failed_blocks' cannot be greater than 'blocks'.")


def _reject_dataset_metrics_mapping(payload: Mapping[str, Any]) -> None:
    dataset_fields = sorted(_DATASET_METRICS_ONLY_FIELDS.intersection(payload))
    if dataset_fields:
        field_list = ", ".join(repr(field) for field in dataset_fields)
        raise RayMetricsError(
            "Ray worker metrics payload contains dataset-level field(s) "
            f"{field_list}; dataset metrics cannot be normalized as worker metrics."
        )


def _merge_model_usage(target: ModelUsageSummary, source: ModelUsageSummary) -> None:
    for model_name, stats in source.items():
        target_stats = target.setdefault(model_name, {})
        _merge_stats_mapping(target_stats, stats)


def _merge_stats_mapping(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in {"tokens_per_second", "requests_per_minute"}:
            continue
        if isinstance(value, Mapping):
            nested_target = target.get(key)
            if not isinstance(nested_target, dict):
                nested_target = {}
                target[key] = nested_target
            _merge_stats_mapping(nested_target, value)
        elif isinstance(value, bool):
            target[key] = value
        elif isinstance(value, (int, float)):
            existing_value = target.get(key, 0)
            target[key] = existing_value + value if isinstance(existing_value, (int, float)) else value
        else:
            target[key] = value


def _recompute_model_usage_rates(model_usage: ModelUsageSummary, elapsed_seconds: float) -> None:
    for stats in model_usage.values():
        token_usage = stats.get("token_usage")
        if isinstance(token_usage, dict):
            input_tokens = _numeric_value(token_usage.get("input_tokens"))
            output_tokens = _numeric_value(token_usage.get("output_tokens"))
            total_tokens = input_tokens + output_tokens
            token_usage["total_tokens"] = total_tokens
            stats["tokens_per_second"] = int(total_tokens / elapsed_seconds) if elapsed_seconds > 0 else 0

        request_usage = stats.get("request_usage")
        if isinstance(request_usage, dict):
            successful_requests = _numeric_value(request_usage.get("successful_requests"))
            failed_requests = _numeric_value(request_usage.get("failed_requests"))
            total_requests = successful_requests + failed_requests
            request_usage["total_requests"] = total_requests
            stats["requests_per_minute"] = int(total_requests / elapsed_seconds * 60) if elapsed_seconds > 0 else 0


def _model_usage_rate_elapsed_seconds(*, elapsed_seconds: float, worker_elapsed_seconds: float) -> float:
    if worker_elapsed_seconds > 0:
        return worker_elapsed_seconds
    return elapsed_seconds


def _numeric_value(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value
