# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from data_designer.integrations.ray._validation import validate_finite_number
from data_designer.integrations.ray.errors import RayMetricsError

ModelUsageSummary = dict[str, dict[str, Any]]
RayTraceEventPayload: TypeAlias = "RayTraceEvent | Mapping[str, Any]"
RayWorkerProfilePayload: TypeAlias = "RayWorkerProfile | Mapping[str, Any]"
RayThrottleSnapshotPayload: TypeAlias = "RayThrottleSnapshot | Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class RayTraceEvent:
    """Bounded execution trace event emitted by a Ray worker."""

    block_id: str
    event_type: str
    timestamp_seconds: float
    elapsed_seconds: float = 0.0
    row_count: int = 0
    worker_hostname: str | None = None
    worker_pid: int | None = None
    ray_task_id: str | None = None
    ray_node_id: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_str("block_id", self.block_id)
        _validate_non_empty_str("event_type", self.event_type)
        _validate_non_negative_float("timestamp_seconds", self.timestamp_seconds)
        _validate_non_negative_float("elapsed_seconds", self.elapsed_seconds)
        _validate_non_negative_int("row_count", self.row_count)
        if self.worker_pid is not None:
            _validate_non_negative_int("worker_pid", self.worker_pid)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RayWorkerProfile:
    """Small per-block profiler summary that does not materialize the full dataset on the driver."""

    block_id: str
    total_rows: int = 0
    columns: list[str] = field(default_factory=list)
    column_dtypes: dict[str, str] = field(default_factory=dict)
    non_null_counts: dict[str, int] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    memory_usage_bytes: int | None = None
    model_usage: ModelUsageSummary | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_non_empty_str("block_id", self.block_id)
        _validate_non_negative_int("total_rows", self.total_rows)
        if self.memory_usage_bytes is not None:
            _validate_non_negative_int("memory_usage_bytes", self.memory_usage_bytes)
        for column in self.columns:
            _validate_non_empty_str("columns item", column)
        for field_name, counts in (("non_null_counts", self.non_null_counts), ("null_counts", self.null_counts)):
            for column_name, value in counts.items():
                _validate_non_empty_str(f"{field_name} key", column_name)
                _validate_non_negative_int(f"{field_name}[{column_name!r}]", value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RayThrottleSnapshot:
    """Worker-local throttle state snapshot for provider throttling diagnosis."""

    provider_name: str
    model_id: str
    domain: str
    current_limit: int
    effective_max: int | None = None
    in_flight: int = 0
    waiters: int = 0
    rate_limit_ceiling: int = 0
    consecutive_rate_limits: int = 0

    def __post_init__(self) -> None:
        _validate_non_empty_str("provider_name", self.provider_name)
        _validate_non_empty_str("model_id", self.model_id)
        _validate_non_empty_str("domain", self.domain)
        _validate_non_negative_int("current_limit", self.current_limit)
        if self.effective_max is not None:
            _validate_non_negative_int("effective_max", self.effective_max)
        _validate_non_negative_int("in_flight", self.in_flight)
        _validate_non_negative_int("waiters", self.waiters)
        _validate_non_negative_int("rate_limit_ceiling", self.rate_limit_ceiling)
        _validate_non_negative_int("consecutive_rate_limits", self.consecutive_rate_limits)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RayDatasetAnalysis:
    """Ray-native analysis artifact composed from bounded worker summaries."""

    total_rows: int = 0
    blocks: int = 0
    failed_blocks: int = 0
    worker_profiles: list[RayWorkerProfile] = field(default_factory=list)
    trace_events: list[RayTraceEvent] = field(default_factory=list)
    trace_events_dropped: int = 0
    throttle_snapshots: list[RayThrottleSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_non_negative_int("total_rows", self.total_rows)
        _validate_non_negative_int("blocks", self.blocks)
        _validate_non_negative_int("failed_blocks", self.failed_blocks)
        _validate_non_negative_int("trace_events_dropped", self.trace_events_dropped)

    @property
    def successful_blocks(self) -> int:
        """Return completed block count after failed blocks are subtracted."""
        return max(self.blocks - self.failed_blocks, 0)

    @property
    def column_names(self) -> list[str]:
        """Return sorted unique column names observed across worker profiles."""
        return sorted({column for profile in self.worker_profiles for column in profile.columns})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def to_report(self, save_path: str | Path | None = None, include_sections: list[Any] | None = None) -> None:
        """Write a compact JSON or HTML report for Ray-native analysis."""
        del include_sections
        if save_path is None:
            return
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path.suffix.lower() in {".html", ".htm"}:
            path.write_text(_analysis_payload_to_html(payload), encoding="utf-8")
            return
        path.write_text(f"{payload}\n", encoding="utf-8")


def normalize_ray_trace_event(payload: RayTraceEventPayload) -> RayTraceEvent:
    """Normalize a trace dataclass or mapping payload."""
    if isinstance(payload, RayTraceEvent):
        return payload
    if not isinstance(payload, Mapping):
        raise RayMetricsError(f"Ray trace event payload must be a mapping, got {type(payload)!r}.")
    return RayTraceEvent(
        block_id=_coerce_str(payload, "block_id"),
        event_type=_coerce_str(payload, "event_type"),
        timestamp_seconds=_coerce_float(payload, "timestamp_seconds", default=0.0),
        elapsed_seconds=_coerce_float(payload, "elapsed_seconds", default=0.0),
        row_count=_coerce_int(payload, "row_count", default=0),
        worker_hostname=_coerce_optional_str(payload.get("worker_hostname")),
        worker_pid=_coerce_optional_int(payload.get("worker_pid"), "worker_pid"),
        ray_task_id=_coerce_optional_str(payload.get("ray_task_id")),
        ray_node_id=_coerce_optional_str(payload.get("ray_node_id")),
        details=_coerce_optional_mapping(payload.get("details")),
    )


def normalize_ray_worker_profile(payload: RayWorkerProfilePayload) -> RayWorkerProfile:
    """Normalize a worker profile dataclass or mapping payload."""
    if isinstance(payload, RayWorkerProfile):
        return payload
    if not isinstance(payload, Mapping):
        raise RayMetricsError(f"Ray worker profile payload must be a mapping, got {type(payload)!r}.")
    return RayWorkerProfile(
        block_id=_coerce_str(payload, "block_id"),
        total_rows=_coerce_int(payload, "total_rows", default=0),
        columns=_coerce_str_list(payload.get("columns")),
        column_dtypes=_coerce_str_mapping(payload.get("column_dtypes")),
        non_null_counts=_coerce_int_mapping(payload.get("non_null_counts")),
        null_counts=_coerce_int_mapping(payload.get("null_counts")),
        memory_usage_bytes=_coerce_optional_int(payload.get("memory_usage_bytes"), "memory_usage_bytes"),
        model_usage=_coerce_model_usage(payload.get("model_usage")),
        warnings=_coerce_str_list(payload.get("warnings")),
    )


def normalize_ray_throttle_snapshot(payload: RayThrottleSnapshotPayload) -> RayThrottleSnapshot:
    """Normalize a throttle snapshot dataclass or mapping payload."""
    if isinstance(payload, RayThrottleSnapshot):
        return payload
    if not isinstance(payload, Mapping):
        raise RayMetricsError(f"Ray throttle snapshot payload must be a mapping, got {type(payload)!r}.")
    return RayThrottleSnapshot(
        provider_name=_coerce_str(payload, "provider_name"),
        model_id=_coerce_str(payload, "model_id"),
        domain=_coerce_str(payload, "domain"),
        current_limit=_coerce_int(payload, "current_limit", default=0),
        effective_max=_coerce_optional_int(payload.get("effective_max"), "effective_max"),
        in_flight=_coerce_int(payload, "in_flight", default=0),
        waiters=_coerce_int(payload, "waiters", default=0),
        rate_limit_ceiling=_coerce_int(payload, "rate_limit_ceiling", default=0),
        consecutive_rate_limits=_coerce_int(payload, "consecutive_rate_limits", default=0),
    )


def _analysis_payload_to_html(payload: str) -> str:
    escaped = html.escape(payload)
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>Data Designer Ray Analysis</title></head>'
        "<body><h1>Data Designer Ray Analysis</h1><pre>"
        f"{escaped}"
        "</pre></body></html>\n"
    )


def _coerce_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be a string.")
    return value


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RayMetricsError("Ray observability optional string fields must be strings when provided.")
    return value


def _coerce_int(payload: Mapping[str, Any], field_name: str, *, default: int) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be an integer.")
    return value


def _coerce_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be an integer when provided.")
    return value


def _coerce_float(payload: Mapping[str, Any], field_name: str, *, default: float) -> float:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label="Ray observability field",
    )
    return float(value)


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RayMetricsError("Ray observability list fields must be lists when provided.")
    if not all(isinstance(item, str) for item in value):
        raise RayMetricsError("Ray observability list field values must be strings.")
    return list(value)


def _coerce_str_mapping(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RayMetricsError("Ray observability string mapping fields must be mappings when provided.")
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RayMetricsError("Ray observability string mapping keys and values must be strings.")
        output[key] = item
    return output


def _coerce_int_mapping(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RayMetricsError("Ray observability count mapping fields must be mappings when provided.")
    output: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int):
            raise RayMetricsError("Ray observability count mapping keys must be strings and values must be integers.")
        output[key] = item
    return output


def _coerce_optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RayMetricsError("Ray observability details must be a mapping when provided.")
    return dict(value)


def _coerce_model_usage(value: Any) -> ModelUsageSummary | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RayMetricsError("Ray observability model_usage must be a mapping when provided.")
    model_usage: ModelUsageSummary = {}
    for model_name, stats in value.items():
        if not isinstance(model_name, str):
            raise RayMetricsError("Ray observability model usage keys must be strings.")
        if not isinstance(stats, Mapping):
            raise RayMetricsError(f"Ray observability model usage for {model_name!r} must be a mapping.")
        model_usage[model_name] = dict(stats)
    return model_usage


def _validate_non_empty_str(field_name: str, value: str) -> None:
    if not isinstance(value, str) or value == "":
        raise RayMetricsError(f"Ray observability field {field_name!r} must be a non-empty string.")


def _validate_non_negative_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be an integer.")
    if value < 0:
        raise RayMetricsError(f"Ray observability field {field_name!r} must be non-negative.")


def _validate_non_negative_float(field_name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"Ray observability field {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label="Ray observability field",
    )
    if value < 0:
        raise RayMetricsError(f"Ray observability field {field_name!r} must be non-negative.")
