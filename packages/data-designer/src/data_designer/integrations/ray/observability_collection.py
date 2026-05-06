# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from data_designer.integrations.ray.metrics import RayDatasetMetrics, RayWorkerMetrics
from data_designer.integrations.ray.observability import (
    RayDatasetAnalysis,
    RayDatasetStats,
    RayThrottleSnapshot,
    RayTraceEvent,
    RayWorkerProfile,
    normalize_ray_dataset_stats,
    normalize_ray_throttle_snapshot,
    normalize_ray_trace_event,
    normalize_ray_worker_profile,
)

_MAX_RAY_DATASET_STATS_CHARS = 65_536
_MAX_RAY_DATASET_STATS_DIAGNOSTIC_LINES = 100
_MAX_RAY_DATASET_STATS_DIAGNOSTIC_LINE_CHARS = 500
_MAX_RAY_DIAGNOSTIC_WARNINGS = 20
_RAY_OPERATOR_DIAGNOSTIC_MARKERS = (
    "operator",
    "map",
    "read",
    "write",
    "repartition",
    "sort",
    "filter",
    "limit",
    "union",
    "zip",
    "join",
    "block",
    "rows",
    "task",
    "wall time",
    "cpu time",
    "udf",
)
_RAY_BACKPRESSURE_DIAGNOSTIC_MARKERS = (
    "backpressure",
    "backpressured",
    "queue",
    "queued",
    "pending",
    "blocked",
    "wait",
    "stall",
    "throttle",
)
_RAY_OBJECT_STORE_DIAGNOSTIC_MARKERS = (
    "object store",
    "object_store",
    "objectref",
    "plasma",
    "spill",
    "spilled",
    "memory",
    "bytes",
    "mib",
    "gib",
)


@dataclass(frozen=True)
class _RayObservabilityOptions:
    profile_workers: bool = False
    trace_enabled: bool = False
    max_worker_profiles: int = 1000
    max_throttle_snapshots: int = 1000


class _RayMetricsCollector:
    def __init__(
        self,
        max_trace_events: int = 1000,
        max_worker_profiles: int = 1000,
        max_throttle_snapshots: int = 1000,
    ) -> None:
        self._payloads: list[dict[str, Any]] = []
        self._worker_profiles: list[dict[str, Any]] = []
        self._worker_profiles_dropped = 0
        self._trace_events: list[dict[str, Any]] = []
        self._trace_events_dropped = 0
        self._throttle_snapshots: list[dict[str, Any]] = []
        self._throttle_snapshots_dropped = 0
        self._max_trace_events = max_trace_events
        self._max_worker_profiles = max_worker_profiles
        self._max_throttle_snapshots = max_throttle_snapshots

    def record(self, payload: dict[str, Any]) -> None:
        self._payloads.append(payload)

    def record_observability(self, payload: dict[str, Any]) -> None:
        profile = payload.get("worker_profile")
        if isinstance(profile, dict):
            if len(self._worker_profiles) >= self._max_worker_profiles:
                self._worker_profiles_dropped += 1
            else:
                self._worker_profiles.append(profile)
        for event in payload.get("trace_events", []) or []:
            if not isinstance(event, dict):
                continue
            if len(self._trace_events) >= self._max_trace_events:
                self._trace_events_dropped += 1
                continue
            self._trace_events.append(event)
        for snapshot in payload.get("throttle_snapshots", []) or []:
            if not isinstance(snapshot, dict):
                continue
            if len(self._throttle_snapshots) >= self._max_throttle_snapshots:
                self._throttle_snapshots_dropped += 1
                continue
            self._throttle_snapshots.append(snapshot)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._payloads)

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "worker_profiles": list(self._worker_profiles),
            "worker_profiles_dropped": self._worker_profiles_dropped,
            "trace_events": list(self._trace_events),
            "trace_events_dropped": self._trace_events_dropped,
            "throttle_snapshots": list(self._throttle_snapshots),
            "throttle_snapshots_dropped": self._throttle_snapshots_dropped,
        }


def _create_metrics_collector(
    ray: Any,
    *,
    max_trace_events: int = 1000,
    max_worker_profiles: int = 1000,
    max_throttle_snapshots: int = 1000,
) -> Any | None:
    remote = getattr(ray, "remote", None)
    if not callable(remote):
        return None
    return remote(_RayMetricsCollector).remote(
        max_trace_events=max_trace_events,
        max_worker_profiles=max_worker_profiles,
        max_throttle_snapshots=max_throttle_snapshots,
    )


def assemble_ray_dataset_analysis(
    metrics: RayDatasetMetrics,
    payload: Mapping[str, Any],
    *,
    ray_dataset_stats: RayDatasetStats | Mapping[str, Any] | None = None,
    diagnostic_warnings: list[str] | None = None,
) -> RayDatasetAnalysis | None:
    warnings = _bounded_warnings(diagnostic_warnings or [])
    _extend_payload_warnings(payload.get("diagnostic_warnings"), warnings)
    dataset_stats = _normalize_dataset_stats(
        ray_dataset_stats if ray_dataset_stats is not None else payload.get("ray_dataset_stats"),
        warnings,
    )
    if not _has_observability_payload(payload) and dataset_stats is None and not warnings:
        return None
    worker_profiles, invalid_worker_profiles = _normalize_observability_items(
        payload.get("worker_profiles"),
        normalize_ray_worker_profile,
        "worker profile",
        warnings,
    )
    trace_events, invalid_trace_events = _normalize_observability_items(
        payload.get("trace_events"),
        normalize_ray_trace_event,
        "trace event",
        warnings,
    )
    throttle_snapshots, invalid_throttle_snapshots = _normalize_observability_items(
        payload.get("throttle_snapshots"),
        normalize_ray_throttle_snapshot,
        "throttle snapshot",
        warnings,
    )
    return RayDatasetAnalysis(
        total_rows=metrics.total_rows,
        blocks=metrics.blocks,
        failed_blocks=metrics.failed_blocks,
        worker_profiles=worker_profiles,
        worker_profiles_dropped=_safe_non_negative_payload_int(payload, "worker_profiles_dropped", warnings)
        + invalid_worker_profiles,
        trace_events=trace_events,
        trace_events_dropped=_safe_non_negative_payload_int(payload, "trace_events_dropped", warnings)
        + invalid_trace_events,
        throttle_snapshots=throttle_snapshots,
        throttle_snapshots_dropped=_safe_non_negative_payload_int(payload, "throttle_snapshots_dropped", warnings)
        + invalid_throttle_snapshots,
        ray_dataset_stats=dataset_stats,
        diagnostic_warnings=warnings,
    )


def _has_observability_payload(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("worker_profiles")
        or payload.get("worker_profiles_dropped")
        or payload.get("trace_events")
        or payload.get("trace_events_dropped")
        or payload.get("throttle_snapshots")
        or payload.get("throttle_snapshots_dropped")
        or payload.get("ray_dataset_stats")
        or payload.get("diagnostic_warnings")
    )


def collect_ray_dataset_stats(dataset: Any | None) -> RayDatasetStats | None:
    """Collect bounded Ray Dataset.stats() output without materializing rows on the driver."""
    if dataset is None:
        return None
    stats = getattr(dataset, "stats", None)
    if not callable(stats):
        return None

    try:
        raw_stats = stats()
    except Exception as exc:
        return RayDatasetStats(warnings=[_format_collection_warning("Ray Dataset.stats() failed", exc)])

    stats_text, warnings = _coerce_ray_dataset_stats_text(raw_stats)
    if stats_text is None:
        return RayDatasetStats(warnings=warnings) if warnings else None

    bounded_stats_text = stats_text[:_MAX_RAY_DATASET_STATS_CHARS]
    stats_text_truncated = len(stats_text) > _MAX_RAY_DATASET_STATS_CHARS
    operator_diagnostics, operator_diagnostics_dropped = _extract_ray_dataset_stats_diagnostics(
        stats_text,
        _RAY_OPERATOR_DIAGNOSTIC_MARKERS,
    )
    backpressure_diagnostics, backpressure_diagnostics_dropped = _extract_ray_dataset_stats_diagnostics(
        stats_text,
        _RAY_BACKPRESSURE_DIAGNOSTIC_MARKERS,
    )
    object_store_diagnostics, object_store_diagnostics_dropped = _extract_ray_dataset_stats_diagnostics(
        stats_text,
        _RAY_OBJECT_STORE_DIAGNOSTIC_MARKERS,
    )
    return RayDatasetStats(
        stats_text=bounded_stats_text,
        stats_text_truncated=stats_text_truncated,
        stats_text_char_count=len(stats_text),
        operator_diagnostics=operator_diagnostics,
        operator_diagnostics_dropped=operator_diagnostics_dropped,
        backpressure_diagnostics=backpressure_diagnostics,
        backpressure_diagnostics_dropped=backpressure_diagnostics_dropped,
        object_store_diagnostics=object_store_diagnostics,
        object_store_diagnostics_dropped=object_store_diagnostics_dropped,
        warnings=warnings,
    )


def _normalize_dataset_stats(
    payload: Any,
    warnings: list[str],
) -> RayDatasetStats | None:
    if payload is None:
        return None
    try:
        return normalize_ray_dataset_stats(payload)
    except Exception as exc:
        _append_warning(warnings, _format_collection_warning("Ray Dataset stats payload normalization failed", exc))
        return None


def _normalize_observability_items(
    raw_items: Any,
    normalizer: Callable[[Any], Any],
    item_label: str,
    warnings: list[str],
) -> tuple[list[Any], int]:
    if raw_items is None:
        return [], 0
    if not isinstance(raw_items, list):
        _append_warning(warnings, f"Ray observability {item_label} payload must be a list when provided.")
        return [], 1

    items: list[Any] = []
    invalid_items = 0
    for index, raw_item in enumerate(raw_items):
        try:
            items.append(normalizer(raw_item))
        except Exception as exc:
            invalid_items += 1
            _append_warning(
                warnings,
                _format_collection_warning(f"Skipped Ray observability {item_label} at index {index}", exc),
            )
    return items, invalid_items


def _safe_non_negative_payload_int(payload: Mapping[str, Any], field_name: str, warnings: list[str]) -> int:
    value = payload.get(field_name, 0) or 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _append_warning(warnings, f"Ray observability field {field_name!r} must be a non-negative integer.")
        return 0
    return value


def _extend_payload_warnings(raw_warnings: Any, warnings: list[str]) -> None:
    if raw_warnings is None:
        return
    if not isinstance(raw_warnings, list):
        _append_warning(warnings, "Ray observability diagnostic_warnings payload must be a list when provided.")
        return
    for warning in raw_warnings:
        if isinstance(warning, str) and warning:
            _append_warning(warnings, warning)
        else:
            _append_warning(warnings, "Ray observability diagnostic_warnings entries must be non-empty strings.")


def _coerce_ray_dataset_stats_text(raw_stats: Any) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if raw_stats is None:
        return None, ["Ray Dataset.stats() returned None."]
    if isinstance(raw_stats, str):
        return raw_stats, warnings
    if isinstance(raw_stats, bytes):
        return raw_stats.decode("utf-8", errors="replace"), [
            "Ray Dataset.stats() returned bytes; decoded it as UTF-8 with replacement."
        ]
    if isinstance(raw_stats, Mapping) or isinstance(raw_stats, (list, tuple)):
        return json.dumps(raw_stats, sort_keys=True, default=str), [
            f"Ray Dataset.stats() returned {type(raw_stats).__name__}; serialized it as JSON text."
        ]

    for method_name in ("to_string", "to_summary"):
        method = getattr(raw_stats, method_name, None)
        if not callable(method):
            continue
        try:
            converted = method()
        except Exception as exc:
            _append_warning(
                warnings,
                _format_collection_warning(f"Ray Dataset.stats() {method_name} conversion failed", exc),
            )
            continue
        if converted is raw_stats:
            continue
        converted_text, converted_warnings = _coerce_ray_dataset_stats_text(converted)
        return converted_text, _bounded_warnings([*warnings, *converted_warnings])

    _append_warning(
        warnings,
        f"Ray Dataset.stats() returned {type(raw_stats).__name__}; converted it with str().",
    )
    return str(raw_stats), warnings


def _extract_ray_dataset_stats_diagnostics(
    stats_text: str,
    markers: tuple[str, ...],
) -> tuple[list[str], int]:
    diagnostics: list[str] = []
    dropped = 0
    for raw_line in stats_text.splitlines():
        line = " ".join(raw_line.strip().split())
        if line == "":
            continue
        normalized_line = line.lower()
        if not any(marker in normalized_line for marker in markers):
            continue
        bounded_line = _bound_diagnostic_line(line)
        if len(diagnostics) >= _MAX_RAY_DATASET_STATS_DIAGNOSTIC_LINES:
            dropped += 1
            continue
        diagnostics.append(bounded_line)
    return diagnostics, dropped


def _bound_diagnostic_line(line: str) -> str:
    if len(line) <= _MAX_RAY_DATASET_STATS_DIAGNOSTIC_LINE_CHARS:
        return line
    return f"{line[:_MAX_RAY_DATASET_STATS_DIAGNOSTIC_LINE_CHARS]}... [truncated]"


def _append_warning(warnings: list[str], warning: str) -> None:
    if len(warnings) >= _MAX_RAY_DIAGNOSTIC_WARNINGS:
        return
    warnings.append(warning)


def _bounded_warnings(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if warning][:_MAX_RAY_DIAGNOSTIC_WARNINGS]


def _format_collection_warning(message: str, exc: Exception) -> str:
    return f"{message}: {type(exc).__name__}: {exc}"


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
    output: Any,
    *,
    block_id: str,
    model_usage: dict[str, dict[str, Any]] | None,
    input_frame: Any | None = None,
) -> RayWorkerProfile:
    warnings: list[str] = []
    process_maxrss_bytes = _process_maxrss_bytes(warnings)
    input_memory_usage_bytes = _dataframe_memory_usage_bytes(input_frame, warnings, label="input")
    try:
        columns = [str(column) for column in output.columns]
        column_dtypes = {str(column): str(dtype) for column, dtype in output.dtypes.items()}
        non_null_counts = {str(column): int(value) for column, value in output.notna().sum().to_dict().items()}
        null_counts = {str(column): int(value) for column, value in output.isna().sum().to_dict().items()}
        memory_usage_bytes = _dataframe_memory_usage_bytes(output, warnings, label="output")
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
        input_memory_usage_bytes=input_memory_usage_bytes,
        process_maxrss_bytes=process_maxrss_bytes,
        model_usage=model_usage,
        warnings=warnings,
    )


def _dataframe_memory_usage_bytes(dataframe: Any | None, warnings: list[str], *, label: str) -> int | None:
    if dataframe is None:
        return None
    try:
        memory_usage = getattr(dataframe, "memory_usage", None)
        if not callable(memory_usage):
            return None
        return int(memory_usage(deep=True).sum())
    except Exception as exc:
        warnings.append(f"Failed to profile Ray worker {label} memory: {type(exc).__name__}: {exc}")
        return None


def _process_maxrss_bytes(warnings: list[str]) -> int | None:
    try:
        resource_module = importlib.import_module("resource")
        value = int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)
    except Exception as exc:
        warnings.append(f"Failed to profile Ray worker process RSS: {type(exc).__name__}: {exc}")
        return None
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _snapshot_worker_throttle(throttle_manager: Any | None) -> list[RayThrottleSnapshot]:
    if throttle_manager is None:
        return []
    snapshot = getattr(throttle_manager, "snapshot", None)
    if not callable(snapshot):
        return []
    try:
        payload = snapshot()
    except Exception:
        return []
    if not isinstance(payload, Mapping):
        return []
    global_caps = _effective_max_by_throttle_key(payload.get("global_caps"))
    domains = payload.get("domains")
    if not isinstance(domains, list):
        return []
    snapshots: list[RayThrottleSnapshot] = []
    for domain_payload in domains:
        if not isinstance(domain_payload, Mapping):
            continue
        provider_name = domain_payload.get("provider_name")
        model_id = domain_payload.get("model_id")
        domain = domain_payload.get("domain")
        if not all(isinstance(value, str) for value in (provider_name, model_id, domain)):
            continue
        effective_max = _safe_optional_int_mapping(domain_payload, "effective_max")
        if effective_max is None:
            effective_max = global_caps.get((provider_name, model_id))
        snapshots.append(
            RayThrottleSnapshot(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                current_limit=_safe_int_mapping(domain_payload, "current_limit"),
                effective_max=effective_max,
                in_flight=_safe_int_mapping(domain_payload, "in_flight"),
                waiters=_safe_int_mapping(domain_payload, "waiters"),
                rate_limit_ceiling=_safe_int_mapping(domain_payload, "rate_limit_ceiling"),
                consecutive_rate_limits=_safe_int_mapping(
                    domain_payload,
                    "consecutive_rate_limits",
                    fallback_field_name="consecutive_429s",
                ),
            )
        )
    return snapshots


def _effective_max_by_throttle_key(global_caps: Any) -> dict[tuple[str, str], int]:
    if not isinstance(global_caps, list):
        return {}
    effective_by_key: dict[tuple[str, str], int] = {}
    for cap_payload in global_caps:
        if not isinstance(cap_payload, Mapping):
            continue
        provider_name = cap_payload.get("provider_name")
        model_id = cap_payload.get("model_id")
        effective_max = cap_payload.get("effective_max")
        if (
            isinstance(provider_name, str)
            and isinstance(model_id, str)
            and isinstance(effective_max, int)
            and not isinstance(effective_max, bool)
            and effective_max >= 0
        ):
            effective_by_key[(provider_name, model_id)] = effective_max
    return effective_by_key


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


def _safe_int_mapping(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    fallback_field_name: str | None = None,
) -> int:
    value = payload.get(field_name)
    if value is None and fallback_field_name is not None:
        value = payload.get(fallback_field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _safe_optional_int_mapping(payload: Mapping[str, Any], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _safe_float_attr(value: Any, attr_name: str) -> float:
    attr = getattr(value, attr_name, 0.0)
    return float(attr) if isinstance(attr, (int, float)) and not isinstance(attr, bool) and attr >= 0 else 0.0
