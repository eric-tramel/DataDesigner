# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from data_designer.integrations.ray._validation import validate_finite_number
from data_designer.integrations.ray.errors import RayMetricsError

ModelUsageSummary: TypeAlias = dict[str, dict[str, Any]]


def coerce_string_field(payload: Mapping[str, Any], field_name: str, *, field_label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise RayMetricsError(f"{field_label} {field_name!r} must be a string.")
    return value


def coerce_optional_string_field(value: Any, field_name: str, *, field_label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RayMetricsError(f"{field_label} {field_name!r} must be a string when provided.")
    return value


def coerce_int_field(payload: Mapping[str, Any], field_name: str, *, default: int, field_label: str) -> int:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"{field_label} {field_name!r} must be an integer.")
    return value


def coerce_optional_int_field(value: Any, field_name: str, *, field_label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"{field_label} {field_name!r} must be an integer when provided.")
    return value


def coerce_float_field(payload: Mapping[str, Any], field_name: str, *, default: float, field_label: str) -> float:
    value = payload.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"{field_label} {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label=field_label,
    )
    return float(value)


def coerce_bool_field(payload: Mapping[str, Any], field_name: str, *, default: bool, field_label: str) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise RayMetricsError(f"{field_label} {field_name!r} must be a boolean.")
    return value


def coerce_model_usage(
    value: Any,
    *,
    telemetry_label: str,
    field_label: str,
) -> ModelUsageSummary | None:
    """Normalize Ray model usage payloads emitted from engine usage stats.

    Per-model values intentionally mirror
    ``data_designer.engine.models.usage.ModelUsageStats.get_usage_stats()``:
    token/request usage counters, derived rate fields, and optional
    ``tool_usage``/``image_usage`` nested mappings when the engine reports
    them. The Ray layer validates the outer transport shape and recursively
    copies nested payloads so optional engine fields and future additions are
    preserved without teaching metrics and observability separate schemas.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RayMetricsError(f"{field_label} must be a mapping when provided.")

    model_usage: ModelUsageSummary = {}
    for model_name, stats in value.items():
        if not isinstance(model_name, str):
            raise RayMetricsError(f"{telemetry_label} model usage keys must be strings.")
        if not isinstance(stats, Mapping):
            raise RayMetricsError(f"{telemetry_label} model usage for {model_name!r} must be a mapping.")
        model_usage[model_name] = _copy_stats_mapping(stats)
    return model_usage


def validate_non_empty_string_field(field_name: str, value: str, *, field_label: str) -> None:
    if not isinstance(value, str) or value == "":
        raise RayMetricsError(f"{field_label} {field_name!r} must be a non-empty string.")


def validate_non_negative_int_field(field_name: str, value: int, *, field_label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RayMetricsError(f"{field_label} {field_name!r} must be an integer.")
    if value < 0:
        raise RayMetricsError(f"{field_label} {field_name!r} must be non-negative.")


def validate_non_negative_float_field(field_name: str, value: float, *, field_label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RayMetricsError(f"{field_label} {field_name!r} must be numeric.")
    validate_finite_number(
        repr(field_name),
        value,
        error_type=RayMetricsError,
        error_label=field_label,
    )
    if value < 0:
        raise RayMetricsError(f"{field_label} {field_name!r} must be non-negative.")


def validate_bool_field(field_name: str, value: bool, *, field_label: str) -> None:
    if not isinstance(value, bool):
        raise RayMetricsError(f"{field_label} {field_name!r} must be a boolean.")


def validate_failed_blocks_not_greater_than_blocks(*, blocks: int, failed_blocks: int, field_label: str) -> None:
    if failed_blocks > blocks:
        raise RayMetricsError(f"{field_label} 'failed_blocks' cannot be greater than 'blocks'.")


def _copy_stats_mapping(stats: Mapping[Any, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in stats.items():
        if not isinstance(key, str):
            raise RayMetricsError("Ray model usage stat keys must be strings.")
        output[key] = _copy_nested_value(value)
    return output


def _copy_nested_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_nested_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_nested_value(item) for item in value]
    return value
