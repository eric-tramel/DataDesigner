# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.integrations.ray import RayDatasetMetrics, RayMetricsError, RayWorkerMetrics
from data_designer.integrations.ray.metrics import aggregate_ray_metrics, normalize_ray_worker_metrics

pytestmark = pytest.mark.ray_fake


def test_aggregate_ray_metrics_sums_worker_metrics_and_model_usage() -> None:
    metrics = aggregate_ray_metrics(
        [
            RayWorkerMetrics(
                total_rows=10,
                blocks=1,
                elapsed_seconds=2.0,
                model_usage={
                    "model-a": {
                        "token_usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
                        "request_usage": {
                            "successful_requests": 1,
                            "failed_requests": 0,
                            "total_requests": 1,
                        },
                        "tokens_per_second": 5,
                        "requests_per_minute": 30,
                    }
                },
            ),
            {
                "total_rows": 5,
                "blocks": 2,
                "failed_blocks": 1,
                "elapsed_seconds": 3.0,
                "model_usage": {
                    "model-a": {
                        "token_usage": {"input_tokens": 2, "output_tokens": 8, "total_tokens": 10},
                        "request_usage": {
                            "successful_requests": 1,
                            "failed_requests": 1,
                            "total_requests": 2,
                        },
                        "tokens_per_second": 3,
                        "requests_per_minute": 40,
                    }
                },
            },
        ]
    )

    assert metrics == RayDatasetMetrics(
        total_rows=15,
        blocks=3,
        failed_blocks=1,
        elapsed_seconds=5.0,
        model_usage={
            "model-a": {
                "token_usage": {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20},
                "request_usage": {
                    "successful_requests": 2,
                    "failed_requests": 1,
                    "total_requests": 3,
                },
                "tokens_per_second": 4,
                "requests_per_minute": 36,
            }
        },
    )
    assert metrics.successful_blocks == 2
    assert metrics.to_dict()["total_rows"] == 15


def test_normalize_ray_worker_metrics_uses_serializable_mapping_defaults() -> None:
    metrics = normalize_ray_worker_metrics({"total_rows": 3})

    assert metrics == RayWorkerMetrics(total_rows=3, output_rows=3, blocks=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_normalize_ray_worker_metrics_rejects_non_finite_elapsed_seconds(value: float) -> None:
    with pytest.raises(RayMetricsError, match="elapsed_seconds.*finite"):
        normalize_ray_worker_metrics({"total_rows": 3, "elapsed_seconds": value})


def test_aggregate_ray_metrics_sums_row_outcomes() -> None:
    metrics = aggregate_ray_metrics(
        [
            RayWorkerMetrics(total_rows=2, input_rows=3, output_rows=2, dropped_rows=1, partial_rows_dropped=True),
            {
                "total_rows": 0,
                "input_rows": 2,
                "output_rows": 0,
                "dropped_rows": 2,
                "all_rows_dropped": True,
                "failed_blocks": 1,
            },
            {"total_rows": 0, "empty_input": True},
        ]
    )

    assert metrics.input_rows == 5
    assert metrics.output_rows == 2
    assert metrics.dropped_rows == 3
    assert metrics.partial_rows_dropped_blocks == 1
    assert metrics.all_rows_dropped_blocks == 1
    assert metrics.empty_input_blocks == 1
    assert metrics.failed_blocks == 1


def test_normalize_ray_worker_metrics_rejects_invalid_payload() -> None:
    with pytest.raises(RayMetricsError, match="must be a mapping"):
        normalize_ray_worker_metrics(object())
