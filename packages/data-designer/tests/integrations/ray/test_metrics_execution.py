# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from fake_ray_harness import FakeActorHandle, FakeRayDataset, install_fake_ray

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.integrations.ray import RayBackend, RayDatasetCreationResults, RayDatasetMetrics
from data_designer.integrations.ray import backend as ray_backend_module
from data_designer.integrations.ray.metrics import RayWorkerMetrics, aggregate_ray_metrics
from data_designer.interface.data_designer import DataDesigner

pytestmark = pytest.mark.ray_fake


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_ray_results_load_metrics_exposes_worker_aggregate(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    worker_metrics = [
        {
            "total_rows": 2,
            "blocks": 1,
            "elapsed_seconds": 1.0,
            "model_usage": {
                "stub-model": {
                    "token_usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
                    "request_usage": {
                        "successful_requests": 1,
                        "failed_requests": 0,
                        "total_requests": 1,
                    },
                }
            },
        },
        {
            "total_rows": 3,
            "blocks": 1,
            "elapsed_seconds": 2.0,
            "model_usage": {
                "stub-model": {
                    "token_usage": {"input_tokens": 1, "output_tokens": 9, "total_tokens": 10},
                    "request_usage": {
                        "successful_requests": 2,
                        "failed_requests": 1,
                        "total_requests": 3,
                    },
                }
            },
        },
    ]
    metrics = aggregate_ray_metrics(worker_metrics)
    results = RayDatasetCreationResults(
        dataset=FakeRayDataset([lazy.pd.DataFrame({"id": [0]})]),
        config_builder=stub_sampler_only_config_builder,
        metrics=metrics,
    )

    assert results.load_metrics() == RayDatasetMetrics(
        total_rows=5,
        blocks=2,
        worker_elapsed_seconds=3.0,
        model_usage={
            "stub-model": {
                "token_usage": {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20},
                "request_usage": {
                    "successful_requests": 3,
                    "failed_requests": 1,
                    "total_requests": 4,
                },
                "tokens_per_second": 6,
                "requests_per_minute": 80,
            }
        },
    )
    assert results.load_metrics().to_dict()["total_rows"] == 5


def test_ray_results_load_metrics_preserves_zero_worker_total_rows(
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
) -> None:
    ray = types.SimpleNamespace(get=lambda ref: ref.value)
    metrics_collector = FakeActorHandle(
        types.SimpleNamespace(snapshot=lambda: [{"total_rows": 0, "blocks": 1, "elapsed_seconds": 0.25}])
    )
    results = RayDatasetCreationResults(
        dataset=FakeRayDataset([lazy.pd.DataFrame({"id": [0, 1, 2, 3, 4]})]),
        config_builder=stub_sampler_only_config_builder,
        metrics=RayDatasetMetrics(total_rows=5, blocks=1, elapsed_seconds=10.0),
        ray=ray,
        metrics_collector=metrics_collector,
    )

    metrics = results.load_metrics()

    assert metrics.total_rows == 0
    assert metrics.blocks == 1
    assert metrics.elapsed_seconds == 10.0
    assert metrics.worker_elapsed_seconds == 0.25


def test_ray_backend_load_metrics_aggregates_worker_emitted_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
    stub_model_providers: Any,
) -> None:
    install_fake_ray(monkeypatch, with_remote=True)
    worker_metrics = [
        {
            "total_rows": 2,
            "blocks": 1,
            "elapsed_seconds": 0.5,
            "model_usage": {
                "stub-model": {
                    "token_usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                    "request_usage": {
                        "successful_requests": 1,
                        "failed_requests": 0,
                        "total_requests": 1,
                    },
                }
            },
        },
        {
            "total_rows": 1,
            "blocks": 1,
            "elapsed_seconds": 1.0,
            "model_usage": {
                "stub-model": {
                    "token_usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
                    "request_usage": {
                        "successful_requests": 1,
                        "failed_requests": 1,
                        "total_requests": 2,
                    },
                }
            },
        },
    ]
    input_dataset = FakeRayDataset(
        [
            lazy.pd.DataFrame({"id": [0, 1]}),
            lazy.pd.DataFrame({"id": [2]}),
        ]
    )

    def generate_batch(
        batch: lazy.pd.DataFrame, *, metrics_collector: Any | None = None, **_: Any
    ) -> lazy.pd.DataFrame:
        row_count = len(batch)
        ray_backend_module._record_worker_metrics(
            metrics_collector,
            RayWorkerMetrics(
                total_rows=row_count,
                blocks=1,
                elapsed_seconds=0.5 if row_count == 2 else 1.0,
                model_usage=worker_metrics[0]["model_usage"] if row_count == 2 else worker_metrics[1]["model_usage"],
            ),
        )
        return batch

    monkeypatch.setattr(ray_backend_module, "_generate_batch", generate_batch)
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=_managed_assets_path(tmp_path),
        backend=RayBackend(batch_size=2),
    )

    results = designer.create(stub_sampler_only_config_builder, input_dataset=input_dataset)
    metrics = results.load_metrics()

    assert results.load_dataset().to_pandas().to_dict(orient="records") == [{"id": 0}, {"id": 1}, {"id": 2}]
    assert metrics.total_rows == 3
    assert metrics.blocks == 2
    assert metrics.elapsed_seconds > 0
    assert metrics.worker_elapsed_seconds == 1.5
    assert metrics.model_usage == {
        "stub-model": {
            "token_usage": {"input_tokens": 6, "output_tokens": 8, "total_tokens": 14},
            "request_usage": {
                "successful_requests": 2,
                "failed_requests": 1,
                "total_requests": 3,
            },
            "tokens_per_second": 9,
            "requests_per_minute": 120,
        }
    }
