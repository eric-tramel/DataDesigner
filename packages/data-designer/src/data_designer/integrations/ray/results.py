# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_designer.integrations.ray.errors import RayDatasetGenerationError
from data_designer.integrations.ray.metrics import (
    RayDatasetMetrics,
    RayWorkerMetrics,
    aggregate_ray_metrics,
    normalize_ray_worker_metrics,
)
from data_designer.integrations.ray.observability import RayDatasetAnalysis
from data_designer.integrations.ray.observability_collection import (
    _has_observability_payload,
    assemble_ray_dataset_analysis,
    collect_ray_dataset_stats,
)


class RayResultArtifacts:
    """Loader boundary for Ray result datasets, metrics, analysis, and output artifacts."""

    def __init__(
        self,
        *,
        dataset: Any,
        metrics: RayDatasetMetrics,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        throttle_manager: Any | None = None,
        output: Any | None = None,
    ) -> None:
        self._dataset = dataset
        self._output = output
        self._metrics_loader = RayMetricsLoader(
            driver_metrics=metrics,
            ray=ray,
            metrics_collector=metrics_collector,
            throttle_manager=throttle_manager,
            materialize_dataset_for_metrics=self._materialize_dataset_for_metrics,
        )
        self._analysis_loader = RayAnalysisLoader(
            ray=ray,
            metrics_collector=metrics_collector,
            metrics_loader=self._metrics_loader,
            dataset_getter=lambda: self._dataset,
        )

    @property
    def dataset(self) -> Any:
        """Return the current Ray Dataset reference."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Any) -> None:
        self._dataset = value

    @property
    def output(self) -> Any:
        """Return the backend-selected output object without triggering new Ray work."""
        return self._output if self._output is not None else self._dataset

    def load_dataset(self) -> Any:
        """Return the current Ray Dataset reference."""
        return self._dataset

    def load_metrics(self) -> RayDatasetMetrics:
        """Load dataset-level metrics through the metrics loader."""
        return self._metrics_loader.load_metrics()

    def load_worker_metrics(self) -> list[RayWorkerMetrics]:
        """Load per-worker metrics through the metrics loader."""
        return self._metrics_loader.load_worker_metrics()

    def load_observability(self) -> RayDatasetAnalysis | None:
        """Load Ray-native analysis through the analysis loader."""
        return self._analysis_loader.load_observability()

    def to_arrow_refs(self) -> list[Any]:
        """Return materialized Arrow ObjectRefs for the Ray result dataset."""
        if self._output is not None:
            return self._output
        try:
            return self._dataset.to_arrow_refs()
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to materialize Arrow ObjectRefs.") from exc

    def _materialize_dataset_for_metrics(self) -> bool:
        materialize = getattr(self._dataset, "materialize", None)
        if not callable(materialize):
            return False
        self._dataset = materialize()
        return True


class RayMetricsLoader:
    """Load and aggregate Ray worker metrics without depending on the public result wrapper."""

    def __init__(
        self,
        *,
        driver_metrics: RayDatasetMetrics,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        throttle_manager: Any | None = None,
        materialize_dataset_for_metrics: Callable[[], bool] | None = None,
    ) -> None:
        self._driver_metrics = driver_metrics
        self._ray = ray
        self._metrics_collector = metrics_collector
        self._throttle_manager = throttle_manager
        self._materialize_dataset_for_metrics = materialize_dataset_for_metrics
        self._metrics_cache: RayDatasetMetrics | None = None
        self._worker_metrics_cache: list[RayWorkerMetrics] | None = None

    def load_metrics(self) -> RayDatasetMetrics:
        """Return driver metrics merged with worker metrics and throttle state when available."""
        if self._metrics_cache is not None:
            return self._metrics_cache
        if self._ray is None or self._metrics_collector is None:
            return self._driver_metrics
        worker_metrics_payloads = self.load_worker_metrics()
        if not worker_metrics_payloads:
            return self._driver_metrics
        worker_metrics = aggregate_ray_metrics(
            worker_metrics_payloads,
            elapsed_seconds=self._driver_metrics.elapsed_seconds,
        )
        self._metrics_cache = _merge_driver_and_worker_metrics(
            self._driver_metrics,
            worker_metrics,
            throttle_metrics=self._load_throttle_metrics(),
        )
        return self._metrics_cache

    def load_worker_metrics(self) -> list[RayWorkerMetrics]:
        """Return per-worker metrics, materializing the dataset once if the first snapshot is empty."""
        if self._ray is None or self._metrics_collector is None:
            return []
        try:
            return list(self._load_worker_metrics_payloads())
        except Exception as exc:
            raise RayDatasetGenerationError("RayBackend failed to load worker metrics.") from exc

    def _load_worker_metrics_payloads(self) -> list[RayWorkerMetrics]:
        if self._worker_metrics_cache is not None:
            return self._worker_metrics_cache
        payloads = self._read_worker_metrics_snapshot()
        if not payloads:
            payloads = self._read_worker_metrics_after_materialization_fallback()
        self._worker_metrics_cache = [normalize_ray_worker_metrics(payload) for payload in payloads]
        return self._worker_metrics_cache

    def _read_worker_metrics_snapshot(self) -> list[Any]:
        return list(self._ray.get(self._metrics_collector.snapshot.remote()))

    def _read_worker_metrics_after_materialization_fallback(self) -> list[Any]:
        if self._materialize_dataset_for_metrics is None:
            return []
        if not self._materialize_dataset_for_metrics():
            return []
        return self._read_worker_metrics_snapshot()

    def _load_throttle_metrics(self) -> dict[str, Any] | None:
        if self._throttle_manager is None:
            return None
        snapshot = getattr(self._throttle_manager, "snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot()

    @property
    def driver_metrics(self) -> RayDatasetMetrics:
        """Return driver-only metrics captured before optional worker aggregation."""
        return self._driver_metrics


class RayAnalysisLoader:
    """Load and normalize Ray observability artifacts without the public result wrapper."""

    def __init__(
        self,
        *,
        ray: Any | None = None,
        metrics_collector: Any | None = None,
        metrics_loader: RayMetricsLoader,
        dataset_getter: Callable[[], Any] | None = None,
    ) -> None:
        self._ray = ray
        self._metrics_collector = metrics_collector
        self._metrics_loader = metrics_loader
        self._dataset_getter = dataset_getter
        self._analysis_cache: RayDatasetAnalysis | None = None

    def load_observability(self) -> RayDatasetAnalysis | None:
        """Return bounded Ray-native profiles, traces, and worker-local throttle snapshots."""
        if self._analysis_cache is not None:
            return self._analysis_cache
        diagnostic_warnings: list[str] = []
        payload = self._load_observability_payload_or_warning(diagnostic_warnings)
        metrics = self._load_metrics_for_analysis(diagnostic_warnings)
        ray_dataset_stats = self._load_ray_dataset_stats(diagnostic_warnings)
        if not _has_observability_payload(payload) and ray_dataset_stats is None and not diagnostic_warnings:
            return None

        self._analysis_cache = assemble_ray_dataset_analysis(
            metrics,
            payload,
            ray_dataset_stats=ray_dataset_stats,
            diagnostic_warnings=diagnostic_warnings,
        )
        return self._analysis_cache

    def _load_observability_payload_or_warning(self, diagnostic_warnings: list[str]) -> dict[str, Any]:
        if self._ray is None or self._metrics_collector is None:
            return {}
        try:
            return self._load_observability_payload()
        except Exception as exc:
            diagnostic_warnings.append(_diagnostic_collection_warning("load Ray observability artifacts", exc))
            return {}

    def _load_metrics_for_analysis(self, diagnostic_warnings: list[str]) -> RayDatasetMetrics:
        try:
            return self._metrics_loader.load_metrics()
        except Exception as exc:
            diagnostic_warnings.append(_diagnostic_collection_warning("load Ray metrics for analysis", exc))
            return self._metrics_loader.driver_metrics

    def _load_ray_dataset_stats(self, diagnostic_warnings: list[str]) -> Any | None:
        if self._dataset_getter is None:
            return None
        try:
            return collect_ray_dataset_stats(self._dataset_getter())
        except Exception as exc:
            diagnostic_warnings.append(_diagnostic_collection_warning("collect Ray Dataset stats", exc))
            return None

    def _load_observability_payload(self) -> dict[str, Any]:
        observability_snapshot = getattr(self._metrics_collector, "observability_snapshot", None)
        if observability_snapshot is None:
            return {}
        payload = self._ray.get(observability_snapshot.remote())
        if not _has_observability_payload(payload):
            self._metrics_loader.load_worker_metrics()
            payload = self._ray.get(observability_snapshot.remote())
        return dict(payload)


def _diagnostic_collection_warning(action: str, exc: Exception) -> str:
    return f"RayBackend failed to {action}: {type(exc).__name__}: {exc}"


def _merge_driver_and_worker_metrics(
    driver_metrics: RayDatasetMetrics,
    worker_metrics: RayDatasetMetrics,
    *,
    throttle_metrics: dict[str, Any] | None = None,
) -> RayDatasetMetrics:
    return RayDatasetMetrics(
        total_rows=worker_metrics.total_rows,
        input_rows=worker_metrics.input_rows,
        output_rows=worker_metrics.output_rows,
        dropped_rows=worker_metrics.dropped_rows,
        all_rows_dropped_blocks=worker_metrics.all_rows_dropped_blocks,
        partial_rows_dropped_blocks=worker_metrics.partial_rows_dropped_blocks,
        empty_input_blocks=worker_metrics.empty_input_blocks,
        blocks=worker_metrics.blocks,
        failed_blocks=worker_metrics.failed_blocks,
        elapsed_seconds=driver_metrics.elapsed_seconds,
        worker_elapsed_seconds=worker_metrics.worker_elapsed_seconds,
        model_usage=worker_metrics.model_usage or driver_metrics.model_usage,
        throttle=throttle_metrics or worker_metrics.throttle or driver_metrics.throttle,
    )
