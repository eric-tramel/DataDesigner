# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise RayBackend with streaming, out-of-core-style dataset consumption.

This benchmark builds a Ray Dataset that looks like a large customer-support
event stream, runs Data Designer over it with expression-only columns, writes
the generated output directly to Parquet, and validates the output by reducing
small per-block quality summaries instead of collecting the full dataset on the
driver.

Example:
    python scripts/benchmarks/benchmark_ray_streaming_out_of_core.py \
      --num-records 100000 --source-blocks 64 --batch-size 512 \
      --stream-batch-size 1024 --sandbox-safe-ray-init
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import data_designer.config as dd
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.integrations.ray import RayBackend
from data_designer.interface.data_designer import DataDesigner

RESULT_PREFIX = "RAY_STREAMING_OUT_OF_CORE_RESULT="
DEFAULT_NUM_RECORDS = 50_000
DEFAULT_SOURCE_BLOCKS = 32
DEFAULT_BATCH_SIZE = 512
DEFAULT_STREAM_BATCH_SIZE = 2048
DEFAULT_RAY_CPUS = 4
EXPECTED_COLUMNS = [
    "event_id",
    "tenant",
    "customer_id",
    "channel",
    "priority",
    "message",
    "ticket_key",
    "routing_key",
    "audit_record",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-records", type=int, default=DEFAULT_NUM_RECORDS)
    parser.add_argument("--source-blocks", type=int, default=DEFAULT_SOURCE_BLOCKS)
    parser.add_argument("--source-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stream-batch-size", type=int, default=DEFAULT_STREAM_BATCH_SIZE)
    parser.add_argument("--stream-sample-batches", type=int, default=3)
    parser.add_argument("--ray-cpus", type=int, default=DEFAULT_RAY_CPUS)
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/dd-ray-streaming-artifacts"))
    parser.add_argument("--managed-assets-path", type=Path, default=Path("/tmp/dd-ray-streaming-managed-assets"))
    parser.add_argument("--parquet-output", type=Path, default=Path("/tmp/dd-ray-streaming-parquet"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--trace-enabled", action="store_true", default=False)
    parser.add_argument("--max-trace-events", type=int, default=256)
    parser.add_argument(
        "--sandbox-safe-ray-init",
        action="store_true",
        default=False,
        help="Patch Ray startup process enumeration for restricted macOS sandboxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.managed_assets_path.mkdir(parents=True, exist_ok=True)
    _reset_output_dir(args.parquet_output)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    ray = __import__("ray")
    if args.sandbox_safe_ray_init:
        _patch_ray_sandbox_process_discovery()
    ray.init(address="local", num_cpus=args.ray_cpus, ignore_reinit_error=True, include_dashboard=False)
    try:
        payload = _run_streaming_case(ray, args)
    finally:
        ray.shutdown()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    _emit(payload)


def _run_streaming_case(ray: Any, args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    rss_before = _driver_maxrss_bytes()
    input_dataset = _support_event_stream(
        ray,
        num_records=args.num_records,
        source_blocks=args.source_blocks,
        source_batch_size=args.source_batch_size,
    )
    source_blocks_observed = _safe_num_blocks(input_dataset)

    designer = DataDesigner(
        artifact_path=args.artifact_root / "streaming-out-of-core",
        managed_assets_path=args.managed_assets_path,
        model_providers=[_unused_model_provider()],
        backend=RayBackend(
            batch_size=args.batch_size,
            output="dataset",
            profile_workers=True,
            trace_enabled=args.trace_enabled,
            max_trace_events=args.max_trace_events,
        ),
    )
    designer.set_run_config(
        RunConfig(
            buffer_size=args.batch_size,
            disable_early_shutdown=True,
            max_conversation_restarts=0,
            max_conversation_correction_steps=0,
        )
    )

    create_started_at = time.perf_counter()
    results = designer.create(
        _support_event_config(),
        input_dataset=input_dataset,
        num_records=args.num_records,
        dataset_name="streaming-out-of-core",
    )
    create_elapsed_seconds = time.perf_counter() - create_started_at
    rss_after_create = _driver_maxrss_bytes()

    write_started_at = time.perf_counter()
    results.dataset.write_parquet(str(args.parquet_output))
    write_elapsed_seconds = time.perf_counter() - write_started_at
    rss_after_write = _driver_maxrss_bytes()

    persisted = ray.data.read_parquet(str(args.parquet_output))
    persisted_rows = int(persisted.count())
    persisted_blocks = _safe_num_blocks(persisted)
    quality = _quality_summary(persisted)
    stream_sample = _stream_sample(
        persisted,
        stream_batch_size=args.stream_batch_size,
        max_batches=args.stream_sample_batches,
    )
    metrics = results.load_metrics().to_dict()
    analysis = results.load_analysis()
    elapsed_seconds = time.perf_counter() - started_at
    parquet_files = sorted(args.parquet_output.glob("*.parquet"))

    payload: dict[str, Any] = {
        "status": "ok" if _all_valid(args.num_records, persisted_rows, quality) else "invalid",
        "setup": {
            "num_records": args.num_records,
            "source_blocks": args.source_blocks,
            "source_batch_size": args.source_batch_size,
            "batch_size": args.batch_size,
            "stream_batch_size": args.stream_batch_size,
            "stream_sample_batches": args.stream_sample_batches,
            "ray_cpus": args.ray_cpus,
            "trace_enabled": args.trace_enabled,
        },
        "timing": {
            "create_elapsed_seconds": create_elapsed_seconds,
            "write_elapsed_seconds": write_elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "persisted_rows_per_second": persisted_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        },
        "memory": {
            "driver_maxrss_bytes_before": rss_before,
            "driver_maxrss_bytes_after_create": rss_after_create,
            "driver_maxrss_bytes_after_write": rss_after_write,
            "driver_maxrss_delta_bytes": max(rss_after_write - rss_before, 0),
        },
        "dataset": {
            "source_blocks_observed": source_blocks_observed,
            "result_blocks": int(metrics.get("blocks") or 0),
            "persisted_blocks": persisted_blocks,
            "persisted_rows": persisted_rows,
            "parquet_file_count": len(parquet_files),
            "parquet_total_bytes": sum(path.stat().st_size for path in parquet_files),
            "parquet_output": str(args.parquet_output),
        },
        "stream_sample": stream_sample,
        "quality": quality,
        "validity": {
            "row_count_matches": persisted_rows == args.num_records,
            "expected_columns_present": not quality["missing_columns"],
            "expected_columns_non_null": all(value == 0 for value in quality["null_counts"].values()),
            "stream_sample_non_empty": stream_sample["rows"] > 0,
            "all_output_valid": _all_valid(args.num_records, persisted_rows, quality),
        },
        "metrics": metrics,
        "analysis": None if analysis is None else analysis.to_dict(),
    }
    return payload


def _support_event_stream(ray: Any, *, num_records: int, source_blocks: int, source_batch_size: int) -> Any:
    return ray.data.range(num_records, override_num_blocks=source_blocks).map_batches(
        _support_event_seed_batch,
        batch_format="pandas",
        batch_size=source_batch_size,
    )


def _support_event_seed_batch(batch: Any) -> Any:
    ids = batch["id"].astype("int64")
    frame = lazy.pd.DataFrame()
    frame["event_id"] = ids
    frame["tenant"] = "tenant-" + (ids % 17).astype(str).str.zfill(2)
    frame["customer_id"] = "cust-" + (ids % 10_000).astype(str).str.zfill(5)
    frame["channel"] = lazy.np.select(
        [ids % 5 == 0, ids % 5 == 1, ids % 5 == 2, ids % 5 == 3],
        ["chat", "email", "phone", "web"],
        default="partner",
    )
    frame["priority"] = lazy.np.select(
        [ids % 13 == 0, ids % 7 == 0, ids % 3 == 0],
        ["urgent", "high", "normal"],
        default="low",
    )
    frame["message"] = (
        "Customer reports issue "
        + (ids % 97).astype(str)
        + " for account "
        + frame["customer_id"]
        + " through "
        + frame["channel"]
        + "."
    )
    return frame


def _support_event_config() -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=[_unused_model_config()])
    builder.add_column(dd.ExpressionColumnConfig(name="ticket_key", expr="{{ tenant }}::{{ event_id }}"))
    builder.add_column(dd.ExpressionColumnConfig(name="routing_key", expr="{{ priority }}::{{ channel }}"))
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="audit_record",
            expr="{{ ticket_key }}::{{ customer_id }}::{{ routing_key }}",
        )
    )
    return builder


def _unused_model_config() -> ModelConfig:
    return ModelConfig(
        alias="unused-model",
        model="unused",
        provider="unused",
        skip_health_check=True,
        inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=1, timeout=30),
    )


def _unused_model_provider() -> ModelProvider:
    return ModelProvider(
        name="unused",
        endpoint="http://127.0.0.1:9/v1",
        provider_type="openai",
        api_key="unused",
    )


def _quality_summary(dataset: Any) -> dict[str, Any]:
    rows = dataset.map_batches(_quality_batch, batch_format="pandas").take_all()
    total_rows = sum(int(row["rows"]) for row in rows)
    null_counts = {column: 0 for column in EXPECTED_COLUMNS}
    empty_string_counts = {column: 0 for column in EXPECTED_COLUMNS}
    missing_columns: set[str] = set()
    for row in rows:
        missing_columns.update(json.loads(row["missing_columns_json"]))
        null_payload = json.loads(row["null_counts_json"])
        empty_payload = json.loads(row["empty_string_counts_json"])
        for column in EXPECTED_COLUMNS:
            null_counts[column] += int(null_payload.get(column, 0))
            empty_string_counts[column] += int(empty_payload.get(column, 0))
    return {
        "rows_reduced": total_rows,
        "missing_columns": sorted(missing_columns),
        "null_counts": null_counts,
        "empty_string_counts": empty_string_counts,
    }


def _quality_batch(batch: Any) -> Any:
    missing = [column for column in EXPECTED_COLUMNS if column not in batch.columns]
    null_counts = {column: int(batch[column].isna().sum()) for column in EXPECTED_COLUMNS if column in batch.columns}
    empty_counts: dict[str, int] = {}
    for column in EXPECTED_COLUMNS:
        if column not in batch.columns:
            continue
        series = batch[column]
        if getattr(series.dtype, "kind", None) in {"O", "U", "S"}:
            empty_counts[column] = int(series.fillna("").astype(str).eq("").sum())
        else:
            empty_counts[column] = 0
    return lazy.pd.DataFrame(
        [
            {
                "rows": len(batch),
                "missing_columns_json": json.dumps(missing, sort_keys=True),
                "null_counts_json": json.dumps(null_counts, sort_keys=True),
                "empty_string_counts_json": json.dumps(empty_counts, sort_keys=True),
            }
        ]
    )


def _stream_sample(dataset: Any, *, stream_batch_size: int, max_batches: int) -> dict[str, Any]:
    batch_count = 0
    rows = 0
    first_rows: list[dict[str, Any]] = []
    for batch in dataset.iter_batches(batch_size=stream_batch_size, batch_format="pandas"):
        batch_count += 1
        rows += len(batch)
        if len(first_rows) < 3:
            first_rows.extend(batch.head(3 - len(first_rows)).to_dict(orient="records"))
        if batch_count >= max_batches:
            break
    return {
        "batch_count": batch_count,
        "rows": rows,
        "first_rows": first_rows,
    }


def _all_valid(expected_rows: int, persisted_rows: int, quality: dict[str, Any]) -> bool:
    return (
        persisted_rows == expected_rows
        and not quality["missing_columns"]
        and all(value == 0 for value in quality["null_counts"].values())
    )


def _safe_num_blocks(dataset: Any) -> int | None:
    num_blocks = getattr(dataset, "num_blocks", None)
    if not callable(num_blocks):
        return None
    try:
        return int(num_blocks())
    except Exception:
        return None


def _driver_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _reset_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            _reset_output_dir(child)
            child.rmdir()
        else:
            child.unlink()


def _validate_args(args: argparse.Namespace) -> None:
    for field_name in (
        "num_records",
        "source_blocks",
        "source_batch_size",
        "batch_size",
        "stream_batch_size",
        "stream_sample_batches",
        "ray_cpus",
    ):
        value = getattr(args, field_name)
        if value < 1:
            raise ValueError(f"--{field_name.replace('_', '-')} must be >= 1.")


def _patch_ray_sandbox_process_discovery() -> None:
    import ray._private.node

    def _sandbox_safe_system_processes(self: object) -> str:
        all_processes = getattr(self, "all_processes", {})
        pids: list[str] = []
        for processes in all_processes.values():
            if processes:
                pids.append(str(processes[0].process.pid))
        return ",".join(pids)

    ray._private.node.Node._get_system_processes_for_resource_isolation = _sandbox_safe_system_processes


def _emit(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True, default=str)}", flush=True)


if __name__ == "__main__":
    main()
