# Ray Backend

`RayBackend` runs Data Designer generation over Ray Data blocks and returns a Ray Dataset by default. To persist Data Designer artifacts through Ray Data instead of collecting records on the driver, construct the backend with `write_artifacts=True`.

```python
from data_designer import DataDesigner
from data_designer.integrations.ray import RayBackend

data_designer = DataDesigner(
    artifact_path="artifacts",
    backend=RayBackend(batch_size=1024, write_artifacts=True),
)
results = data_designer.create(config_builder, num_records=1_000_000, dataset_name="ray-run")
```

When artifact writing is enabled, Ray writes generated blocks through the Data Designer Ray datasink. The returned `results.load_dataset()` is a Ray Dataset read from the persisted final parquet folder.

The artifact layout matches the local backend where practical:

```text
artifacts/
  ray-run/
    builder_config.json
    metadata.json
    parquet-files/
      batch_00000.parquet
      batch_00001.parquet
    dropped-columns-parquet-files/
      batch_00000.parquet
    processors-files/
      processor_name/
        batch_00000.parquet
```

`metadata.json` records target and actual row counts, batch size, completed batch count, schema, and relative paths for final parquet files. When present, dropped-column and processor artifact paths are included under `dropped-columns-parquet-files` and `processor-files`.

`distributed_artifact_writes=True` is the default. Use a cluster-visible artifact path when running Ray workers on multiple nodes, or set `distributed_artifact_writes=False` to force Ray Data write tasks onto the driver node.

## Streaming Worker Chunks

Set `output_chunk_rows` to have Ray workers emit generated output in smaller Ray Data chunks. For partition-local jobs, RayBackend uses the engine async row-group path, so a worker can release each generated row group instead of first materializing the full `map_batches` output frame. When `write_artifacts=True`, dropped-column and processor artifact payloads are attached to each emitted chunk and split back out by the Ray datasink. If hidden ordering would need to reconcile row-count-changing output, RayBackend falls back to the older materialized chunking path to preserve ordering semantics.

```python
backend = RayBackend(
    batch_size=4096,
    output_chunk_rows=512,
    output="dataset",
)
```

The `scripts/benchmarks/benchmark_ray_streaming_out_of_core.py` benchmark accepts `--output-chunk-rows` to exercise this path against Ray Data streaming reads and Parquet writes.

## Actor Pool Defaults

`RayBackend` uses provider-aware actor-pool defaults for model-generated columns. If you do not pass
`RayExecutionOptions(use_actor_pool=...)`, the backend inspects the model configs referenced by the job:

- External provider APIs get a Ray Data actor pool with `max_size` capped by the effective provider throttle budget from `max_parallel_requests`. The automatic external-provider pool does not set `min_size` or `initial_size`, so Ray can read input blocks before starting model-client actors on small clusters while the actor cap still bounds fanout.
- Multiple aliases that point at the same provider and model use the most constrained `max_parallel_requests` value, matching Data Designer's global provider throttle behavior.
- Local endpoints such as `http://localhost:8000/v1` or `http://127.0.0.1:8000/v1` do not get an automatic actor pool size. Their best sizing depends on GPU placement, model memory, and the Ray resource requests you choose.
- Sampler-only and expression-only jobs stay in Ray task mode by default.

These defaults keep external API jobs from creating more Ray model-client workers than the provider throttle can use. Global provider throttling remains enabled by default, so even if you override actor-pool sizing, request concurrency is still coordinated across Ray workers by each model config's `max_parallel_requests`.

Pass `RayExecutionOptions(use_actor_pool=False)` to force task execution, or `RayExecutionOptions(use_actor_pool=True, ...)` to provide your own actor-pool sizing:

```python
from data_designer.integrations.ray import RayBackend, RayExecutionOptions

backend = RayBackend(
    batch_size=1024,
    execution_options=RayExecutionOptions(
        use_actor_pool=True,
        actor_pool_min_size=1,
        actor_pool_initial_size=2,
        actor_pool_max_size=8,
    ),
)
```

For external APIs, start with `actor_pool_max_size` at or below the sum of the effective `max_parallel_requests` values for the provider/model pairs used by the job. Raising the actor count above that value usually adds queued workers rather than more provider throughput.

For local GPU inference, size the actor pool around GPU resources instead of API limits. For example, if each actor reserves one GPU, set `num_gpus=1` and `actor_pool_max_size` to the number of GPUs you want Ray to use. If each actor reserves half a GPU, set `num_gpus=0.5` and cap the actor pool at roughly twice the target GPU count. Keep `max_parallel_requests` aligned with what each local model server can handle without increasing tail latency.
