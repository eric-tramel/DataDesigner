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
