# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig, LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import ProcessorType, SchemaTransformProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.dataset_builders import block_execution as block_execution_module
from data_designer.engine.dataset_builders.block_execution import (
    BlockExecutionOptions,
    DataDesignerBlockRuntimeContext,
    execute_dataset_block,
    execute_dataset_block_stream,
)
from data_designer.engine.dataset_builders.utils.task_model import TaskTrace
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver


def _managed_assets_path(tmp_path: Path) -> Path:
    path = tmp_path / "managed-assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_context(tmp_path: Path, stub_model_providers: Any) -> DataDesignerBlockRuntimeContext:
    return DataDesignerBlockRuntimeContext(
        model_providers=list(stub_model_providers),
        default_provider_name="provider-1",
        secret_resolver=PlaintextResolver(),
        seed_readers=[DataFrameSeedReader()],
        managed_assets_path=_managed_assets_path(tmp_path),
        run_config=RunConfig(buffer_size=2),
    )


def test_execute_dataset_block_generates_from_scratch_and_applies_drop_processor(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(name="kind", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]})
    )
    config_builder.add_column(ExpressionColumnConfig(name="label", expr="{{ kind }}"))
    config_builder.add_processor(processor_type=ProcessorType.DROP_COLUMNS, name="drop-kind", column_names=["kind"])

    result = execute_dataset_block(
        config_builder=config_builder,
        runtime_context=_runtime_context(tmp_path, stub_model_providers),
        num_records=3,
        options=BlockExecutionOptions(use_async=False),
    )

    assert result.input_rows == 3
    assert result.output_rows == 3
    assert result.dropped_rows == 0
    assert result.raw_dataframe.columns.to_list() == ["kind", "label"]
    assert result.dataframe.to_dict(orient="records") == [{"label": "A"}, {"label": "A"}, {"label": "A"}]


def test_execute_dataset_block_uses_input_frame_as_seed(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="x_label", expr="{{ x }}-{{ label }}"))
    input_frame = lazy.pd.DataFrame({"x": [1, 2], "label": ["a", "b"]})

    result = execute_dataset_block(
        config_builder=config_builder,
        runtime_context=_runtime_context(tmp_path, stub_model_providers),
        input_frame=input_frame,
        options=BlockExecutionOptions(use_async=False),
    )

    assert result.input_rows == 2
    assert result.output_rows == 2
    assert result.dataframe.to_dict(orient="records") == [
        {"x": 1, "label": "a", "x_label": "1-a"},
        {"x": 2, "label": "b", "x_label": "2-b"},
    ]
    assert input_frame.to_dict(orient="records") == [{"x": 1, "label": "a"}, {"x": 2, "label": "b"}]


def test_execute_dataset_block_streams_ordered_chunks_and_summary(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="x_label", expr="{{ x }}-{{ label }}"))
    config_builder.add_processor(processor_type=ProcessorType.DROP_COLUMNS, name="drop-label", column_names=["label"])
    input_frame = lazy.pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": ["a", "b", "c", "d", "e"]})

    stream = execute_dataset_block_stream(
        rows_per_chunk=2,
        config_builder=config_builder,
        runtime_context=_runtime_context(tmp_path, stub_model_providers),
        input_frame=input_frame,
        options=BlockExecutionOptions(use_async=True),
    )

    chunks = list(stream)

    assert [chunk.input_start for chunk in chunks] == [0, 2, 4]
    assert [chunk.input_rows for chunk in chunks] == [2, 2, 1]
    assert [len(chunk.dataframe) for chunk in chunks] == [2, 2, 1]
    assert chunks[0].raw_dataframe.columns.to_list() == ["x", "label", "x_label"]
    assert chunks[0].dataframe.columns.to_list() == ["x", "x_label"]
    assert lazy.pd.concat([chunk.dataframe for chunk in chunks], ignore_index=True).to_dict(orient="records") == [
        {"x": 1, "x_label": "1-a"},
        {"x": 2, "x_label": "2-b"},
        {"x": 3, "x_label": "3-c"},
        {"x": 4, "x_label": "4-d"},
        {"x": 5, "x_label": "5-e"},
    ]
    assert stream.summary.input_rows == 5
    assert stream.summary.output_rows == 5
    assert stream.summary.dropped_rows == 0
    assert stream.summary.all_rows_dropped is False


def test_execute_dataset_block_stream_captures_chunk_processor_artifacts(
    tmp_path: Path,
    stub_model_configs: Any,
    stub_model_providers: Any,
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="x_label", expr="{{ x }}-{{ label }}"))
    config_builder.add_processor(
        SchemaTransformProcessorConfig(name="schema-transform", template={"combined": "{{ x_label }}"})
    )
    config_builder.add_processor(processor_type=ProcessorType.DROP_COLUMNS, name="drop-label", column_names=["label"])
    input_frame = lazy.pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})

    stream = execute_dataset_block_stream(
        rows_per_chunk=2,
        config_builder=config_builder,
        runtime_context=_runtime_context(tmp_path, stub_model_providers),
        input_frame=input_frame,
        options=BlockExecutionOptions(use_async=True, capture_stream_artifacts=True),
    )

    chunks = list(stream)

    assert [chunk.processor_artifacts["schema-transform"].to_dict(orient="records") for chunk in chunks] == [
        [{"combined": "1-a"}, {"combined": "2-b"}],
        [{"combined": "3-c"}],
    ]
    assert lazy.pd.concat([chunk.dataframe for chunk in chunks], ignore_index=True).to_dict(orient="records") == [
        {"x": 1, "x_label": "1-a"},
        {"x": 2, "x_label": "2-b"},
        {"x": 3, "x_label": "3-c"},
    ]


def test_execute_dataset_block_uses_model_columns(
    stub_resource_provider: Any, stub_model_configs: Any, stub_model_providers: Any
) -> None:
    stub_resource_provider.model_registry.get_model_config.return_value = stub_model_configs[0]
    stub_resource_provider.model_registry.get_model_provider.return_value = stub_model_providers[0]
    stub_resource_provider.model_registry.get_model_usage_stats.return_value = {}
    stub_resource_provider.model_registry.get_usage_deltas.return_value = {}
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["topic"]})
    )
    config_builder.add_column(LLMTextColumnConfig(name="text", prompt="{{ seed }}", model_alias="stub-model"))

    result = execute_dataset_block(
        config_builder=config_builder,
        resource_provider=stub_resource_provider,
        num_records=1,
        options=BlockExecutionOptions(use_async=False),
    )

    assert result.output_rows == 1
    assert result.dataframe["text"].to_list() == ["Generated summary text"]
    stub_resource_provider.model_registry.get_model.assert_called()


@pytest.mark.parametrize(
    ("processed", "expected_dropped", "expected_all", "expected_partial"),
    [
        (lazy.pd.DataFrame({"value": []}), 3, True, False),
        (lazy.pd.DataFrame({"value": [1, 2]}), 1, False, True),
    ],
)
def test_execute_dataset_block_reports_row_outcomes_and_task_traces(
    monkeypatch: pytest.MonkeyPatch,
    stub_resource_provider: Any,
    stub_model_configs: Any,
    processed: Any,
    expected_dropped: int,
    expected_all: bool,
    expected_partial: bool,
) -> None:
    trace = TaskTrace(column="value", row_group=0, row_index=None, task_type="batch", status="ok")

    class StubBuilder:
        def __init__(self, **_: Any) -> None:
            self.task_traces = [trace]

        def build_block(self, *, num_records: int, current_batch_number: int | None = None) -> Any:
            del current_batch_number
            return lazy.pd.DataFrame({"value": list(range(num_records))}), processed

    stub_resource_provider.model_registry.get_model_usage_stats.return_value = {}
    stub_resource_provider.model_registry.get_usage_deltas.return_value = {}
    monkeypatch.setattr(block_execution_module, "DatasetBuilder", StubBuilder)
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(ExpressionColumnConfig(name="value", expr="1"))

    result = execute_dataset_block(
        config_builder=config_builder,
        resource_provider=stub_resource_provider,
        num_records=3,
        options=BlockExecutionOptions(use_async=False),
    )

    assert result.input_rows == 3
    assert result.output_rows == len(processed)
    assert result.dropped_rows == expected_dropped
    assert result.all_rows_dropped is expected_all
    assert result.partial_rows_dropped is expected_partial
    assert result.task_traces == [trace]
