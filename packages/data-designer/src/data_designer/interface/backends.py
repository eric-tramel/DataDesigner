# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.mcp import MCPProviderT, ToolConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import SeedSource
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.resources.person_reader import PersonReader, create_person_reader
from data_designer.engine.resources.resource_provider import ResourceProvider, create_resource_provider
from data_designer.engine.resources.seed_reader import SeedReader, SeedReaderRegistry
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage

if TYPE_CHECKING:
    from data_designer.engine.models.clients.throttle_manager import ThrottleManagerLike


class BackendRuntimeContext(Protocol):
    """Stable runtime state exposed to execution backends.

    Backends may depend on this interface for engine resources and runtime
    configuration, but should not receive or inspect a public ``DataDesigner``
    instance.
    """

    @property
    def model_providers(self) -> tuple[ModelProvider, ...]: ...

    @property
    def model_provider_registry(self) -> ModelProviderRegistry: ...

    @property
    def default_provider_name(self) -> str: ...

    @property
    def secret_resolver(self) -> SecretResolver: ...

    @property
    def seed_readers(self) -> tuple[SeedReader, ...]: ...

    @property
    def managed_assets_path(self) -> Path: ...

    @property
    def artifact_path(self) -> Path: ...

    @property
    def person_reader(self) -> PersonReader | None: ...

    @property
    def mcp_providers(self) -> tuple[MCPProviderT, ...]: ...

    @property
    def run_config(self) -> RunConfig: ...

    def resolve_person_reader(self) -> PersonReader:
        """Return the configured person reader or the local managed-assets reader."""
        ...

    def create_seed_reader_registry(
        self,
        *,
        seed_readers: Sequence[SeedReader] | None = None,
    ) -> SeedReaderRegistry:
        """Create a seed-reader registry for a backend-specific execution scope."""
        ...

    def create_resource_provider(
        self,
        *,
        artifact_storage: ArtifactStorage,
        model_configs: list[ModelConfig],
        seed_dataset_source: SeedSource | None = None,
        seed_readers: Sequence[SeedReader] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        throttle_manager: ThrottleManagerLike | None = None,
    ) -> ResourceProvider:
        """Create a resource provider from this backend runtime snapshot."""
        ...


@dataclass(frozen=True)
class DataDesignerRuntimeContext:
    """Concrete backend runtime context produced by the interface layer."""

    model_providers: tuple[ModelProvider, ...]
    model_provider_registry: ModelProviderRegistry
    default_provider_name: str
    secret_resolver: SecretResolver
    seed_readers: tuple[SeedReader, ...]
    managed_assets_path: Path
    artifact_path: Path
    person_reader: PersonReader | None
    mcp_providers: tuple[MCPProviderT, ...]
    run_config: RunConfig

    def resolve_person_reader(self) -> PersonReader:
        if self.person_reader is not None:
            return self.person_reader
        return create_person_reader(str(self.managed_assets_path))

    def create_seed_reader_registry(
        self,
        *,
        seed_readers: Sequence[SeedReader] | None = None,
    ) -> SeedReaderRegistry:
        effective_seed_readers = self.seed_readers if seed_readers is None else seed_readers
        return SeedReaderRegistry(readers=effective_seed_readers)

    def create_resource_provider(
        self,
        *,
        artifact_storage: ArtifactStorage,
        model_configs: list[ModelConfig],
        seed_dataset_source: SeedSource | None = None,
        seed_readers: Sequence[SeedReader] | None = None,
        tool_configs: list[ToolConfig] | None = None,
        throttle_manager: ThrottleManagerLike | None = None,
    ) -> ResourceProvider:
        return create_resource_provider(
            artifact_storage=artifact_storage,
            model_configs=model_configs,
            secret_resolver=self.secret_resolver,
            model_provider_registry=self.model_provider_registry,
            seed_reader_registry=self.create_seed_reader_registry(seed_readers=seed_readers),
            person_reader=self.resolve_person_reader(),
            seed_dataset_source=seed_dataset_source,
            run_config=self.run_config,
            mcp_providers=list(self.mcp_providers),
            tool_configs=tool_configs,
            throttle_manager=throttle_manager,
        )


class DataDesignerBackend(Protocol):
    """Execution backend hook for non-local Data Designer runtimes."""

    def create(
        self,
        *,
        runtime_context: BackendRuntimeContext,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        input_dataset: Any | None = None,
    ) -> Any:
        """Create a dataset using the backend-specific data plane."""
