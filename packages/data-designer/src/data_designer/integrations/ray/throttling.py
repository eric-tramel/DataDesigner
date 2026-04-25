# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import time
from typing import Any

from data_designer.config.run_config import RunConfig, ThrottleConfig
from data_designer.engine.models.clients.throttle_manager import (
    CAPACITY_POLL_INTERVAL,
    DEFAULT_ACQUIRE_TIMEOUT,
    ThrottleDomain,
    ThrottleManager,
)
from data_designer.integrations.ray.errors import RayDatasetGenerationError

_RAY_GET_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="dd-ray-throttle")


class RayThrottleCoordinator:
    """Ray actor payload that owns job-wide provider throttle state."""

    def __init__(self, throttle_config: ThrottleConfig | None = None) -> None:
        self._manager = ThrottleManager(throttle_config)

    def register(
        self,
        *,
        provider_name: str,
        model_id: str,
        alias: str,
        max_parallel_requests: int,
    ) -> None:
        self._manager.register(
            provider_name=provider_name,
            model_id=model_id,
            alias=alias,
            max_parallel_requests=max_parallel_requests,
        )

    def try_acquire(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> float:
        return self._manager.try_acquire(
            provider_name=provider_name,
            model_id=model_id,
            domain=domain,
            now=now,
        )

    def release_success(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        self._manager.release_success(
            provider_name=provider_name,
            model_id=model_id,
            domain=domain,
            now=now,
        )

    def release_rate_limited(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        self._manager.release_rate_limited(
            provider_name=provider_name,
            model_id=model_id,
            domain=domain,
            retry_after=retry_after,
            now=now,
        )

    def release_failure(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        self._manager.release_failure(
            provider_name=provider_name,
            model_id=model_id,
            domain=domain,
            now=now,
        )

    def snapshot(self) -> dict[str, object]:
        return self._manager.snapshot()


class RayThrottleManagerProxy:
    """Throttle manager interface backed by a Ray actor.

    Acquire loops run in the caller process so the coordinator actor remains
    available to process release calls from other workers.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    def register(
        self,
        *,
        provider_name: str,
        model_id: str,
        alias: str,
        max_parallel_requests: int,
    ) -> None:
        _ray_get(
            self._coordinator.register.remote(
                provider_name=provider_name,
                model_id=model_id,
                alias=alias,
                max_parallel_requests=max_parallel_requests,
            )
        )

    def try_acquire(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> float:
        return float(
            _ray_get(
                self._coordinator.try_acquire.remote(
                    provider_name=provider_name,
                    model_id=model_id,
                    domain=domain,
                    now=now,
                )
            )
        )

    async def try_acquire_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> float:
        return float(
            await _ray_get_async(
                self._coordinator.try_acquire.remote(
                    provider_name=provider_name,
                    model_id=model_id,
                    domain=domain,
                    now=now,
                )
            )
        )

    def acquire_sync(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout
        wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)
        while wait != 0.0:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or wait > remaining:
                raise TimeoutError(
                    f"Ray throttle acquire timed out after {timeout:.0f}s "
                    f"for {provider_name}/{model_id} [{domain.value}]"
                )
            time.sleep(min(wait, remaining, CAPACITY_POLL_INTERVAL))
            wait = self.try_acquire(provider_name=provider_name, model_id=model_id, domain=domain)

    async def acquire_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        deadline = time.monotonic() + timeout
        wait = await self.try_acquire_async(provider_name=provider_name, model_id=model_id, domain=domain)
        while wait != 0.0:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or wait > remaining:
                raise TimeoutError(
                    f"Ray throttle acquire timed out after {timeout:.0f}s "
                    f"for {provider_name}/{model_id} [{domain.value}]"
                )
            await asyncio.sleep(min(wait, remaining, CAPACITY_POLL_INTERVAL))
            wait = await self.try_acquire_async(provider_name=provider_name, model_id=model_id, domain=domain)

    def release_success(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        _ray_get(
            self._coordinator.release_success.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                now=now,
            )
        )

    async def release_success_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        await _ray_get_async(
            self._coordinator.release_success.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                now=now,
            )
        )

    def release_rate_limited(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        _ray_get(
            self._coordinator.release_rate_limited.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                retry_after=retry_after,
                now=now,
            )
        )

    async def release_rate_limited_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        await _ray_get_async(
            self._coordinator.release_rate_limited.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                retry_after=retry_after,
                now=now,
            )
        )

    def release_failure(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        _ray_get(
            self._coordinator.release_failure.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                now=now,
            )
        )

    async def release_failure_async(
        self,
        *,
        provider_name: str,
        model_id: str,
        domain: ThrottleDomain,
        now: float | None = None,
    ) -> None:
        await _ray_get_async(
            self._coordinator.release_failure.remote(
                provider_name=provider_name,
                model_id=model_id,
                domain=domain,
                now=now,
            )
        )

    def snapshot(self) -> dict[str, object]:
        return dict(_ray_get(self._coordinator.snapshot.remote()))


def create_ray_throttle_manager(ray: Any, run_config: RunConfig) -> RayThrottleManagerProxy:
    """Create a Ray actor-backed throttle manager for one RayBackend job."""
    remote = getattr(ray, "remote", None)
    if not callable(remote):
        raise RayDatasetGenerationError("RayBackend global provider throttling requires ray.remote.")
    coordinator = remote(RayThrottleCoordinator).remote(run_config.throttle)
    return RayThrottleManagerProxy(coordinator)


def _ray_get(ref: Any) -> Any:
    try:
        return importlib.import_module("ray").get(ref)
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend global provider throttle coordination failed.") from exc


async def _ray_get_async(ref: Any) -> Any:
    try:
        if inspect.isawaitable(ref):
            return await ref
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_RAY_GET_EXECUTOR, _ray_get, ref)
    except RayDatasetGenerationError:
        raise
    except Exception as exc:
        raise RayDatasetGenerationError("RayBackend global provider throttle coordination failed.") from exc
