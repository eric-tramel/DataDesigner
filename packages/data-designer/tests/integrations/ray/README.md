# Ray Integration Test Taxonomy

Ray integration tests use pytest markers so local runs and CI can select the intended runtime surface:

- `ray_fake`: fake-Ray and local contract tests that do not require a real Ray runtime.
- `ray_worker_boundary`: direct worker and payload boundary tests for Ray execution internals.
- `ray_real_smoke`: opt-in tests that require an installed real Ray runtime.
- `ray_live_provider`: opt-in tests that call a live model provider.
- `ray_benchmark`: benchmark helper tests for Ray benchmark scripts.

Useful commands:

```bash
uv run pytest packages/data-designer/tests/integrations/ray -m "ray_fake or ray_worker_boundary or ray_benchmark"
uv run pytest packages/data-designer/tests/integrations/ray -m ray_fake
DATA_DESIGNER_RUN_REAL_RAY_SMOKE=1 uv run pytest packages/data-designer/tests/integrations/ray -m "ray_real_smoke and not ray_live_provider"
DATA_DESIGNER_RUN_REAL_RAY_SMOKE=1 OPENAI_API_KEY=... uv run pytest packages/data-designer/tests/integrations/ray -m ray_live_provider
```
