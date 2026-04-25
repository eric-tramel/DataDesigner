# Deployment Options: Library vs. Microservice

Data Designer is available as both an **open-source library** and a **NeMo Microservice**. This guide helps you choose the right deployment option for your use case.

## Deployment Architectures at a Glance

Data Designer supports three main deployment patterns:

<div class="grid cards" markdown>

-   **Library + Your LLM Provider**

    ---

    Each user runs the library locally and connects to their choice of LLM provider.

    ![Library with Decentralized Providers](../images/deployment-library-decentralized.png)

-   **Library + Enterprise Gateway**

    ---

    Users run the library locally but share a centralized enterprise LLM gateway with RBAC and governance.

    ![Library with Enterprise Gateway](../images/deployment-enterprise-gateway.png)

-   **SDG as a Service (Microservice)**

    ---

    A centralized SDG service that multiple users access via REST API.

    ![SDG Microservice](../images/deployment-microservice.png)

</div>

## Quick Comparison

| Aspect | Open-Source Library | NeMo Microservice |
|--------|---------------------|-------------------|
| **What it is** | Python package you import and run | REST API service exposing `preview` and `create` methods |
| **Best for** | Developers with LLM access who want flexibility and customization | Teams using NeMo Microservices platform |
| **LLM Access** | You provide (any OpenAI-compatible API) | Integrated with NeMo Microservices Platform |
| **Installation** | `pip install data-designer` | Deploy via NeMo Microservices platform |
| **Scaling** | You manage inference capacity | Managed alongside other NeMo services |

!!! success "Same Configuration API"
    Both the library and microservice use the **same `DataDesignerConfigBuilder` API**. Start with the library, and your configurations migrate seamlessly if you later adopt the NeMo platform.

## 📦 When to Use the Open-Source Library

The library is the right choice for most users. Choose it if you:

### You Have Access to LLMs

![Library with Decentralized Providers](../images/deployment-library-decentralized.png){ align=right width="350" }

You have API keys or endpoints for LLM inference:

- **Cloud APIs**: NVIDIA API Catalog (build.nvidia.com), OpenAI, Azure OpenAI, Anthropic
- **Self-hosted**: vLLM, TGI, TensorRT-LLM, or any OpenAI-compatible server
- **Enterprise gateways**: Centralized LLM gateway with RBAC, rate limiting, or other enterprise features

```python
from data_designer.interface import DataDesigner
from data_designer.config import ModelConfig

# Use any OpenAI-compatible endpoint
model = ModelConfig(
    alias="my-model",
    model="nvidia/nemotron-3-nano-30b-a3b",
    provider="nvidia",  # or "openai", or a custom ModelProvider
)

dd = DataDesigner()
# Your code controls the full workflow
```

### You Need Maximum Flexibility

- **Custom plugins**: Extend Data Designer with custom column generators, validators, or processors
- **Local development**: Rapid iteration with immediate feedback
- **Integration**: Embed Data Designer into existing Python pipelines or notebooks
- **Experimentation**: Research workflows with custom models or configurations

### You Already Have Enterprise LLM Infrastructure

![Library with Enterprise Gateway](../images/deployment-enterprise-gateway.png){ align=right width="350" }

!!! tip "Library + Enterprise LLM Gateway"
    Many enterprises already have centralized LLM access through API gateways with:

    - Role-based access control (RBAC)
    - Rate limiting and quotas
    - Audit logging
    - Cost allocation

    In this case, **use the library** and point it at your enterprise gateway. You get enterprise-grade LLM access while retaining full control over your Data Designer workflows.

```python
from data_designer.config import ModelConfig, ModelProvider

# Define your enterprise gateway as a provider
enterprise_provider = ModelProvider(
    name="enterprise-gateway",
    endpoint="https://llm-gateway.yourcompany.com/v1",
    api_key="ENTERPRISE_LLM_KEY",  # Environment variable name (uppercase) or actual key
)

# Use the provider in your model config
model = ModelConfig(
    alias="enterprise-llm",
    model="gpt-4",
    provider="enterprise-gateway",  # References the provider above
)
```

---

## ☁️ When to Use the Microservice

![SDG Microservice](../images/deployment-microservice.png){ align=right width="350" }

The NeMo Microservice exposes Data Designer's `preview` and `create` methods as REST API endpoints. Choose it if you:

### You're Using the NeMo Microservices Platform

The primary value of the microservice is **integration with other NeMo Microservices**:

- **NeMo Inference Microservices (NIMs)**: Seamless integration with NVIDIA's optimized inference endpoints
- **NeMo Customizer**: Generate synthetic data for model fine-tuning workflows
- **NeMo Evaluator**: Create evaluation datasets alongside model assessment
- **Unified deployment**: Single platform for your entire AI pipeline


### You Want to Expose SDG as a Team Service

If you need to provide synthetic data generation as a shared service:

- **Multi-tenant access**: Multiple teams submit generation jobs via API
- **Job management**: Queue, monitor, and manage generation jobs centrally
- **Resource sharing**: Shared infrastructure for SDG workloads

When users can submit configs containing Jinja templates to a shared engine, template rendering becomes a remote code execution concern and part of your security boundary. See [Security](security.md) for guidance on when to keep the default `JinjaRenderingEngine.SECURE` mode.

---

## Ray Backend Trusted-Cluster Boundary

The optional Ray backend is a library scaling mode for trusted Ray clusters. It serializes the job configuration,
model provider definitions, secret resolver state, seed-reader state, MCP provider definitions, and runtime settings
to Ray workers so each worker can construct its local generation engine.

Use option objects for Ray-specific planning and execution controls, while keeping common constructor arguments direct:

```python
from data_designer.integrations.ray import RayBackend, RayBlockPlanning, RayExecutionOptions

backend = RayBackend(
    batch_size=128,
    output="dataset",
    auto_init=False,
    block_planning=RayBlockPlanning(target_block_size=10_000, min_blocks=4),
    execution_options=RayExecutionOptions(num_cpus=1, concurrency=8),
)
```

`batch_size`, `output`, and `auto_init` remain the direct constructor controls for common use. Older individual
Ray planning and execution kwargs such as `override_num_blocks`, `read_concurrency`, `num_cpus`, and
`map_concurrency` are still accepted as compatibility shims, but do not combine them with `block_planning` or
`execution_options`; effective mixed values raise a configuration error. Prefer the option objects for new code.

Use Ray only when the driver and workers are inside the same trusted operational boundary, such as a single-tenant
cluster or a managed cluster with equivalent isolation. A Ray worker may receive provider endpoint configuration and
may be able to resolve API keys through the configured `SecretResolver` or worker environment. Treat every Ray worker
that can run the job as trusted with the same access level as the driver process.

Avoid running the Ray backend on shared clusters where untrusted users can inspect task payloads, object-store data,
worker logs, runtime environments, or process environment variables. If you must use shared infrastructure, add
platform controls outside Data Designer: tenant-isolated Ray clusters, scoped service accounts, restricted dashboard
and log access, network policies around provider endpoints, and secret management that grants only the workers for a
single job access to the required keys.

The Ray backend does not turn Data Designer into a multi-tenant service boundary. For shared service deployments,
prefer a service architecture with explicit authentication, authorization, audit logging, and request validation.

---

## 🧭 Decision Flowchart

```
                    ┌─────────────────────────┐
                    │ Are you using the NeMo  │
                    │ Microservices platform? │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                   YES                      NO
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────────────┐
        │ Use Microservice  │   │ Do you need to expose SDG │
        │                   │   │ as a shared REST service? │
        │ Integrates with   │   └─────────────┬─────────────┘
        │ NIMs, Customizer, │                 │
        │ Evaluator         │     ┌───────────┴───────────┐
        └───────────────────┘     ▼                       ▼
                                 YES                      NO
                                  │                       │
                                  ▼                       ▼
                      ┌─────────────────────┐   ┌─────────────────┐
                      │ Consider if the     │   │ Use the Library │
                      │ overhead is worth   │   │                 │
                      │ it vs. library +    │   │ Most flexible   │
                      │ enterprise gateway  │   │ option for      │
                      └─────────────────────┘   │ direct use      │
                                                └─────────────────┘
```

---

## Learn More

- **Library**: Continue with this documentation
- **Microservice**: See the [NeMo Data Designer Microservice documentation](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/index.html){target="_blank"}
- **Security model**: See [Security](security.md)
