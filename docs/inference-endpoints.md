# Inference Endpoint automation

The endpoint runner follows the lifecycle exposed by `huggingface_hub`: create or get, resume when needed, wait for `running`, benchmark the endpoint URL, then keep, pause, scale to zero, or delete.

Official references:

- [Programmatic Inference Endpoint management](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints)
- [Create an endpoint](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint)
- [`HfApi.create_inference_endpoint` reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint)
- [Inference Endpoints API](https://api.endpoints.huggingface.cloud/)

## Endpoint definitions

One endpoint TOML is the complete executable unit: deployment, lifecycle, and ordered GuideLLM suite.

Two deployment sources are supported:

- `source = "custom"` maps to `HfApi.create_inference_endpoint`. It requires explicit framework, accelerator, provider, region, instance type/size, and optional image/args. Hardware availability and namespace quota are checked automatically before creation.
- `source = "catalog"` maps to `HfApi.create_inference_endpoint_from_catalog`. It requires only name, repository, and optional accelerator/namespace; HF selects the catalog's tested hardware, engine image, and arguments.

Custom example:

```toml
[endpoint]
source = "custom"
name = "glm-5-3"
repository = "zai-org/GLM-5.3"
revision = "187fb9fff6319062325ff825627ef6db084d9bc6"
framework = "custom"
accelerator = "gpu"
vendor = "aws"
region = "us-west-2"
instance_type = "nvidia-h200"
instance_size = "x8"

[endpoint.custom_image.vLLM]
url = "vllm/vllm-openai:v0.28.0"
healthRoute = "/health"
port = 8000
tensorParallelSize = 8
```

Catalog example:

```toml
[endpoint]
source = "catalog"
name = "glm-5-3-flash"
repository = "zai-org/GLM-5.3-Flash"
accelerator = "gpu"
```

The catalog helper intentionally has no engine parameter. A catalog card already identifies its tested engine and server-side recipe. For the four checked-in cards that engine is vLLM; passing the repository is equivalent to clicking Deploy in the Model Catalog UI. Use `source = "custom"` only to override that recipe with an explicitly versioned engine/image.

For custom deployments, pin the model `revision` and image version. Catalog deployments are snapshots of mutable catalog recipes; therefore each benchmark run stores the actual secret-redacted endpoint response in `endpoint.json`.

Engine-aware images must retain the API's exact image key and field casing. On multi-accelerator vLLM or SGLang instances, configure `tensorParallelSize` consistently with allocated accelerators. Container arguments such as `--tp 8` do not replace the API field.

Never put `token` or `secrets` in TOML. The loader rejects both. Use `HF_TOKEN` or `uv run hf auth login` for API authentication.

## Benchmark suite

`[benchmark]` makes endpoint creation and every benchmark one command:

```toml
[benchmark]
results_dir = "../../results"
model = "glm-5.3" # optional; omit to detect /v1/models
request_format = "/v1/chat/completions"
scenarios = [
  "../benchmarks/guidellm/smoke.json",
  "../benchmarks/guidellm/concurrent-512x128.json",
  "../benchmarks/guidellm/sweep-1024x256.json",
]
```

Model-specific OpenAI request fields can be applied to the whole suite:

```toml
[benchmark.extra_body.chat_template_kwargs]
enable_thinking = false
```

The runner maps this to GuideLLM `backend.extras.body`. Qwen capacity suites use no-thinking mode so a fixed 128-token output budget produces answer content instead of ending entirely inside `reasoning_content`. Reasoning-mode performance should use a separate endpoint config with a larger output budget.

Paths are relative to the endpoint TOML. Scenarios are validated before infrastructure mutation, run in order against one endpoint, and stop on the first nonzero GuideLLM exit. Cleanup runs once after the complete suite.

## Lifecycle policy

`[run]` controls orchestration:

```toml
[run]
reuse_existing = false
startup_timeout_seconds = 1800
after_run = "pause"
```

- `reuse_existing = false`: fail if the endpoint name exists. This prevents benchmarking an unreviewed stale configuration.
- `reuse_existing = true`: use the existing endpoint as-is; the runner does not reconcile it to TOML.
- `after_run = "keep"`: leave it running.
- `after_run = "pause"`: stop it until an explicit resume; no idle cost.
- `after_run = "scale-to-zero"`: stop it but permit an inference request to trigger a cold start.
- `after_run = "delete"`: permanently remove configuration, logs, and usage metrics.

Cleanup runs after the full suite and also when endpoint startup or GuideLLM fails. For an existing endpoint passed through `--endpoint`, the default is `keep` to avoid changing infrastructure unexpectedly. Override it explicitly with `--after-run`.

`delete` is intentionally available only as an explicit config or CLI value. Use it for disposable benchmark endpoints after logs and configuration evidence are no longer needed.

## Common commands

Run a complete create → wait → benchmark suite → pause cycle:

```bash
uv run ie-benchmark \
  --endpoint-config configs/endpoints/glm-5.3-catalog.toml
```

No separate `hf endpoints hardware` or GuideLLM command is required.

Validate the same complete plan without mutation:

```bash
uv run ie-benchmark \
  --endpoint-config configs/endpoints/glm-5.3-catalog.toml \
  --dry-run
```

Override the configured suite:

```bash
uv run ie-benchmark \
  --endpoint-config configs/endpoints/glm-5.3-catalog.toml \
  --scenario configs/benchmarks/guidellm/smoke.json
```

Reuse a named endpoint:

```bash
uv run ie-benchmark \
  --endpoint namespace/name \
  --scenario configs/benchmarks/guidellm/sweep-256x128.json \
  --after-run pause
```

Debug deployment failure:

```bash
scripts/debug/endpoint_debug.sh namespace/name
```

The debug helper checks the stored endpoint snapshot, recent errors, and configured GGUF paths at the pinned model revision. Catalog-backed endpoint records can retain a path snapshot after upstream files are renamed; see `docs/issues/ie-catalog-stale-modelpath.md`.

## Catalog namespace regression

`huggingface_hub` 1.29.0 constructs the object returned by `create_inference_endpoint_from_catalog` with the endpoint name in its namespace slot. Using that object directly produces paths such as `/endpoint/<endpoint-name>/<endpoint-name>` and a misleading `401 Unauthorized`, even though the token was accepted for creation.

The runner discards that return object and immediately refetches by the authenticated or configured namespace. Wait, pause, scale-to-zero, and delete therefore target `/endpoint/<namespace>/<endpoint-name>`.
