# Hugging Face Inference Endpoints lab

Versioned tools, deployment definitions, benchmark scenarios, and investigation notes for Hugging Face Inference Endpoints.

The primary workflow creates or resumes one endpoint, waits for it to become healthy, runs every GuideLLM scenario configured for that model, stores immutable run directories, and applies one explicit cost-control action after the suite.

## Setup

Requirements: `uv`, Python 3.12, Hugging Face write access for endpoint creation, and an accepted license for gated models. Use `uv sync` and `uv run`; the checked-in Python version and lockfile are part of benchmark reproducibility.

```bash
uv sync
uv run hf auth login
```

Dependencies are pinned in `pyproject.toml` and `uv.lock`. GuideLLM is pinned to `0.7.3`; update it deliberately and validate the checked-in scenarios before accepting a new lockfile.

## Benchmark a remote endpoint

An HF Inference Endpoint URL:

```bash
uv run ie-benchmark \
  --url https://example.us-east-1.aws.endpoints.huggingface.cloud \
  --scenario configs/benchmarks/guidellm/smoke.json
```

An existing HF Inference Endpoint:

```bash
uv run ie-benchmark \
  --endpoint namespace/endpoint-name \
  --scenario configs/benchmarks/guidellm/sweep-256x128.json
```

Paused or scaled-to-zero endpoints are resumed and awaited. Existing endpoints remain running by default; use `--after-run pause` when desired.

## One-command endpoint suites

Endpoint TOML files contain deployment, lifecycle, and benchmark-suite configuration. No separate hardware lookup or GuideLLM command is required:

```bash
uv run ie-benchmark --endpoint-config configs/endpoints/glm-5.3-catalog.toml
```

That one command:

1. validates all scenario files;
2. asks the catalog to resolve its tested engine, image, container arguments, and hardware;
3. creates the endpoint, or resumes it when explicit reuse is enabled;
4. waits for `running`;
5. runs every `[benchmark].scenarios` entry in order;
6. stops immediately if GuideLLM fails;
7. pauses the endpoint in `finally`.

Available model suites:

| config | deployment |
|---|---|
| `glm-5.3-catalog.toml` | catalog vLLM recipe, H200 x8 |
| `glm-5.3-flash-catalog.toml` | catalog vLLM recipe, H200 x8 |
| `qwen3.8-flash-next-catalog.toml` | catalog dedicated Qwen/vLLM recipe, H200 x4 |
| `ornith-1.5-35b-a3b-catalog.toml` | catalog vLLM recipe, A100 x2 |

These match the UI's Model Catalog deploy action. The catalog API accepts `repoId`, optional endpoint name, accelerator, and namespace. It does not accept an engine override: the selected catalog card already fixes the engine—in these four cases, vLLM—and resolves the tested image and arguments server-side. Use a custom endpoint definition only when deliberately benchmarking a different engine or image such as raw vLLM `0.28.0`.

Inspect a complete suite without creating or resuming anything:

```bash
uv run ie-benchmark \
  --endpoint-config configs/endpoints/glm-5.3-catalog.toml \
  --dry-run
```

Use repeated `--scenario` arguments only to override a config's default suite.

Endpoint configs never silently reuse an endpoint. Set `run.reuse_existing = true` only after verifying the existing deployment. The default `run.after_run = "pause"` prevents idle GPU cost when endpoint startup or GuideLLM fails. Other explicit policies are `keep`, `scale-to-zero`, and `delete`.

`HF_TOKEN` takes precedence over the token saved by `uv run hf auth login`. For `--url`, automatic authentication sends that token only to `*.endpoints.huggingface.cloud` and `*.hf.jobs`; use `--auth hf` for an authenticated custom domain or `--auth none` explicitly. Tokens are injected through a mode-`0600` temporary GuideLLM config, removed after execution, and omitted from checked-in configs and the saved scenario snapshot.

## Results

Each run writes to:

```text
results/<UTC timestamp>-<scenario>-<suffix>/
├── benchmarks.csv       # GuideLLM summary
├── benchmarks.json      # authoritative GuideLLM report
├── endpoint.json        # secret-redacted deployed endpoint snapshot
├── run.json             # target, versions, status, timestamps
└── scenario.json        # effective, secret-free scenario
```

`results/` is gitignored because reports can be large and may retain sampled prompts and responses. Promote selected reports to a deliberate external artifact store or a reviewed documentation path.

## Project layout

```text
configs/
  benchmarks/guidellm/   Versioned native GuideLLM scenarios
  endpoints/             Versioned endpoint deployment definitions
src/hf_ie_tools/         Endpoint lifecycle and GuideLLM orchestration
scripts/
  benchmarks/            Older lightweight throughput probes
  debug/                 Endpoint diagnostics
  jobs/                  HF Jobs serving helpers
  validation/            Functional and OCR endpoint validation
docs/
  issues/                 Reproduction notes and upstream issue drafts
tests/                    Offline orchestration contract tests
results/                  Generated and gitignored benchmark artifacts
```

GuideLLM scenarios use its native JSON schema instead of a project-specific benchmark abstraction. This keeps profiles, constraints, datasets, and outputs portable to upstream GuideLLM. The runner injects only the endpoint backend, authentication, optional model ID, and run label.

See [`docs/guidellm.md`](docs/guidellm.md) for scenario maintenance and metric caveats, and [`docs/inference-endpoints.md`](docs/inference-endpoints.md) for endpoint lifecycle and configuration rules.

## Functional validation and legacy probes

```bash
uv run --with openai python scripts/validation/validate_endpoint.py <url>
uv run --with openai --with pillow python scripts/validation/validate_ocr.py <url>
uv run --with httpx python scripts/benchmarks/stress_sweep.py <url> 1,4,8,16,32 32 256

scripts/jobs/job_serve.sh vllm h200 Qwen/Qwen3.8-27B -- --reasoning-parser qwen3
scripts/jobs/job_wait.sh <job-id> 8000
scripts/debug/endpoint_debug.sh <namespace>/<endpoint-name>
```

These scripts remain standalone investigation tools. GuideLLM is the maintained performance benchmark path because it records TTFT, inter-token latency, latency distributions, errors, and standardized reports.

## Development checks

```bash
uv run python -m unittest discover -s tests -v
uv run ie-benchmark \
  --url https://example.invalid \
  --scenario configs/benchmarks/guidellm/smoke.json \
  --dry-run
```
