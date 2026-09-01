# GuideLLM benchmark workflow

## Why GuideLLM

GuideLLM targets OpenAI-compatible servers and records LLM-specific latency and throughput metrics, including time to first token (TTFT), inter-token latency (ITL), request latency distributions, token counts, throughput, and errors. It supports synchronous, concurrent, rate-based, throughput, and sweep profiles.

Upstream references:

- [GuideLLM repository and quick start](https://github.com/vllm-project/guidellm)
- [Backend configuration](https://github.com/vllm-project/guidellm/blob/main/docs/guides/backends.md)
- [Metrics](https://github.com/vllm-project/guidellm/blob/main/docs/guides/metrics.md)
- [Outputs](https://github.com/vllm-project/guidellm/blob/main/docs/guides/outputs.md)

On macOS, GuideLLM `0.7.3`'s default `fork` worker context segfaulted before the first request in local integration testing. `ie-benchmark` sets `GUIDELLM__MP_CONTEXT_TYPE=spawn` on macOS unless the caller already chose a context. The effective value is recorded in `run.json`. Linux retains GuideLLM's default.

## Scenario contract

Files under `configs/benchmarks/guidellm/` are native GuideLLM scenario JSON. Keep workload behavior there:

- `profile`: traffic shape and concurrency/rate behavior.
- `constraints`: request count, duration, error, and saturation bounds.
- `data`: prompt/output token distribution or dataset.
- `seed`: reproducibility.
- `metrics`: retained request sample size.
- `outputs`: optional non-default output formats.

Do not put a backend or token in a checked-in scenario. `ie-benchmark` injects `spec.backend` at runtime. CLI values deliberately win over scenario defaults for target, model, request format, and authentication.

Checked-in workloads:

| scenario | contract |
|---|---|
| `smoke.json` | three synchronous 128→64 requests; compatibility gate |
| `latency-512x128.json` | 20 sequential requests; first 4 excluded as kernel warmup |
| `concurrent-512x128.json` | streams 1–32; 8s warmup and 2s cooldown per level |
| `throughput-512x128.json` | intentional saturation at concurrency 256; not an SLA latency test |
| `poisson-512x128.json` | bursty 1–32 request/s; 8s warmup per rate |
| `sweep-256x128.json` | short-input adaptive capacity sweep |
| `sweep-1024x256.json` | medium-input adaptive capacity sweep |
| `sweep-4096x512.json` | long-input adaptive capacity sweep |
| `multimodal-image-720p.json` | synthetic 720p JPEG at streams 1, 2, 4, 8 |

Endpoint TOML files select an ordered subset. A suite always starts with `smoke.json`; a failure stops later, more expensive scenarios.

vLLM may JIT-compile shape-specific Triton kernels during the first request and again at a previously unseen concurrency. Those requests produce real cold-shape latency but must not contaminate steady-state percentiles. The latency and concurrency scenarios therefore exclude explicit warmup phases. `smoke.json` deliberately retains cold-start/JIT behavior as a compatibility diagnostic.

The throughput profile intentionally overloads the queue. Its TTFT and end-to-end latency describe saturation collapse, not user-facing latency. Use its completed output tokens/s to find the ceiling, then use the concurrent profile to select the highest stream count that still meets TTFT and ITL SLOs.

To add a workload, copy the closest scenario and change only observable workload dimensions. Name fixed-token scenarios `<profile>-<input>x<output>.json`. Use a new scenario rather than mutating a historical workload after results have been compared against it.

## Running

Direct URL:

```bash
uv run ie-benchmark --url "$ENDPOINT_URL" \
  --scenario configs/benchmarks/guidellm/smoke.json
```

Explicit model selection is useful when `/v1/models` exposes more than one entry:

```bash
uv run ie-benchmark --url "$ENDPOINT_URL" \
  --model org/model \
  --scenario configs/benchmarks/guidellm/sweep-256x128.json
```

The default request format is `/v1/chat/completions`. Override it with `--request-format` for completion, responses, or embedding endpoints. The scenario's dataset must be meaningful for that API.

## Authentication

Authenticated HF endpoints expect `Authorization: Bearer <HF token>`. For managed endpoint arguments, the runner reads `HF_TOKEN`, then the token saved by `uv run hf auth login`. For a direct `--url`, the `auto` policy sends the token only to `*.endpoints.huggingface.cloud` and `*.hf.jobs`; this prevents leaking an HF credential to an arbitrary OpenAI-compatible host. Use `--auth hf` for an authenticated custom domain or `--auth none` to force an unauthenticated request. The runner places the token in an ephemeral mode-`0600` JSON config because GuideLLM's supported backend field is `api_key`. The path—not the secret—appears in process arguments, and the file is removed in `finally`.

The saved `scenario.json` omits `api_key` and records only `authenticated: true`. GuideLLM's own `benchmarks.json` is expected to serialize its Pydantic `SecretStr` as redacted; still review reports before sharing because sampled prompts and outputs may be sensitive.

## Reading results

`benchmarks.json` is the authoritative GuideLLM report. `benchmarks.csv` is the comparison surface. `run.json` records execution state, package versions, target, scenario source, and timestamps. `scenario.json` is the effective secret-free workload snapshot.

Compare runs only when these are held constant:

- model repository and immutable revision;
- engine image digest or version and engine arguments;
- accelerator type/count and replica count;
- scenario data, seed, profile, constraints, and request format;
- endpoint autoscaling state and warm/cold-start policy.

For capacity, use TTFT and ITL percentiles together with completed request rate and error rate. Peak aggregate tokens/second alone can hide unacceptable per-user latency. Treat a sweep's saturation boundary as workload-specific, not a universal endpoint concurrency limit.

## Visualizing saved results

GuideLLM can re-import `benchmarks.json` and render an interactive self-contained HTML report:

```bash
RUN=results/<run-directory>
uv run guidellm export "$RUN/benchmarks.json" \
  --output "kind=html,path=$RUN/benchmarks.html"
open "$RUN/benchmarks.html"
```

Generate a static chart for documents or CI artifacts:

```bash
uv run guidellm export "$RUN/benchmarks.json" \
  --output "kind=plot,path=$RUN/benchmarks.png,dpi=160"
```

The project includes GuideLLM's `plot` extra, so both commands work after `uv sync`. Calling `uv run guidellm export <report>` without explicit outputs renders the console summary and exports JSON, HTML, and CSV defaults.

## Updating GuideLLM

1. Change the exact GuideLLM version in `pyproject.toml`.
2. Run `uv lock`.
3. Run the offline tests.
4. Run `smoke.json` against a known endpoint.
5. Run one established sweep and compare schema, request behavior, metric names, and output files.
6. Commit the dependency change, lockfile, and any required clean-cut scenario migration together.

GuideLLM's CLI and scenario schema are still evolving. Pinning prevents historical commands from silently changing meaning.
