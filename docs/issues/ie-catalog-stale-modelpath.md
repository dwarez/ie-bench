# IE: catalog `modelPath` snapshot goes stale when the GGUF repo renames a quant; deploy proceeds and crash-loops on ENOENT

## Summary

A catalog-backed llamacpp endpoint launches with a `modelPath` that no longer exists in the
model repo. The platform downloads what it can, skips the missing main model silently, and lets
`llama-server` die on `ENOENT`. Endpoint ends in `failed` with a crash loop.

Two distinct defects:

1. **Stale snapshot.** `model.image.llamacpp.modelPath` is copied into the endpoint record at
   creation and never reconciled. The catalog entry now reads `gemma-4-26B-A4B-it-Q4_0.gguf`;
   the deployed endpoint still carries `gemma-4-26B-A4B-it-Q4_K_M.gguf`, which was deleted
   upstream on 2026-07-16.
2. **No fail-fast validation.** Nothing validates `modelPath` / `mmprojModelPath` /
   `specModelPath` against the pinned revision's file list before starting the container.

## Affected endpoint

- `hf-dwarez/gemma-4-26b-a4b-it-gguf-nig`
- Repo `ggml-org/gemma-4-26B-A4B-it-GGUF`, revision `bb4531cda34d1ea09d9814959ed4d5833cf2a4c8` (== `main`)
- Image `ghcr.io/ggml-org/llama.cpp:server-cuda`, `nvidia-a100 x1`
- `fromCatalog: true`, args `-kvu --fit-target 2048`

## Evidence

Stored config (`GET /v2/endpoint/hf-dwarez/gemma-4-26b-a4b-it-gguf-nig`) — one gguf main-model
path in the entire payload, and it is the deleted one:

```json
{
  "state": "failed",
  "updatedAt": "2026-08-26T17:15:45.710Z",
  "revision": "bb4531cda34d1ea09d9814959ed4d5833cf2a4c8",
  "modelPath": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
  "mmproj":    "mmproj-gemma-4-26B-A4B-it-BF16.gguf",
  "spec":      "dflash-gemma-4-26B-A4B-it-BF16.gguf"
}
```

File existence at that exact revision (`HEAD .../resolve/<rev>/<file>`):

| HTTP | file |
|---|---|
| **404** | `gemma-4-26B-A4B-it-Q4_K_M.gguf` |
| 302 | `gemma-4-26B-A4B-it-Q4_0.gguf` |
| 302 | `mmproj-gemma-4-26B-A4B-it-BF16.gguf` |
| 302 | `dflash-gemma-4-26B-A4B-it-BF16.gguf` |

Upstream repo history — the file was deleted 10 days *before* the pinned revision:

```
2026-07-26 bb4531cd  Upload folder using huggingface_hub   <- pinned revision
2026-07-16 3d3dca20  Delete gemma-4-26B-A4B-it-Q4_K_M.gguf <- file removed
2026-07-16 663ba234  Upload folder ...                     <- Q4_K_M and Q4_0 both present
```

Remaining main-model quants: `BF16` (50GB), `Q8_0` (26GB), `Q4_0` (14GB). No K-quants.

`convert.log` shows the rename was deliberate — 4-bit is now derived from the QAT checkpoint
(`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`), so a generic Q4_K_M was redundant:

```
llama-quantize --pure --tensor-type '^token_embd=q8_0' \
  gemma-4-26B-A4B-it-QAT-BF16.gguf gemma-4-26B-A4B-it-Q4_0.gguf Q4_0
  -> quant size = 13925.86 MiB (4.63 BPW)
+ echo gemma-4-26B-A4B-it-Q4_0.gguf
```

## Container logs

Launch confirms the stale path is what reaches `llama-server`, ~12s after `updatedAt`:

```
17:15:57.810  I spec common_specu: auto-detected speculative type 'draft-dflash' from the draft model metadata
17:15:57.818  I srv    load_model: loading model '/repository/gemma-4-26B-A4B-it-Q4_K_M.gguf'
17:15:58.011  E gguf_init_from_file: failed to open GGUF file '/repository/gemma-4-26B-A4B-it-Q4_K_M.gguf' (No such file or directory)
17:15:58.011  E common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model
17:15:58.011  E cmn  common_init_: failed to load model '/repository/gemma-4-26B-A4B-it-Q4_K_M.gguf'
17:15:58.012  E srv  llama_server: exiting due to model loading error
```

Then an identical cycle at `17:15:59.606` -> `17:15:59.893`. `Exit code: 1`, state `failed`.

## Three things that make this hard to diagnose

1. **Draft/mmproj resolve first.** `auto-detected speculative type 'draft-dflash'` is logged
   *before* the failure, so `/repository` looks correctly populated. Only `modelPath` is dead.
2. **It mimics an unrelated llama.cpp bug.** `common_fit_params: ... failed to load model` is the
   same error surface as the empty-draft-path bug in `llamacpp-hfd-draft-bug.md`. Different cause:
   `-kvu --fit-target 2048` runs the fit pass, which loads the *main* model first and hits ENOENT.
3. **UI and stored config disagree.** The catalog entry displays the corrected `Q4_0`, so the
   config looks right; the launched endpoint record still holds `Q4_K_M`. Only the v2 API payload
   reveals the divergence.

## Expected

Any of:

- Endpoint creation/update validates the three GGUF paths against the pinned revision and rejects
  with `file not found in repository <repo>@<rev>: <path>` plus the available `.gguf` list.
- The download step fails loudly on a missing requested file instead of starting the container with
  an incomplete `/repository`.
- Catalog-backed endpoints reconcile `image.llamacpp.*` against the catalog entry, or surface
  "catalog config changed since deploy" in the UI.

## Actual

Silent skip, container starts, `llama-server` exits 1, crash loop, `failed`.

## Fix for this endpoint

Set `modelPath` to `gemma-4-26B-A4B-it-Q4_0.gguf` (QAT-derived, 4.63 BPW, 14GB). `mmproj` and
`spec` are already valid; `-kvu --fit-target 2048` unchanged; 14 + 1 + 0.8 GB is comfortable on
A100 80GB. Per platform behaviour, re-specify `command` + `args` + the full `image` block on the
`update_endpoint` call — the non-deep merge can drop them.

## Detection

Catalog CI should assert, per entry, that `modelPath` / `mmprojModelPath` / `specModelPath` resolve
at the pinned revision. This class of breakage arrives with any upstream re-quantisation, and
auto-converted repos (`github.com/ggml-org/convert`) rename quants without notice.
