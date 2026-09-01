# `-hfd/--hf-repo-draft`: unmatched tag is non-fatal, server dies later with empty draft path `''`

## Environment

- llama.cpp **b10524** (`ghcr.io/ggml-org/llama.cpp:server-cuda`, built 2026-08-20)
- 1x NVIDIA L4 24GB, CUDA 13
- Repo: `unsloth/Qwen3.8-27B-GGUF` — ships an MTP draft head in a subfolder: `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`

## Repro

```bash
llama-server -hf  unsloth/Qwen3.8-27B-GGUF:Q4_K_M \
             -hfd unsloth/Qwen3.8-27B-GGUF:MTP \
             -kvu --fit-target 2048 --jinja
```

## Expected

`:MTP` resolves to `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (or the server aborts immediately with "tag matched no file"). Draft downloads, speculative decoding starts.

## Actual

1. Tag resolution fails, but only logs an error and continues:

   ```
   E common_download_get_hf_plan: no GGUF files found in repository unsloth/Qwen3.8-27B-GGUF
   I Available GGUF files:
   I  - MTP/mtp-Qwen3.8-27B-Q4_0.gguf
   I  - Qwen3.8-27B-UD-Q4_K_M.gguf
   ...
   ```

2. The server carries an **empty draft path** forward. The main model downloads and loads fine.

3. `--fit` memory measurement trips over the empty path:

   ```
   E llama_model_load_from_file_impl: exactly one out metadata, path_model, and file must be defined
   W srv load_model: [spec] failed to measure draft model memory: failed to load model
   ```

4. Speculative init fails and the server exits:

   ```
   I common_speculative_init_result: loading draft model ''
   E common_speculative_init_result: failed to load draft model, ''
   E srv llama_server: exiting due to model loading error
   ```

## Workaround (confirmed working)

```bash
curl -L -o /tmp/draft.gguf \
  https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
llama-server -hf unsloth/Qwen3.8-27B-GGUF:Q4_K_M -md /tmp/draft.gguf ...
```

Draft loads, speculative decoding runs (55% draft acceptance measured: 159/288 tokens).

## Two asks

1. **Fail fast:** if `-hfd` matches no file, abort at argument resolution with a clear message instead of propagating an empty path into model loading.
2. **Matching by path, not just quant tag:** quant-tag matching cannot address subfolder files or non-standard names (MTP drafts, `BF16/` splits). Note `:Q4_0` in this repo is ambiguous between `Qwen3.8-27B-Q4_0.gguf` and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`. Consider accepting an exact repo path after the colon, e.g. `-hfd unsloth/Qwen3.8-27B-GGUF:MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
