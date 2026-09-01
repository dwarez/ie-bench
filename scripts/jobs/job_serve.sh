#!/bin/bash
# Launch a serving job on HF Jobs with an exposed port.
#
# Usage:
#   ./job_serve.sh <engine> <flavor> <model> [extra server args...]
#
#   engine: vllm | sglang | llamacpp
#   flavor: hf jobs flavor (hf jobs hardware to list), e.g. l4x1, a100-large, h200
#   model:  vllm/sglang: Hub repo id (Qwen/Qwen3.8-27B)
#           llamacpp:    repo:quant (ggml-org/gemma-4-12B-it-GGUF:Q4_0) — note:
#                        quant tags match substrings; ambiguous tags or draft models
#                        in subfolders need manual download + -m/-md (see README)
#
# Prints the job id and the public URL (HF-token auth). Then:
#   ./job_wait.sh <job-id> <port>   # wait for health
#
# Examples:
#   ./job_serve.sh vllm h200 Qwen/Qwen3.8-27B -- --reasoning-parser qwen3 --enable-auto-tool-choice
#   ./job_serve.sh llamacpp l4x1 ggml-org/gemma-4-12B-it-GGUF:Q4_0 -- -np 16 -kvu --fit-target 2048
set -eu

ENGINE=$1; FLAVOR=$2; MODEL=$3; shift 3
[ "${1:-}" = "--" ] && shift

case $ENGINE in
  vllm)
    IMAGE=vllm/vllm-openai:latest; PORT=8000
    CMD=(vllm serve "$MODEL" --host 0.0.0.0 --port 8000 "$@")
    ;;
  sglang)
    IMAGE=lmsysorg/sglang:latest; PORT=30000
    CMD=(python -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port 30000 "$@")
    ;;
  llamacpp)
    IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda; PORT=8080
    CMD=(/app/llama-server -hf "$MODEL" --host 0.0.0.0 --port 8080 --jinja "$@")
    ;;
  *)
    echo "unknown engine: $ENGINE (vllm|sglang|llamacpp)" >&2; exit 1
    ;;
esac

# NOTE: jobs ignore the image ENTRYPOINT (command is exec'd directly) and
# 'hf jobs run' eats dashed args unless separated with '--'.
hf jobs run --flavor "$FLAVOR" --timeout 4h --expose "$PORT" -d -l "app=ie-test-$ENGINE" \
  "$IMAGE" -- "${CMD[@]}"
