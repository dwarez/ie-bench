#!/usr/bin/env python3
"""Concurrency sweep load test for OpenAI-compatible endpoints (llama.cpp / vLLM / sglang).

Usage:
  uv run --with httpx python stress_sweep.py <base_url> [levels] [reqs_per_level] [max_tokens]

  base_url:        root or /v1 URL
  levels:          comma list of concurrency levels (default 1,4,8,16,32)
  reqs_per_level:  requests per level (default 32)
  max_tokens:      generation length (default 256)

Model id is auto-detected from /v1/models. Auth: HF_TOKEN env var (or `hf auth token`).
Reports aggregate tok/s and median per-request tok/s per level. Prompts are varied
per request to defeat prompt caching.
"""
import asyncio
import os
import subprocess
import sys
import time

import httpx

BASE_URL = sys.argv[1].rstrip("/")
if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"
LEVELS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 8, 16, 32]
REQS_PER_LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 32
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 256
TOKEN = os.environ.get("HF_TOKEN") or subprocess.run(
    ["hf", "auth", "token"], capture_output=True, text=True
).stdout.strip()

PROMPT = (
    "You are analyzing server logs. Here is the context: the fleet runs inference on GPUs, "
    "requests arrive in bursts, and the scheduler batches them. Describe the trade-offs of "
    "continuous batching versus static batching for token generation latency and throughput. "
    "Include details about memory bandwidth, KV cache management, and preemption. "
    "Request variant number {i}: begin your analysis now."
)


async def detect_model(client):
    r = await client.get(f"{BASE_URL}/models")
    r.raise_for_status()
    return r.json()["data"][0]["id"]


async def one_request(client, model, i, sem):
    async with sem:
        t0 = time.perf_counter()
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(i=i)}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7,
            },
        )
        wall = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        tim = d.get("timings") or {}
        usage = d.get("usage") or {}
        return {
            "out_tokens": tim.get("predicted_n") or usage.get("completion_tokens", 0),
            "per_req_tps": tim.get("predicted_per_second", 0.0),
            "wall": wall,
        }


async def run_level(model, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=600) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[one_request(client, model, i, sem) for i in range(REQS_PER_LEVEL)],
            return_exceptions=True,
        )
        wall = time.perf_counter() - t0
    ok = [r for r in results if not isinstance(r, Exception)]
    errs = len(results) - len(ok)
    total_out = sum(r["out_tokens"] for r in ok)
    agg = total_out / wall if wall else 0
    per_req = sorted(r["per_req_tps"] for r in ok if r["per_req_tps"])
    med = per_req[len(per_req) // 2] if per_req else 0
    print(
        f"conc={concurrency:3d} | agg={agg:7.1f} tok/s | median per-req={med:6.1f} tok/s "
        f"| total_out={total_out} | wall={wall:5.1f}s | errors={errs}"
    )


async def main():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30) as client:
        model = await detect_model(client)
    print(f"target={BASE_URL} model={model} reqs/level={REQS_PER_LEVEL} max_tokens={MAX_TOKENS}")
    for c in LEVELS:
        await run_level(model, c)


asyncio.run(main())
