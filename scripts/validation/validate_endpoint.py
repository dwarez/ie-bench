#!/usr/bin/env python3
"""Functional validation suite for OpenAI-compatible serving endpoints (HF Jobs / Inference Endpoints).

Usage:
  uv run --with openai python validate_endpoint.py <base_url> [--model ID] [--engine vllm|sglang|llamacpp] [--no-vision]

base_url: root or /v1 URL (e.g. https://<job-id>--8000.hf.jobs or an IE endpoint URL)
--model:  served model id; default = first entry of GET /v1/models
--engine: llamacpp reports forced tool_choice as KNOWN (broken upstream), not a failure
--no-vision: skip image test (text-only models)

Auth: HF_TOKEN env var (falls back to `hf auth token`).
Exit code 1 if any test fails.
"""
import argparse
import json
import os
import subprocess
import sys

from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("base_url")
p.add_argument("--model", default=None)
p.add_argument("--engine", default="auto", choices=["vllm", "sglang", "llamacpp", "auto"])
p.add_argument("--no-vision", action="store_true")
args = p.parse_args()

BASE_URL = args.base_url.rstrip("/")
if not BASE_URL.endswith("/v1"):
    BASE_URL += "/v1"
TOKEN = os.environ.get("HF_TOKEN") or subprocess.run(
    ["hf", "auth", "token"], capture_output=True, text=True
).stdout.strip()

client = OpenAI(base_url=BASE_URL, api_key=TOKEN, timeout=600)

results = []


def report(name, ok, detail="", known=False):
    results.append((name, ok, known))
    tag = "KNOWN" if (known and not ok) else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------- models
models = client.models.list()
MODEL = args.model or models.data[0].id
report("models.list", len(models.data) > 0, f"serving: {[m.id for m in models.data]}, testing: {MODEL}")


def reasoning_of(msg):
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)


# ------------------------------------------------- basic chat + reasoning
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Give me three primes above 100."}],
    temperature=1.0,
    top_p=0.95,
    max_tokens=4096,
)
msg = resp.choices[0].message
reasoning = reasoning_of(msg)
leak = "<think>" in (msg.content or "")
report("chat.basic", bool(msg.content), f"content_len={len(msg.content or '')}")
report(
    "chat.reasoning_no_leak",
    not leak,
    f"reasoning_len={len(reasoning or '')}, think_in_content={leak}",
)

# ------------------------------------------------------- tool calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What's the weather in Paris right now?"}],
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    max_tokens=4096,
)
msg = resp.choices[0].message
tc = msg.tool_calls or []
ok = len(tc) >= 1 and tc[0].function.name == "get_weather"
try:
    ok = ok and "paris" in json.dumps(json.loads(tc[0].function.arguments)).lower()
except Exception:
    ok = False
report("tools.single_call", ok, f"calls={[(t.function.name, t.function.arguments) for t in tc]}")

if tc:
    messages = [
        {"role": "user", "content": "What's the weather in Paris right now?"},
        msg.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": tc[0].id, "content": json.dumps({"temperature": 18, "unit": "celsius", "condition": "cloudy"})},
    ]
    resp2 = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=4096)
    final = resp2.choices[0].message.content or ""
    report("tools.roundtrip", "18" in final, f"final={final[:120]!r}")
else:
    report("tools.roundtrip", False, "skipped: no tool call")

# forced tool choice (broken on llama.cpp — known upstream limitation)
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi!"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_population"}},
        max_tokens=1024,
    )
    tc = resp.choices[0].message.tool_calls or []
    ok = len(tc) == 1 and tc[0].function.name == "get_population"
    report("tools.forced_choice", ok, f"calls={[(t.function.name, t.function.arguments) for t in tc]}",
           known=(args.engine == "llamacpp"))
except Exception as e:
    report("tools.forced_choice", False, f"{type(e).__name__}", known=(args.engine == "llamacpp"))

# ------------------------------------------------- thinking control
# models differ: enable_thinking kwarg (Qwen) or reasoning_effort levels (Inkling)
suppressed = None
for attempt, extra in [
    ("enable_thinking=False", {"chat_template_kwargs": {"enable_thinking": False}}),
    ('reasoning_effort="none"', {"reasoning_effort": "none"}),
]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=1024,
            extra_body=extra,
        )
        m = resp.choices[0].message
        if not reasoning_of(m) and "hello" in (m.content or "").lower():
            suppressed = attempt
            break
    except Exception:
        continue
report("chat.thinking_control", suppressed is not None, f"worked: {suppressed}")

# --------------------------------------------------------- streaming
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=1024,
    stream=True,
    stream_options={"include_usage": True},
)
chunks, saw_content, usage = 0, False, None
for chunk in stream:
    chunks += 1
    if not chunk.choices:
        usage = chunk.usage
        continue
    if chunk.choices[0].delta.content:
        saw_content = True
report("chat.streaming", chunks > 1 and saw_content and usage is not None, f"chunks={chunks}")

# ------------------------------------------------------------ vision
if not args.no_vision:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"}},
                    {"type": "text", "text": "What is in this image? Answer in one word."},
                ],
            }],
            max_tokens=2048,
        )
        content = resp.choices[0].message.content or ""
        report("vision.image_input", len(content) > 0, f"answer={content[:80]!r}")
    except Exception as e:
        report("vision.image_input", False, f"{type(e).__name__}: {e}")

# --------------------------------------------- parallel tool calls
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Tell me the weather in Paris and the population of Tokyo."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=4096,
)
tc = resp.choices[0].message.tool_calls or []
names = sorted(t.function.name for t in tc)
report("tools.parallel", names == ["get_population", "get_weather"],
       f"calls={[(t.function.name, t.function.arguments) for t in tc]}")

# -------------------------------------- structured output (JSON schema)
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Extract: John Doe, 34 years old, lives in Berlin."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
                    "required": ["name", "age", "city"],
                    "additionalProperties": False,
                },
            },
        },
        max_tokens=1024,
    )
    person = json.loads(resp.choices[0].message.content)
    ok = person.get("name") == "John Doe" and person.get("age") == 34 and "berlin" in str(person.get("city", "")).lower()
except Exception as e:
    ok, person = False, f"{type(e).__name__}: {e}"
report("structured.json_schema", ok, f"parsed={person}")

# ------------------------------------------------------------- summary
fails = [n for n, ok, known in results if not ok and not known]
knowns = [n for n, ok, known in results if not ok and known]
print(f"\n{len(results) - len(fails) - len(knowns)}/{len(results)} passed"
      + (f" | KNOWN LIMITATIONS: {knowns}" if knowns else "")
      + (f" | FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
