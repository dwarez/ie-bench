from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RunResult:
    directory: Path
    returncode: int


def load_scenario(path: Path) -> dict[str, Any]:
    if path.suffix != ".json":
        raise ValueError(f"GuideLLM scenarios must be JSON files: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read GuideLLM scenario {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"GuideLLM scenario must contain a JSON object: {path}")
    spec = value.get("spec")
    if spec is not None and not isinstance(spec, dict):
        raise TypeError(f"GuideLLM scenario 'spec' must be an object: {path}")
    return value


def build_scenario(
    source: dict[str, Any],
    *,
    target: str,
    token: str | None,
    model: str | None,
    request_format: str,
    run_id: str,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = copy.deepcopy(source)
    spec = scenario.setdefault("spec", {})
    backend: dict[str, Any] = {
        "kind": "openai_http",
        "target": target.rstrip("/"),
        "request_format": request_format,
    }
    if token:
        backend["api_key"] = token
    if model:
        backend["model"] = model
    if extra_body:
        backend["extras"] = {"body": copy.deepcopy(extra_body)}
    spec["backend"] = backend

    metadata = scenario.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise TypeError("GuideLLM scenario metadata.labels must be an object")
    labels["run_id"] = run_id

    safe_scenario = copy.deepcopy(scenario)
    safe_scenario["spec"]["backend"].pop("api_key", None)
    if token:
        safe_scenario["spec"]["backend"]["authenticated"] = True
    return scenario, safe_scenario


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, default=str, indent=2, sort_keys=True) + "\n")


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(
                    marker in key.lower()
                    for marker in (
                        "authorization",
                        "api_key",
                        "password",
                        "secret",
                        "token",
                    )
                )
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _report_failure(path: Path) -> str | None:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"cannot read GuideLLM report: {error}"
    benchmarks = report.get("benchmarks") if isinstance(report, dict) else None
    if not isinstance(benchmarks, list) or not benchmarks:
        return "GuideLLM report contains no benchmarks"
    for index, benchmark in enumerate(benchmarks, start=1):
        try:
            totals = benchmark["metrics"]["request_totals"]
            successful = totals["successful"]
        except (KeyError, TypeError):
            return f"GuideLLM benchmark {index} has no request totals"
        if not isinstance(successful, int | float) or successful <= 0:
            return f"GuideLLM benchmark {index} completed no successful requests"
    return None


def run_guidellm(
    *,
    target: str,
    scenario_path: Path,
    results_root: Path,
    token: str | None,
    model: str | None = None,
    request_format: str = "/v1/chat/completions",
    endpoint_name: str | None = None,
    extra_body: dict[str, Any] | None = None,
    deployment: dict[str, Any] | None = None,
    guidellm_bin: str = "guidellm",
    dry_run: bool = False,
) -> RunResult:
    source = load_scenario(scenario_path)
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{scenario_path.stem}-{uuid4().hex[:6]}"
    run_dir = results_root.resolve() / run_id
    scenario, safe_scenario = build_scenario(
        source,
        target=target,
        token=token,
        model=model,
        request_format=request_format,
        extra_body=extra_body,
        run_id=run_id,
    )
    command_display = [guidellm_bin, "run", "--config", "<generated-config>"]

    if dry_run:
        print(f"run directory: {run_dir}")
        print(f"command: {shlex.join(command_display)}")
        print(json.dumps(safe_scenario["spec"]["backend"], indent=2, sort_keys=True))
        return RunResult(run_dir, 0)

    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "scenario.json", safe_scenario)
    if deployment is not None:
        _write_json(run_dir / "endpoint.json", _redact_secrets(deployment))
    process_environment = os.environ.copy()
    if sys.platform == "darwin":
        # GuideLLM 0.7.3 defaults to fork; its initialized tokenizer/HTTP state
        # segfaults in the child on macOS. Spawn is the platform-safe context.
        process_environment.setdefault("GUIDELLM__MP_CONTEXT_TYPE", "spawn")
    metadata = {
        "endpoint": endpoint_name,
        "finished_at": None,
        "guidellm_version": _package_version("guidellm"),
        "huggingface_hub_version": _package_version("huggingface-hub"),
        "multiprocessing_context": process_environment.get(
            "GUIDELLM__MP_CONTEXT_TYPE", "guidellm-default"
        ),
        "run_id": run_id,
        "scenario_source": str(scenario_path.resolve()),
        "started_at": now.isoformat(),
        "status": "running",
        "target": target.rstrip("/"),
    }
    _write_json(run_dir / "run.json", metadata)

    temporary_path: Path | None = None
    returncode = 1
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="ie-guidellm-", delete=False
        ) as temporary:
            json.dump(scenario, temporary)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o600)
        command = [guidellm_bin, "run", "--config", str(temporary_path)]
        returncode = subprocess.run(
            command,
            cwd=run_dir,
            env=process_environment,
            check=False,
        ).returncode
        if returncode == 0 and (
            failure := _report_failure(run_dir / "benchmarks.json")
        ):
            print(f"benchmark failed: {failure}", file=sys.stderr)
            returncode = 2
        return RunResult(run_dir, returncode)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        metadata["returncode"] = returncode
        metadata["status"] = "completed" if returncode == 0 else "failed"
        _write_json(run_dir / "run.json", metadata)
