from __future__ import annotations

import os
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hf_ie_tools.ui import console, info, success, warning

_ALLOWED_AFTER_RUN = {"keep", "pause", "scale-to-zero", "delete"}
_ALLOWED_REQUEST_FORMATS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
}
_REQUIRED_CUSTOM_FIELDS = {
    "name",
    "repository",
    "framework",
    "accelerator",
    "instance_size",
    "instance_type",
    "region",
    "vendor",
}
_CATALOG_FIELDS = {"accelerator", "name", "namespace", "repository", "source"}


@dataclass(frozen=True)
class EndpointPlan:
    source: str
    create: dict[str, Any]
    reuse_existing: bool
    startup_timeout_seconds: int
    after_run: str
    scenarios: tuple[Path, ...]
    results_dir: Path | None
    model: str | None
    extra_body: dict[str, Any]
    request_format: str


@dataclass
class PreparedEndpoint:
    endpoint: Any
    reference: str
    url: str
    after_run: str

    def cleanup(self) -> None:
        if self.after_run == "keep":
            return
        info(f"{self.after_run} endpoint [bold]{self.reference}[/]")
        if self.after_run == "pause":
            self.endpoint.pause()
        elif self.after_run == "scale-to-zero":
            self.endpoint.scale_to_zero()
        elif self.after_run == "delete":
            self.endpoint.delete()
        success(f"endpoint cleanup complete: {self.after_run}")


def get_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token
    try:
        from huggingface_hub import get_token as get_saved_token
    except ImportError:
        return None
    return get_saved_token()


def load_endpoint_plan(path: Path) -> EndpointPlan:
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read endpoint config {path}: {error}") from error
    create = document.get("endpoint")
    if not isinstance(create, dict):
        raise TypeError(f"endpoint config requires an [endpoint] table: {path}")
    source = create.get("source", "custom")
    if source not in {"catalog", "custom"}:
        raise ValueError("endpoint source must be 'catalog' or 'custom'")
    required = (
        {"name", "repository"} if source == "catalog" else _REQUIRED_CUSTOM_FIELDS
    )
    missing = sorted(required - create.keys())
    if missing:
        raise ValueError(f"endpoint config is missing: {', '.join(missing)}")
    if source == "catalog":
        unknown_catalog_fields = set(create) - _CATALOG_FIELDS
        if unknown_catalog_fields:
            raise ValueError(
                "catalog endpoint has unsupported keys: "
                + ", ".join(sorted(unknown_catalog_fields))
            )
    forbidden = {"token", "secrets"} & create.keys()
    if forbidden:
        raise ValueError(
            "do not store credentials in endpoint configs; remove: "
            + ", ".join(sorted(forbidden))
        )
    create = dict(create)
    create.pop("source", None)

    run = document.get("run", {})
    if not isinstance(run, dict):
        raise TypeError("endpoint config [run] must be a table")
    unknown = set(run) - {
        "after_run",
        "reuse_existing",
        "startup_timeout_seconds",
    }
    if unknown:
        raise ValueError(f"unknown endpoint [run] keys: {', '.join(sorted(unknown))}")
    after_run = run.get("after_run", "pause")
    if after_run not in _ALLOWED_AFTER_RUN:
        raise ValueError(
            f"endpoint after_run must be one of {', '.join(sorted(_ALLOWED_AFTER_RUN))}"
        )
    timeout = run.get("startup_timeout_seconds", 1800)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("endpoint startup_timeout_seconds must be a positive integer")
    reuse = run.get("reuse_existing", False)
    if not isinstance(reuse, bool):
        raise TypeError("endpoint reuse_existing must be true or false")

    benchmark = document.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise TypeError("endpoint config [benchmark] must be a table")
    unknown_benchmark = set(benchmark) - {
        "extra_body",
        "model",
        "request_format",
        "results_dir",
        "scenarios",
    }
    if unknown_benchmark:
        raise ValueError(
            f"unknown endpoint [benchmark] keys: {', '.join(sorted(unknown_benchmark))}"
        )
    scenario_values = benchmark.get("scenarios", [])
    if not isinstance(scenario_values, list) or not all(
        isinstance(value, str) and value for value in scenario_values
    ):
        raise TypeError("endpoint benchmark.scenarios must be an array of paths")
    scenarios = tuple((path.parent / value).resolve() for value in scenario_values)
    results_value = benchmark.get("results_dir")
    if results_value is not None and not isinstance(results_value, str):
        raise TypeError("endpoint benchmark.results_dir must be a path string")
    results_dir = (
        (path.parent / results_value).resolve() if results_value is not None else None
    )
    model = benchmark.get("model")
    if model is not None and not isinstance(model, str):
        raise TypeError("endpoint benchmark.model must be a string")
    extra_body = benchmark.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("endpoint benchmark.extra_body must be a table")
    request_format = benchmark.get("request_format", "/v1/chat/completions")
    if request_format not in _ALLOWED_REQUEST_FORMATS:
        raise ValueError(
            f"unsupported endpoint benchmark.request_format: {request_format}"
        )
    return EndpointPlan(
        source=source,
        create=create,
        reuse_existing=reuse,
        startup_timeout_seconds=timeout,
        after_run=after_run,
        scenarios=scenarios,
        results_dir=results_dir,
        model=model,
        request_format=request_format,
        extra_body=extra_body,
    )


def _split_reference(reference: str) -> tuple[str, str]:
    namespace, separator, name = reference.partition("/")
    if not separator or not namespace or not name or "/" in name:
        raise ValueError("endpoint must be written as <namespace>/<name>")
    return namespace, name


def _status(endpoint: Any) -> str:
    status = endpoint.status
    return getattr(status, "value", status)


def _start(endpoint: Any, timeout: int) -> Any:
    status = _status(endpoint)
    if status in {"paused", "scaledToZero"}:
        info(f"resuming endpoint from [yellow]{status}[/]")
        endpoint.resume()
        status = _status(endpoint)
    if status != "running":
        with console.status(
            f"[bold cyan]Waiting for endpoint[/] [dim](state: {status})[/]",
            spinner="dots",
        ):
            endpoint.wait(timeout=timeout)
    if not endpoint.url:
        endpoint.fetch()
    if not endpoint.url:
        raise RuntimeError("running endpoint did not expose a URL")
    success(f"endpoint ready: {endpoint.url}")
    return endpoint


def _prepare_started(
    endpoint: Any,
    *,
    reference: str,
    timeout: int,
    after_run: str,
) -> PreparedEndpoint:
    prepared = PreparedEndpoint(endpoint, reference, endpoint.url or "", after_run)
    try:
        _start(endpoint, timeout)
    except BaseException:
        try:
            prepared.cleanup()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve startup error
            warning(f"endpoint cleanup also failed for {reference}: {cleanup_error}")
        raise
    prepared.url = endpoint.url
    return prepared


def prepare_existing_endpoint(
    reference: str,
    *,
    token: str,
    timeout: int,
    after_run: str,
) -> PreparedEndpoint:
    from huggingface_hub import HfApi

    namespace, name = _split_reference(reference)
    endpoint = HfApi(token=token).get_inference_endpoint(name, namespace=namespace)
    return _prepare_started(
        endpoint,
        reference=reference,
        timeout=timeout,
        after_run=after_run,
    )


def _validate_custom_hardware(api: Any, create: dict[str, Any]) -> None:
    desired = (
        create["vendor"],
        create["region"],
        create["accelerator"],
        create["instance_type"],
        create["instance_size"],
    )
    available = api.list_inference_endpoints_hardware(namespace=create.get("namespace"))
    if any(
        (
            item.vendor,
            item.region,
            item.accelerator,
            item.instance_type,
            item.instance_size,
        )
        == desired
        for item in available
    ):
        return
    description = "/".join(str(value) for value in desired)
    raise RuntimeError(
        f"hardware is unavailable or out of quota for this namespace: {description}"
    )


def prepare_endpoint_from_plan(
    plan: EndpointPlan,
    *,
    token: str,
    after_run: str | None = None,
) -> PreparedEndpoint:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    create = dict(plan.create)
    requested_namespace = create.get("namespace")
    name = create["name"]
    api = HfApi(token=token)
    namespace = requested_namespace or api.whoami()["name"]
    endpoint = None
    try:
        endpoint = api.get_inference_endpoint(name, namespace=namespace)
    except HfHubHTTPError as error:
        if getattr(error.response, "status_code", None) != 404:
            raise

    reference = f"{namespace}/{name}"
    if endpoint is not None and not plan.reuse_existing:
        raise RuntimeError(
            f"endpoint {reference} already exists; set run.reuse_existing=true to reuse it"
        )
    if endpoint is None:
        info(f"creating endpoint [bold]{reference}[/] from {plan.source}")
        if plan.source == "catalog":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"'HfApi\\.create_inference_endpoint_from_catalog'.*",
                    category=UserWarning,
                )
                api.create_inference_endpoint_from_catalog(
                    repo_id=create["repository"],
                    name=name,
                    accelerator=create.get("accelerator"),
                    namespace=namespace,
                )
        else:
            _validate_custom_hardware(api, create)
            api.create_inference_endpoint(**create)
        # huggingface_hub 1.29.0 builds the catalog return object with the
        # endpoint name as its namespace. Refetch to bind the actual namespace;
        # otherwise wait/pause requests target /endpoint/<name>/<name> and 401.
        endpoint = api.get_inference_endpoint(name, namespace=namespace)
    else:
        info(f"reusing endpoint [bold]{reference}[/]")

    return _prepare_started(
        endpoint,
        reference=reference,
        timeout=plan.startup_timeout_seconds,
        after_run=after_run or plan.after_run,
    )
