from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub.errors import HfHubHTTPError

from hf_ie_tools.endpoints import (
    EndpointPlan,
    PreparedEndpoint,
    get_token,
    load_endpoint_plan,
    prepare_endpoint_from_plan,
    prepare_existing_endpoint,
)
from hf_ie_tools.runner import load_scenario, run_guidellm
from hf_ie_tools.ui import console, error, success

_DEFAULT_SCENARIO = Path("configs/benchmarks/guidellm/smoke.json")
_AFTER_RUN_CHOICES = ("keep", "pause", "scale-to-zero", "delete")
_REQUEST_FORMATS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ie-benchmark",
        description="Run a versioned GuideLLM scenario against an HF Inference Endpoint.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="OpenAI-compatible endpoint URL")
    target.add_argument(
        "--endpoint",
        metavar="NAMESPACE/NAME",
        help="existing HF Inference Endpoint; paused endpoints are resumed",
    )
    target.add_argument(
        "--endpoint-config",
        type=Path,
        metavar="PATH",
        help="TOML endpoint definition to create (or explicitly reuse)",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        type=Path,
        help=(
            "GuideLLM JSON scenario; repeatable. Endpoint-config scenarios are "
            "used when omitted."
        ),
    )
    parser.add_argument(
        "--model", help="served model ID; otherwise GuideLLM detects it"
    )
    parser.add_argument("--request-format", choices=_REQUEST_FORMATS)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--auth",
        choices=(("auto", "hf", "none")),
        default="auto",
        help=(
            "authentication for --url: auto sends an HF token only to known HF "
            "service domains (default: auto)"
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=1800,
        help="seconds to wait for an existing endpoint (default: 1800)",
    )
    parser.add_argument(
        "--after-run",
        choices=_AFTER_RUN_CHOICES,
        help="override endpoint cleanup; existing endpoints default to keep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the secret-free execution plan",
    )
    return parser


def _resolve_scenarios(
    args: argparse.Namespace, plan: EndpointPlan | None
) -> tuple[Path, ...]:
    configured = plan.scenarios if plan is not None else ()
    scenarios = tuple(args.scenarios or configured or (_DEFAULT_SCENARIO,))
    for scenario in scenarios:
        load_scenario(scenario)
    return scenarios


def _dry_run_managed(
    args: argparse.Namespace,
    plan: EndpointPlan | None,
    scenarios: tuple[Path, ...],
) -> int:
    if plan is not None:
        create = plan.create
        namespace = create.get("namespace", "<current-user>")
        print(
            f"endpoint: {namespace}/{create['name']} "
            f"(source: {plan.source}, reuse: {plan.reuse_existing})"
        )
        print(f"repository: {create['repository']}")
        if plan.source == "custom":
            print(
                "hardware: "
                f"{create['vendor']}/{create['region']} "
                f"{create['instance_type']} {create['instance_size']} "
                "(validated automatically before creation)"
            )
        else:
            print("hardware/image: selected by the Inference Endpoints catalog")
        print(f"after run: {args.after_run or plan.after_run}")
    else:
        print(f"endpoint: {args.endpoint} (existing)")
        print(f"after run: {args.after_run or 'keep'}")
    print("scenarios:")
    for scenario in scenarios:
        print(f"  - {scenario.resolve()}")
    print("GuideLLM target: <endpoint URL available after startup>")
    return 0


def _is_hf_service_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower()
    return hostname.endswith((".endpoints.huggingface.cloud", ".hf.jobs"))


def execute(args: argparse.Namespace) -> int:
    if args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be positive")
    plan = load_endpoint_plan(args.endpoint_config) if args.endpoint_config else None
    scenarios = _resolve_scenarios(args, plan)
    results_root = args.results_dir or (
        plan.results_dir if plan is not None and plan.results_dir else Path("results")
    )
    model = args.model or (plan.model if plan is not None else None)
    request_format = args.request_format or (
        plan.request_format if plan is not None else "/v1/chat/completions"
    )
    extra_body = plan.extra_body if plan is not None else None
    if args.dry_run and not args.url:
        return _dry_run_managed(args, plan, scenarios)

    managed: PreparedEndpoint | None = None
    if args.url:
        if args.auth == "hf" or (args.auth == "auto" and _is_hf_service_url(args.url)):
            token = get_token()
            if not token:
                raise RuntimeError(
                    "HF_TOKEN or an authenticated `hf auth login` is required"
                )
        else:
            token = None
        target = args.url
        endpoint_name = None
    else:
        if args.auth != "auto":
            raise ValueError("--auth applies only with --url")
        token = get_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN or an authenticated `hf auth login` is required"
            )
        if plan is not None:
            managed = prepare_endpoint_from_plan(
                plan,
                token=token,
                after_run=args.after_run,
            )
        else:
            managed = prepare_existing_endpoint(
                args.endpoint,
                token=token,
                timeout=args.startup_timeout,
                after_run=args.after_run or "keep",
            )
        target = managed.url
        endpoint_name = managed.reference

    try:
        for scenario in scenarios:
            console.rule(f"[bold cyan]{scenario.name}[/]", style="cyan")
            result = run_guidellm(
                target=target,
                scenario_path=scenario,
                results_root=results_root,
                token=token,
                model=model,
                request_format=request_format,
                extra_body=extra_body,
                endpoint_name=endpoint_name,
                deployment=(
                    getattr(managed.endpoint, "raw", None)
                    if managed is not None
                    else None
                ),
                guidellm_bin=os.environ.get("GUIDELLM_BIN", "guidellm"),
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                success(f"results: {result.directory}")
            if result.returncode:
                return result.returncode
        return 0
    finally:
        if managed is not None:
            managed.cleanup()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        returncode = execute(args)
    except HfHubHTTPError as caught:
        if getattr(caught.response, "status_code", None) == 401:
            error(
                "Hugging Face authentication failed (401). Set a current HF_TOKEN "
                "or run `uv run hf auth login`."
            )
        error(str(caught))
        sys.exit(1)
    except (OSError, RuntimeError, TypeError, ValueError) as caught:
        error(str(caught))
        sys.exit(1)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
