from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import httpx
from huggingface_hub.errors import HfHubHTTPError

from hf_ie_tools.endpoints import (
    _prepare_started,
    load_endpoint_plan,
    prepare_endpoint_from_plan,
)

_VALID_ENDPOINT = """
[endpoint]
name = "bench"
repository = "org/model"
framework = "custom"
accelerator = "gpu"
instance_size = "x1"
instance_type = "nvidia-l4"
region = "us-east-1"
vendor = "aws"

[run]
reuse_existing = true
startup_timeout_seconds = 900
after_run = "pause"

[benchmark]
results_dir = "../results"
scenarios = ["smoke.json", "sweep.json"]
"""


class EndpointPlanTests(unittest.TestCase):
    def _load(self, content: str):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "endpoint.toml"
            path.write_text(content)
            return load_endpoint_plan(path)

    def test_loads_lifecycle_and_create_arguments(self) -> None:
        plan = self._load(_VALID_ENDPOINT)
        self.assertEqual(plan.create["repository"], "org/model")
        self.assertTrue(plan.reuse_existing)
        self.assertEqual(plan.startup_timeout_seconds, 900)
        self.assertEqual(plan.after_run, "pause")
        self.assertEqual(plan.source, "custom")
        self.assertEqual(
            tuple(path.name for path in plan.scenarios), ("smoke.json", "sweep.json")
        )
        self.assertEqual(plan.results_dir.name, "results")

    def test_loads_catalog_endpoint_without_manual_hardware(self) -> None:
        plan = self._load(
            """
[endpoint]
source = "catalog"
name = "flash"
repository = "zai-org/GLM-5.3-Flash"
accelerator = "gpu"
"""
        )
        self.assertEqual(plan.source, "catalog")
        self.assertEqual(plan.create["repository"], "zai-org/GLM-5.3-Flash")
        self.assertNotIn("instance_type", plan.create)

    def test_catalog_create_refetches_with_the_real_namespace(self) -> None:
        plan = self._load(
            """
[endpoint]
source = "catalog"
name = "bench"
repository = "org/model"
accelerator = "gpu"
"""
        )
        not_found = HfHubHTTPError(
            "not found",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://example.test")
            ),
        )
        normalized = SimpleNamespace(
            status="running",
            url="https://endpoint.test",
            raw={},
        )
        api = Mock()
        api.whoami.return_value = {"name": "hf-user"}
        api.get_inference_endpoint.side_effect = [not_found, normalized]
        api.create_inference_endpoint_from_catalog.return_value = SimpleNamespace(
            namespace="bench"
        )

        with patch("huggingface_hub.HfApi", return_value=api):
            prepared = prepare_endpoint_from_plan(plan, token="hf_test")

        self.assertEqual(prepared.reference, "hf-user/bench")
        self.assertIs(prepared.endpoint, normalized)
        self.assertEqual(
            api.get_inference_endpoint.call_args_list,
            [
                call("bench", namespace="hf-user"),
                call("bench", namespace="hf-user"),
            ],
        )
        api.create_inference_endpoint_from_catalog.assert_called_once_with(
            repo_id="org/model",
            name="bench",
            accelerator="gpu",
            namespace="hf-user",
        )

    def test_checked_in_endpoint_suites_reference_existing_scenarios(self) -> None:
        configs = sorted(Path("configs/endpoints").glob("*.toml"))
        self.assertTrue(configs)
        for config in configs:
            with self.subTest(config=config):
                plan = load_endpoint_plan(config)
                self.assertTrue(plan.scenarios)
                for scenario in plan.scenarios:
                    self.assertTrue(scenario.is_file(), scenario)

    def test_rejects_committed_secrets(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not store credentials"):
            self._load(
                _VALID_ENDPOINT.replace(
                    "framework =", 'secrets = { TOKEN = "x" }\nframework ='
                )
            )

    def test_startup_failure_still_applies_cost_cleanup(self) -> None:
        class FailingEndpoint:
            status = "paused"
            url = None
            pause_calls = 0

            def resume(self) -> None:
                self.status = "pending"

            def wait(self, timeout: int) -> None:
                raise TimeoutError(f"not running after {timeout}s")

            def pause(self) -> None:
                self.pause_calls += 1

        endpoint = FailingEndpoint()
        with self.assertRaises(TimeoutError):
            _prepare_started(
                endpoint,
                reference="org/bench",
                timeout=10,
                after_run="pause",
            )
        self.assertEqual(endpoint.pause_calls, 1)


if __name__ == "__main__":
    unittest.main()
