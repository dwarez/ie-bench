from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hf_ie_tools.cli import _is_hf_service_url, build_parser, execute
from hf_ie_tools.runner import RunResult


class AuthenticationTests(unittest.TestCase):
    def test_recognizes_hf_managed_service_domains(self) -> None:
        self.assertTrue(
            _is_hf_service_url(
                "https://abc.us-east-1.aws.endpoints.huggingface.cloud/v1"
            )
        )
        self.assertTrue(_is_hf_service_url("https://job-id--8000.hf.jobs"))

    def test_does_not_send_hf_token_to_arbitrary_or_lookalike_host(self) -> None:
        self.assertFalse(_is_hf_service_url("https://api.example.com"))
        self.assertFalse(
            _is_hf_service_url(
                "https://abc.endpoints.huggingface.cloud.attacker.example"
            )
        )


class EndpointSuiteTests(unittest.TestCase):
    def test_one_command_prepares_once_runs_all_scenarios_then_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("smoke.json", "sweep.json"):
                (root / name).write_text("{}")
            config = root / "endpoint.toml"
            config.write_text(
                """
[endpoint]
source = "catalog"
name = "bench"
repository = "org/model"
accelerator = "gpu"

[benchmark]
scenarios = ["smoke.json", "sweep.json"]
"""
            )
            cleanup = Mock()
            prepared = SimpleNamespace(
                url="https://endpoint.test",
                reference="org/bench",
                endpoint=SimpleNamespace(raw={"model": {"repository": "org/model"}}),
                cleanup=cleanup,
            )
            args = build_parser().parse_args(["--endpoint-config", str(config)])

            with (
                patch("hf_ie_tools.cli.get_token", return_value="hf_test"),
                patch(
                    "hf_ie_tools.cli.prepare_endpoint_from_plan",
                    return_value=prepared,
                ) as prepare,
                patch(
                    "hf_ie_tools.cli.run_guidellm",
                    side_effect=[
                        RunResult(root / "run-1", 0),
                        RunResult(root / "run-2", 0),
                    ],
                ) as run,
            ):
                self.assertEqual(execute(args), 0)

            prepare.assert_called_once()
            self.assertEqual(run.call_count, 2)
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
