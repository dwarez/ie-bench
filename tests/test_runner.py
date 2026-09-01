from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hf_ie_tools.runner import _report_failure, build_scenario, run_guidellm


class ScenarioTests(unittest.TestCase):
    def test_build_scenario_keeps_token_out_of_snapshot(self) -> None:
        source = {"spec": {"data": [{"kind": "synthetic_text"}]}}
        runtime, snapshot = build_scenario(
            source,
            target="https://example.test/",
            token="hf_secret",
            model="org/model",
            request_format="/v1/chat/completions",
            run_id="run-1",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        self.assertEqual(runtime["spec"]["backend"]["api_key"], "hf_secret")
        self.assertNotIn("api_key", snapshot["spec"]["backend"])
        self.assertTrue(snapshot["spec"]["backend"]["authenticated"])
        self.assertFalse(
            runtime["spec"]["backend"]["extras"]["body"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )
        self.assertEqual(source, {"spec": {"data": [{"kind": "synthetic_text"}]}})

    def test_report_rejects_a_benchmark_with_no_successful_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "benchmarks.json"
            report.write_text(
                json.dumps(
                    {"benchmarks": [{"metrics": {"request_totals": {"successful": 0}}}]}
                )
            )
            self.assertIn("no successful requests", _report_failure(report))

    def test_runner_invokes_guidellm_and_records_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scenario_path = root / "smoke.json"
            scenario_path.write_text(
                json.dumps({"spec": {"data": [{"kind": "synthetic_text"}]}})
            )
            fake_guidellm = root / "guidellm"
            fake_guidellm.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "config = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])\n"
                "pathlib.Path('received.json').write_text(config.read_text())\n"
                "report = {'benchmarks': [{'metrics': {'request_totals': {'successful': 1}}}]}\n"
                "pathlib.Path('benchmarks.json').write_text(json.dumps(report))\n"
            )
            fake_guidellm.chmod(0o755)

            result = run_guidellm(
                target="https://example.test",
                scenario_path=scenario_path,
                results_root=root / "results",
                token="hf_secret",
                deployment={
                    "model": {"repository": "org/model"},
                    "secrets": {"API_TOKEN": "must-not-persist"},
                },
                guidellm_bin=str(fake_guidellm),
            )

            self.assertEqual(result.returncode, 0)
            received = json.loads((result.directory / "received.json").read_text())
            self.assertEqual(received["spec"]["backend"]["api_key"], "hf_secret")
            snapshot = json.loads((result.directory / "scenario.json").read_text())
            self.assertNotIn("api_key", snapshot["spec"]["backend"])
            endpoint = json.loads((result.directory / "endpoint.json").read_text())
            self.assertEqual(endpoint["model"]["repository"], "org/model")
            self.assertEqual(endpoint["secrets"], "<redacted>")
            metadata = json.loads((result.directory / "run.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertTrue((result.directory / "benchmarks.json").is_file())


if __name__ == "__main__":
    unittest.main()
