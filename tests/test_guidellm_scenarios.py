from __future__ import annotations

import unittest
from pathlib import Path

from guidellm.benchmark.schemas import BenchmarkScenario

from hf_ie_tools.runner import build_scenario, load_scenario


class GuideLLMScenarioTests(unittest.TestCase):
    def test_checked_in_scenarios_match_pinned_guidellm_schema(self) -> None:
        scenario_directory = Path("configs/benchmarks/guidellm")
        scenario_paths = sorted(scenario_directory.glob("*.json"))
        self.assertTrue(scenario_paths)

        for path in scenario_paths:
            with self.subTest(path=path):
                runtime, _ = build_scenario(
                    load_scenario(path),
                    target="https://example.invalid",
                    token="hf_test",
                    model="org/model",
                    request_format="/v1/chat/completions",
                    run_id="schema-check",
                )
                BenchmarkScenario.model_validate(runtime)


if __name__ == "__main__":
    unittest.main()
