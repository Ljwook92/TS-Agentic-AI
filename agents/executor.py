from __future__ import annotations

from dataclasses import dataclass

from schemas.state import AnalysisPlan, AnalysisState, ExecutionResult
from tools.legacy_runner import LegacyRunner


@dataclass
class Executor:
    runner: LegacyRunner = LegacyRunner()

    def execute(self, plan: AnalysisPlan, state: AnalysisState) -> ExecutionResult:
        return self.runner.run(tool_name=plan.tool_name, task=state.task, overrides=plan.params)
