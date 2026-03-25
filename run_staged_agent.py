from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.evaluator import Evaluator
from agents.executor import Executor
from agents.planner import Planner
from schemas.state import AnalysisPlan, AnalysisState
from tools.data_inspector import DataInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged agent runner for plan-then-execute workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("plan", "execute"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--task", choices=["af", "ba", "pred"], required=True)
        sub.add_argument("--state-path", required=True)
        sub.add_argument("--plan-path", required=True)

    return parser.parse_args()


def save_plan(plan_path: Path, task: str, state_path: Path, plan: AnalysisPlan) -> None:
    payload = {
        "task": task,
        "state_path": str(state_path),
        "plan": {
            "tool_name": plan.tool_name,
            "rationale": plan.rationale,
            "params": plan.params,
        },
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2))


def load_plan(plan_path: Path) -> AnalysisPlan:
    payload = json.loads(plan_path.read_text())
    return AnalysisPlan(**payload["plan"])


def run_plan(task: str, state_path: Path, plan_path: Path) -> None:
    state = AnalysisState.load_or_create(state_path=state_path, task=task)
    inspector = DataInspector()
    planner = Planner()

    state.set_data_snapshot(inspector.inspect(task=state.task))
    plan = planner.next_plan(state=state)
    state.save()
    save_plan(plan_path=plan_path, task=task, state_path=state_path, plan=plan)

    print(f"tool={plan.tool_name}")
    print(f"rationale={plan.rationale}")
    print(f"plan_path={plan_path}")


def run_execute(task: str, state_path: Path, plan_path: Path) -> None:
    state = AnalysisState.load_or_create(state_path=state_path, task=task)
    inspector = DataInspector()
    executor = Executor()
    evaluator = Evaluator()

    state.set_data_snapshot(inspector.inspect(task=state.task))
    plan = load_plan(plan_path)

    print(f"tool={plan.tool_name}")
    print(f"rationale={plan.rationale}")

    if plan.tool_name == "inspect_only":
        state.save()
        print("status=blocked")
        print("decision=needs_review")
        return

    result = executor.execute(plan=plan, state=state)
    evaluation = evaluator.evaluate(state=state, result=result)
    state.record(plan=plan, result=result, evaluation=evaluation)
    state.save()

    print(f"status={result.status}")
    print(f"decision={evaluation.decision}")
    if evaluation.summary:
        print(evaluation.summary)


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    plan_path = Path(args.plan_path)

    if args.command == "plan":
        run_plan(task=args.task, state_path=state_path, plan_path=plan_path)
        return

    run_execute(task=args.task, state_path=state_path, plan_path=plan_path)


if __name__ == "__main__":
    main()
