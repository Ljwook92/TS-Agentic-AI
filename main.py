from __future__ import annotations

import argparse
from pathlib import Path

from agents.evaluator import Evaluator
from agents.executor import Executor
from agents.planner import Planner
from schemas.state import AnalysisState
from tools.data_inspector import DataInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TS_Agentic_AI entrypoint")
    parser.add_argument("--task", choices=["af", "ba", "pred"], default="af")
    parser.add_argument("--tool", help="Optional explicit tool override")
    parser.add_argument("--model", help="Optional model override")
    parser.add_argument("--mode", help="Optional legacy split/mode override")
    parser.add_argument("--ts-length", type=int, help="Optional time-series length override")
    parser.add_argument("--interval", type=int, help="Optional interval override")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override")
    parser.add_argument("--sample-limit", type=int, help="Optional dataset subset size for dataset generation")
    parser.add_argument("--state-path", default="memory/latest_state.json")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    state = AnalysisState.load_or_create(state_path=state_path, task=args.task)

    planner = Planner()
    executor = Executor()
    evaluator = Evaluator()
    inspector = DataInspector()

    state.set_data_snapshot(inspector.inspect(task=state.task))

    if args.tool:
        plan = planner.make_direct_plan(
            state=state,
            tool_name=args.tool,
            model_name=args.model,
            mode=args.mode,
            ts_length=args.ts_length,
            interval=args.interval,
            batch_size=args.batch_size,
            sample_limit=args.sample_limit,
        )
    else:
        plan = planner.next_plan(state=state)

    print(f"tool={plan.tool_name}")
    print(f"rationale={plan.rationale}")

    if args.plan_only:
        state.save()
        return

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


if __name__ == "__main__":
    main()
