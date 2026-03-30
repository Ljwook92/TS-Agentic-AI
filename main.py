from __future__ import annotations

import argparse
from pathlib import Path

from agents.evaluator import Evaluator
from agents.executor import Executor
from agents.planner import Planner
from schemas.state import AnalysisState
from tools.data_inspector import DataInspector
from tools.reporting import ReportGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TS_Agentic_AI entrypoint")
    parser.add_argument("--task", choices=["af", "ba", "pred"], default="af")
    parser.add_argument("--tool", help="Optional explicit tool override")
    parser.add_argument("--model", help="Optional model override")
    parser.add_argument("--attn-version", help="Optional attention version override for Swin-based models")
    parser.add_argument("--mode", help="Optional legacy split/mode override")
    parser.add_argument("--ts-length", type=int, help="Optional time-series length override")
    parser.add_argument("--interval", type=int, help="Optional interval override")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override")
    parser.add_argument("--learning-rate", type=float, help="Optional learning rate override")
    parser.add_argument("--num-heads", type=int, help="Optional attention head count override")
    parser.add_argument("--embedding-dim", type=int, help="Optional embedding/hidden size override")
    parser.add_argument("--epochs", type=int, help="Optional epoch override for legacy training scripts")
    parser.add_argument("--sample-limit", type=int, help="Optional dataset subset size for dataset generation")
    parser.add_argument("--state-path", default="memory/latest_state.json")
    parser.add_argument("--report", action="store_true", help="Print a comparison report for the provided state file")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum autonomous agent steps when no explicit tool override is provided")
    return parser.parse_args()


def resolve_inspection_target(
    state: AnalysisState,
    explicit_ts_length: int | None,
    explicit_interval: int | None,
) -> tuple[int, int]:
    if explicit_ts_length is not None:
        return explicit_ts_length, explicit_interval if explicit_interval is not None else 1

    for entry in reversed(state.history):
        ts_length = entry.plan.params.get("ts_length")
        interval = entry.plan.params.get("interval")
        if isinstance(ts_length, int):
            return ts_length, interval if isinstance(interval, int) else 1
    return 6, 1


def should_continue(decision: str) -> bool:
    return decision in {
        "continue",
        "needs_resource_review",
        "retry_with_smaller_batch",
        "retry_with_shorter_sequence",
        "retry_with_longer_sequence",
        "retry_with_spatiotemporal",
        "needs_experiment_upgrade",
        "needs_dataset_generation",
        "needs_data_filtering",
    }


def apply_plan_overrides(
    *,
    plan,
    explicit_attn_version: str | None,
    explicit_mode: str | None,
    explicit_ts_length: int | None,
    explicit_interval: int | None,
    explicit_batch_size: int | None,
    explicit_learning_rate: float | None,
    explicit_num_heads: int | None,
    explicit_embedding_dim: int | None,
    explicit_epochs: int | None,
    explicit_sample_limit: int | None,
):
    params = dict(plan.params)

    if explicit_mode is not None and plan.tool_name.startswith("dataset_gen_"):
        params["mode"] = explicit_mode
    if explicit_sample_limit is not None and plan.tool_name.startswith("dataset_gen_"):
        params["sample_limit"] = explicit_sample_limit

    if explicit_ts_length is not None:
        params["ts_length"] = explicit_ts_length
    if explicit_interval is not None:
        params["interval"] = explicit_interval

    if plan.tool_name.startswith("run_"):
        if explicit_batch_size is not None:
            params["batch_size"] = explicit_batch_size
        if explicit_learning_rate is not None:
            params["learning_rate"] = explicit_learning_rate
        if explicit_num_heads is not None:
            params["num_heads"] = explicit_num_heads
        if explicit_embedding_dim is not None:
            params["embedding_dim"] = explicit_embedding_dim
        if explicit_epochs is not None:
            params["epochs"] = explicit_epochs
        if explicit_attn_version is not None and plan.tool_name == "run_spatial_temp_model":
            params["attn_version"] = explicit_attn_version

    plan.params = params
    return plan


def save_report(state: AnalysisState, reporter: ReportGenerator) -> Path:
    report_text = reporter.build_report(state)
    report_path = reporter.default_report_path(state)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text + "\n")
    return report_path


def run_one_step(
    *,
    state: AnalysisState,
    planner: Planner,
    executor: Executor,
    evaluator: Evaluator,
    inspector: DataInspector,
    explicit_tool: str | None,
    explicit_model: str | None,
    explicit_attn_version: str | None,
    explicit_mode: str | None,
    explicit_ts_length: int | None,
    explicit_interval: int | None,
    explicit_batch_size: int | None,
    explicit_learning_rate: float | None,
    explicit_num_heads: int | None,
    explicit_embedding_dim: int | None,
    explicit_epochs: int | None,
    explicit_sample_limit: int | None,
) -> tuple[str, str]:
    inspect_ts_length, inspect_interval = resolve_inspection_target(
        state=state,
        explicit_ts_length=explicit_ts_length,
        explicit_interval=explicit_interval,
    )
    state.set_data_snapshot(inspector.inspect(task=state.task, ts_length=inspect_ts_length, interval=inspect_interval))

    if explicit_tool:
        plan = planner.make_direct_plan(
            state=state,
            tool_name=explicit_tool,
            model_name=explicit_model,
            attn_version=explicit_attn_version,
            mode=explicit_mode,
            ts_length=explicit_ts_length,
            interval=explicit_interval,
            batch_size=explicit_batch_size,
            learning_rate=explicit_learning_rate,
            num_heads=explicit_num_heads,
            embedding_dim=explicit_embedding_dim,
            epochs=explicit_epochs,
            sample_limit=explicit_sample_limit,
        )
    else:
        plan = planner.next_plan(state=state)
        plan = apply_plan_overrides(
            plan=plan,
            explicit_attn_version=explicit_attn_version,
            explicit_mode=explicit_mode,
            explicit_ts_length=explicit_ts_length,
            explicit_interval=explicit_interval,
            explicit_batch_size=explicit_batch_size,
            explicit_learning_rate=explicit_learning_rate,
            explicit_num_heads=explicit_num_heads,
            explicit_embedding_dim=explicit_embedding_dim,
            explicit_epochs=explicit_epochs,
            explicit_sample_limit=explicit_sample_limit,
        )

    print(f"tool={plan.tool_name}")
    print(f"rationale={plan.rationale}")

    if plan.tool_name == "inspect_only":
        state.save()
        print("status=blocked")
        print("decision=needs_review")
        return "blocked", "needs_review"

    result = executor.execute(plan=plan, state=state)
    evaluation = evaluator.evaluate(state=state, result=result)

    state.record(plan=plan, result=result, evaluation=evaluation)
    state.save()

    print(f"status={result.status}")
    print(f"decision={evaluation.decision}")
    if evaluation.summary:
        print(evaluation.summary)

    return result.status, evaluation.decision


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    state = AnalysisState.load_or_create(state_path=state_path, task=args.task)

    planner = Planner()
    executor = Executor()
    evaluator = Evaluator()
    inspector = DataInspector()
    reporter = ReportGenerator()

    if args.report:
        report_text = reporter.build_report(state)
        report_path = save_report(state, reporter)
        print(report_text)
        print("")
        print(f"report_path={report_path}")
        return

    if args.plan_only:
        inspect_ts_length, inspect_interval = resolve_inspection_target(
            state=state,
            explicit_ts_length=args.ts_length,
            explicit_interval=args.interval,
        )
        state.set_data_snapshot(inspector.inspect(task=state.task, ts_length=inspect_ts_length, interval=inspect_interval))
        if args.tool:
            plan = planner.make_direct_plan(
                state=state,
                tool_name=args.tool,
                model_name=args.model,
                attn_version=args.attn_version,
                mode=args.mode,
                ts_length=args.ts_length,
                interval=args.interval,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_heads=args.num_heads,
                embedding_dim=args.embedding_dim,
                epochs=args.epochs,
                sample_limit=args.sample_limit,
            )
        else:
            plan = planner.next_plan(state=state)
        print(f"tool={plan.tool_name}")
        print(f"rationale={plan.rationale}")
        state.save()
        return

    if args.tool:
        run_one_step(
            state=state,
            planner=planner,
            executor=executor,
            evaluator=evaluator,
            inspector=inspector,
            explicit_tool=args.tool,
            explicit_model=args.model,
            explicit_attn_version=args.attn_version,
            explicit_mode=args.mode,
            explicit_ts_length=args.ts_length,
            explicit_interval=args.interval,
            explicit_batch_size=args.batch_size,
            explicit_learning_rate=args.learning_rate,
            explicit_num_heads=args.num_heads,
            explicit_embedding_dim=args.embedding_dim,
            explicit_epochs=args.epochs,
            explicit_sample_limit=args.sample_limit,
        )
        report_path = save_report(state, reporter)
        print(f"report_path={report_path}")
        return

    for step in range(1, args.max_steps + 1):
        print(f"step={step}")
        _, decision = run_one_step(
            state=state,
            planner=planner,
            executor=executor,
            evaluator=evaluator,
            inspector=inspector,
            explicit_tool=None,
            explicit_model=None,
            explicit_attn_version=None,
            explicit_mode=None,
            explicit_ts_length=args.ts_length,
            explicit_interval=args.interval,
            explicit_batch_size=args.batch_size,
            explicit_learning_rate=args.learning_rate,
            explicit_num_heads=args.num_heads,
            explicit_embedding_dim=args.embedding_dim,
            explicit_epochs=args.epochs,
            explicit_sample_limit=args.sample_limit,
        )
        if not should_continue(decision):
            report_path = save_report(state, reporter)
            print(f"report_path={report_path}")
            return
    print("status=stopped")
    print("decision=max_steps_reached")
    print("Autonomous loop stopped after reaching the configured step limit.")
    report_path = save_report(state, reporter)
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
