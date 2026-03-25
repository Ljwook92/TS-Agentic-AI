from __future__ import annotations

from dataclasses import dataclass

from schemas.state import AnalysisState, HistoryEntry


PRIMARY_METRIC_ORDER = ("f1", "iou", "dice", "accuracy")


@dataclass
class ReportGenerator:
    def build_report(self, state: AnalysisState) -> str:
        lines: list[str] = []
        lines.append(f"Task: {state.task}")
        lines.append(f"Total steps: {len(state.history)}")
        lines.append("")
        lines.extend(self._summary_lines(state))
        lines.append("")
        lines.extend(self._comparison_table(state))
        lines.append("")
        lines.extend(self._reasoning_lines(state))
        return "\n".join(lines).rstrip()

    def _summary_lines(self, state: AnalysisState) -> list[str]:
        best_entry = self._best_entry(state)
        if best_entry is None:
            return ["Summary", "No completed metric-bearing runs were found in this state."]

        primary_name, primary_value = self._primary_metric_item(best_entry)
        return [
            "Summary",
            f"Best run: {best_entry.plan.tool_name} with {primary_name}={primary_value:.4f}",
            f"Best rationale: {best_entry.plan.rationale}",
            f"Best evaluator decision: {best_entry.evaluation.decision}",
        ]

    def _comparison_table(self, state: AnalysisState) -> list[str]:
        lines = [
            "Comparison",
            "| Step | Tool | Key Params | Decision | Metrics |",
            "| --- | --- | --- | --- | --- |",
        ]
        for idx, entry in enumerate(state.history, 1):
            key_params = self._format_params(entry.plan.params)
            metrics = self._format_metrics(entry.evaluation.metrics)
            lines.append(
                f"| {idx} | {entry.plan.tool_name} | {key_params} | {entry.evaluation.decision} | {metrics} |"
            )
        return lines

    def _reasoning_lines(self, state: AnalysisState) -> list[str]:
        lines = ["Reasoning"]
        previous_metric: tuple[str, float] | None = None
        for idx, entry in enumerate(state.history, 1):
            current_metric = self._primary_metric_item(entry)
            if current_metric is None:
                lines.append(
                    f"Step {idx}: {entry.plan.tool_name} was chosen because {entry.plan.rationale}"
                )
                continue

            metric_name, metric_value = current_metric
            if previous_metric is None:
                lines.append(
                    f"Step {idx}: {entry.plan.tool_name} established the first measurable result with {metric_name}={metric_value:.4f}."
                )
            else:
                prev_name, prev_value = previous_metric
                delta = metric_value - prev_value
                if delta > 0:
                    lines.append(
                        f"Step {idx}: {entry.plan.tool_name} improved {metric_name} by {delta:.4f} versus the previous measured run by changing {self._format_params(entry.plan.params)}."
                    )
                elif delta < 0:
                    lines.append(
                        f"Step {idx}: {entry.plan.tool_name} reduced {metric_name} by {abs(delta):.4f} versus the previous measured run, suggesting the latest change was not helpful."
                    )
                else:
                    lines.append(
                        f"Step {idx}: {entry.plan.tool_name} produced no material change in {metric_name}; the configuration likely plateaued."
                    )
            previous_metric = current_metric
        return lines

    def _best_entry(self, state: AnalysisState) -> HistoryEntry | None:
        metric_entries = [entry for entry in state.history if self._primary_metric_item(entry) is not None]
        if not metric_entries:
            return None
        return max(metric_entries, key=lambda entry: self._primary_metric_item(entry)[1])  # type: ignore[index]

    def _primary_metric_item(self, entry: HistoryEntry) -> tuple[str, float] | None:
        metrics = entry.evaluation.metrics
        for key in PRIMARY_METRIC_ORDER:
            if key in metrics:
                return key, metrics[key]
        if metrics:
            key = max(metrics, key=metrics.get)
            return key, metrics[key]
        return None

    def _format_params(self, params: dict[str, object]) -> str:
        if not params:
            return "-"
        preferred = ("model", "ts_length", "interval", "batch_size", "epochs", "mode", "sample_limit", "channels")
        keys = [key for key in preferred if key in params]
        keys.extend(key for key in params if key not in keys)
        return ", ".join(f"{key}={params[key]}" for key in keys)

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        if not metrics:
            return "-"
        ordered_keys = [key for key in PRIMARY_METRIC_ORDER if key in metrics]
        ordered_keys.extend(key for key in metrics if key not in ordered_keys)
        return ", ".join(f"{key}={metrics[key]:.4f}" for key in ordered_keys)
