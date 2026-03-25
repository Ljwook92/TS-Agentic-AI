from __future__ import annotations

from dataclasses import dataclass

from schemas.state import AnalysisState, HistoryEntry


PRIMARY_METRIC_ORDER = ("f1", "iou", "dice", "accuracy")
BOUNDED_METRICS = {"f1", "iou", "dice", "accuracy"}


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
            previous_entry = state.history[idx - 2] if idx > 1 else None

            if current_metric is None:
                lines.append(
                    f"Step {idx}: {entry.plan.tool_name} was chosen because {entry.plan.rationale} Decision: {entry.evaluation.decision}."
                )
                continue

            metric_name, metric_value = current_metric
            if previous_entry and previous_entry.evaluation.decision == "needs_debug" and previous_entry.plan.tool_name == entry.plan.tool_name:
                lines.append(
                    f"Step {idx}: {entry.plan.tool_name} produced the first usable metric after a prior debug failure, yielding {metric_name}={metric_value:.4f}."
                )
            elif previous_metric is None:
                lines.append(
                    f"Step {idx}: {entry.plan.tool_name} established the first measurable result with {metric_name}={metric_value:.4f}."
                )
            else:
                prev_name, prev_value = previous_metric
                if prev_name != metric_name:
                    lines.append(
                        f"Step {idx}: {entry.plan.tool_name} changed the primary metric from {prev_name} to {metric_name}; the latest usable score was {metric_name}={metric_value:.4f}."
                    )
                else:
                    delta = metric_value - prev_value
                    if delta > 0:
                        lines.append(
                            f"Step {idx}: {entry.plan.tool_name} improved {metric_name} by {delta:.4f}{self._summarize_param_change(state, idx)}"
                        )
                    elif delta < 0:
                        lines.append(
                            f"Step {idx}: {entry.plan.tool_name} reduced {metric_name} by {abs(delta):.4f}, suggesting the latest change was not helpful."
                        )
                    else:
                        lines.append(
                            f"Step {idx}: {entry.plan.tool_name} produced no material change in {metric_name}; the configuration likely plateaued."
                        )
            previous_metric = current_metric
        return lines

    def _summarize_param_change(self, state: AnalysisState, idx: int) -> str:
        entry = state.history[idx - 1]
        previous_entry = state.history[idx - 2] if idx > 1 else None
        if previous_entry is None:
            return "."

        changed_keys: list[str] = []
        all_keys = list(dict.fromkeys([*previous_entry.plan.params.keys(), *entry.plan.params.keys()]))
        for key in all_keys:
            if previous_entry.plan.params.get(key) != entry.plan.params.get(key):
                changed_keys.append(f"{key}={entry.plan.params.get(key)}")
        if changed_keys:
            return f" after changing {', '.join(changed_keys)}."
        if previous_entry.plan.tool_name != entry.plan.tool_name:
            return f" after switching from {previous_entry.plan.tool_name}."
        return " after re-running the same configuration."

    def _best_entry(self, state: AnalysisState) -> HistoryEntry | None:
        metric_entries = [entry for entry in state.history if self._primary_metric_item(entry) is not None]
        if not metric_entries:
            return None
        return max(metric_entries, key=lambda entry: self._primary_metric_item(entry)[1])  # type: ignore[index]

    def _primary_metric_item(self, entry: HistoryEntry) -> tuple[str, float] | None:
        metrics = self._normalized_metrics(entry.evaluation.metrics)
        for key in PRIMARY_METRIC_ORDER:
            if key in metrics:
                return key, metrics[key]
        if metrics:
            key = max(metrics, key=metrics.get)
            return key, metrics[key]
        return None

    def _normalized_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in metrics.items():
            if key in BOUNDED_METRICS and not (0.0 <= value <= 1.0):
                continue
            normalized[key] = value
        return normalized

    def _invalid_metric_items(self, metrics: dict[str, float]) -> list[tuple[str, float]]:
        invalid: list[tuple[str, float]] = []
        for key, value in metrics.items():
            if key in BOUNDED_METRICS and not (0.0 <= value <= 1.0):
                invalid.append((key, value))
        return invalid

    def _format_params(self, params: dict[str, object]) -> str:
        if not params:
            return "-"
        preferred = ("model", "attn_version", "num_heads", "embedding_dim", "learning_rate", "ts_length", "interval", "batch_size", "epochs", "mode", "sample_limit", "channels")
        keys = [key for key in preferred if key in params]
        keys.extend(key for key in params if key not in keys)
        return ", ".join(f"{key}={params[key]}" for key in keys)

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        normalized = self._normalized_metrics(metrics)
        invalid = self._invalid_metric_items(metrics)
        parts: list[str] = []
        ordered_keys = [key for key in PRIMARY_METRIC_ORDER if key in normalized]
        ordered_keys.extend(key for key in normalized if key not in ordered_keys)
        parts.extend(f"{key}={normalized[key]:.4f}" for key in ordered_keys)
        parts.extend(f"suspect_{key}={value:.4f}" for key, value in invalid)
        return ", ".join(parts) if parts else "-"
