from __future__ import annotations

from dataclasses import dataclass

from schemas.state import AnalysisPlan, AnalysisState


@dataclass
class Planner:
    """Chooses the next action. The first MVP uses deterministic policies."""

    def next_plan(self, state: AnalysisState) -> AnalysisPlan:
        snapshot = state.data_snapshot
        if snapshot is None:
            raise ValueError("Planner requires a data snapshot before routing.")

        if not snapshot.has_raw_data_root:
            return AnalysisPlan(
                tool_name="inspect_only",
                rationale="Raw data root is unavailable, so execution should stop for manual setup.",
                params={},
            )

        if state.task == "pred":
            if not snapshot.has_firepred:
                return AnalysisPlan(
                    tool_name="inspect_only",
                    rationale="Prediction task requested but no FirePred folders were detected.",
                    params={},
                )
            if not snapshot.has_prepared_train or not snapshot.has_prepared_val:
                return AnalysisPlan(
                    tool_name="dataset_gen_pred",
                    rationale="Prediction arrays are missing, so generate prepared datasets first.",
                    params={"mode": "train"},
                )
            return AnalysisPlan(
                tool_name="run_spatial_temp_model_pred",
                rationale="Prediction task defaults to the prediction-oriented spatial-temporal path.",
                params={},
            )

        if not snapshot.has_prepared_train or not snapshot.has_prepared_val:
            return AnalysisPlan(
                tool_name="dataset_gen_afba",
                rationale="AF/BA prepared arrays are missing, so dataset generation must happen first.",
                params={"mode": "train"},
            )

        if not state.history:
            return AnalysisPlan(
                tool_name="run_spatial_model",
                rationale="Start with the cheapest spatial baseline before escalating.",
                params={},
            )

        last_eval = state.history[-1].evaluation
        last_result = state.history[-1].result
        if last_eval.decision == "retry_with_smaller_batch":
            return AnalysisPlan(
                tool_name=last_result.tool_name,
                rationale="Retry the same tool with a smaller batch size after an OOM-like failure.",
                params={"batch_size": 1},
            )
        if last_eval.decision == "retry_with_shorter_sequence":
            return AnalysisPlan(
                tool_name="dataset_gen_pred" if state.task == "pred" else "dataset_gen_afba",
                rationale="Reduce temporal window length because the sequence was too short for the current configuration.",
                params={"mode": "train", "ts_length": 4},
            )
        if last_eval.decision == "retry_with_spatiotemporal":
            return AnalysisPlan(
                tool_name="run_spatial_temp_model",
                rationale="Escalate to the stronger spatiotemporal model after weak baseline metrics.",
                params={},
            )

        return AnalysisPlan(
            tool_name="run_spatial_model",
            rationale="No escalation trigger found, reuse the baseline path.",
            params={},
        )

    def make_direct_plan(
        self,
        state: AnalysisState,
        tool_name: str,
        model_name: str | None = None,
        mode: str | None = None,
        ts_length: int | None = None,
        interval: int | None = None,
        batch_size: int | None = None,
        sample_limit: int | None = None,
    ) -> AnalysisPlan:
        params = {}
        if model_name:
            params["model"] = model_name
        if mode:
            params["mode"] = mode
        if ts_length is not None:
            params["ts_length"] = ts_length
        if interval is not None:
            params["interval"] = interval
        if batch_size is not None:
            params["batch_size"] = batch_size
        if sample_limit is not None:
            params["sample_limit"] = sample_limit
        return AnalysisPlan(
            tool_name=tool_name,
            rationale="Direct tool selection requested by the operator.",
            params=params,
        )
