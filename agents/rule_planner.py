from __future__ import annotations

from dataclasses import dataclass

from schemas.state import AnalysisPlan, AnalysisState


@dataclass
class RulePlanner:
    """Deterministic fallback planner used when no LLM is configured."""

    def _literature_basis(self, task: str, theme: str) -> str:
        if theme == "spatiotemporal_upgrade":
            if task == "pred":
                return (
                    "Literature basis: TS-SatFire defines spatial-temporal benchmarks for this dataset, "
                    "and Using_Deep_Learning_for_Spatial_and_Temporal_Analysis_of_Wildfire_Start_and_Progression "
                    "supports moving from spatial-only baselines to models that explicitly capture temporal dependence."
                )
            return (
                "Literature basis: TS-SatFire and Near real-time wildfire progression mapping with VIIRS time-series and autoregressive SwinUNETR "
                "both support using spatiotemporal models when spatial baselines underperform on wildfire progression mapping."
            )
        if theme == "longer_sequence":
            if task == "pred":
                return (
                    "Literature basis: TS-SatFire and Wildfire_Progression_Prediction_and_Validation_Using_Satellite_Data_and_Remote_Sensing_in_Sonoma_California "
                    "both motivate testing longer temporal context when short-window prediction remains weak."
                )
            return (
                "Literature basis: TS-SatFire and Wildfire_Progression_Time_Series_Mapping_With_Interferometric_Synthetic_Aperture_Radar_InSAR "
                "both motivate trying longer temporal windows when current progression metrics remain weak."
            )
        if theme == "attention_upgrade":
            return (
                "Literature basis: the VIIRS time-series SwinUNETR progression literature supports stronger temporal attention before only extending sequence length."
            )
        if theme == "capacity_upgrade":
            return (
                "Literature basis: after temporal modeling is selected, stronger model capacity is a cleaner next ablation than immediately mixing in a longer sequence window."
            )
        if theme == "lr_upgrade":
            return (
                "Literature basis: optimization changes should be tried before attributing weak performance entirely to temporal context length."
            )
        return "Literature basis: TS-SatFire remains the strongest source of truth for this workflow."

    def _next_dataset_mode(self, state: AnalysisState) -> str:
        snapshot = state.data_snapshot
        if snapshot is None:
            return "train"
        if not snapshot.has_prepared_train:
            return "train"
        if not snapshot.has_prepared_val:
            return "val"
        if not snapshot.has_prepared_test:
            return "test"
        return "train"

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
            if not snapshot.has_prepared_train or not snapshot.has_prepared_val or not snapshot.has_prepared_test:
                mode = self._next_dataset_mode(state)
                return AnalysisPlan(
                    tool_name="dataset_gen_pred",
                    rationale=f"Prediction arrays for split '{mode}' are missing, so generate prepared datasets first.",
                    params={"mode": mode},
                )
            return AnalysisPlan(
                tool_name="run_spatial_temp_model_pred",
                rationale="Prediction task defaults to the prediction-oriented spatial-temporal path.",
                params={},
            )

        if not snapshot.has_prepared_train or not snapshot.has_prepared_val or not snapshot.has_prepared_test:
            mode = self._next_dataset_mode(state)
            return AnalysisPlan(
                tool_name="dataset_gen_afba",
                rationale=f"AF/BA prepared arrays for split '{mode}' are missing, so dataset generation must happen first.",
                params={"mode": mode},
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
            mode = self._next_dataset_mode(state)
            return AnalysisPlan(
                tool_name="dataset_gen_pred" if state.task == "pred" else "dataset_gen_afba",
                rationale="Reduce temporal window length because the sequence was too short for the current configuration.",
                params={"mode": mode, "ts_length": 4},
            )
        if last_eval.decision == "needs_dataset_generation":
            mode = self._next_dataset_mode(state)
            return AnalysisPlan(
                tool_name="dataset_gen_pred" if state.task == "pred" else "dataset_gen_afba",
                rationale=f"Prepared arrays for split '{mode}' are still missing. Generate them before retrying the model path.",
                params={"mode": mode},
            )
        if last_eval.decision == "retry_with_spatiotemporal":
            return AnalysisPlan(
                tool_name="run_spatial_temp_model",
                rationale=(
                    "Escalate to the stronger spatiotemporal model after weak baseline metrics. "
                    + self._literature_basis(state.task, "spatiotemporal_upgrade")
                ),
                params={"attn_version": "v1", "embedding_dim": 48, "num_heads": 4, "learning_rate": 0.0001},
            )
        if last_eval.decision == "retry_with_longer_sequence":
            next_ts = 6
            if state.history:
                last_ts = state.history[-1].plan.params.get("ts_length")
                if isinstance(last_ts, int) and last_ts >= 6:
                    next_ts = min(last_ts + 2, 10)
            return AnalysisPlan(
                tool_name="dataset_gen_pred" if state.task == "pred" else "dataset_gen_afba",
                rationale=(
                    f"Metrics are still weak, so regenerate datasets with a longer temporal window (ts_length={next_ts}). "
                    + self._literature_basis(state.task, "longer_sequence")
                ),
                params={"mode": "train", "ts_length": next_ts, "interval": 1},
            )
        if last_eval.decision == "needs_experiment_upgrade":
            last_tool = last_result.tool_name
            last_ts = last_result.command[last_result.command.index("-ts") + 1] if "-ts" in last_result.command else None
            if last_tool == "run_spatial_model":
                return AnalysisPlan(
                    tool_name="run_spatial_temp_model",
                    rationale=(
                        "The baseline stopped improving, so upgrade to the spatiotemporal model. "
                        + self._literature_basis(state.task, "spatiotemporal_upgrade")
                    ),
                    params={"ts_length": int(last_ts) if last_ts else 4, "attn_version": "v1", "embedding_dim": 48, "num_heads": 4, "learning_rate": 0.0001},
                )
            if last_tool in {"run_spatial_temp_model", "run_spatial_temp_model_pred", "run_seq_model"}:
                last_attn_version = None
                if "-av" in last_result.command:
                    last_attn_version = last_result.command[last_result.command.index("-av") + 1]
                last_lr = None
                if "-lr" in last_result.command:
                    last_lr = last_result.command[last_result.command.index("-lr") + 1]
                last_hidden = None
                if "-ed" in last_result.command:
                    last_hidden = last_result.command[last_result.command.index("-ed") + 1]
                if last_tool == "run_spatial_temp_model" and last_attn_version != "v2":
                    return AnalysisPlan(
                        tool_name="run_spatial_temp_model",
                        rationale=(
                            "The spatiotemporal model plateaued, so switch attention from v1 to v2 before changing the temporal window. "
                            + self._literature_basis(state.task, "attention_upgrade")
                        ),
                        params={"ts_length": int(last_ts) if last_ts else 4, "attn_version": "v2", "embedding_dim": int(last_hidden) if last_hidden else 48, "learning_rate": float(last_lr) if last_lr else 0.0001},
                    )
                if last_tool == "run_spatial_temp_model_pred" and last_attn_version != "v2":
                    return AnalysisPlan(
                        tool_name="run_spatial_temp_model_pred",
                        rationale=(
                            "The prediction spatiotemporal model plateaued, so switch attention from v1 to v2 before changing the temporal window. "
                            + self._literature_basis(state.task, "attention_upgrade")
                        ),
                        params={"ts_length": int(last_ts) if last_ts else 4, "attn_version": "v2", "embedding_dim": int(last_hidden) if last_hidden else 48, "learning_rate": float(last_lr) if last_lr else 0.0001},
                    )
                if last_tool in {"run_spatial_temp_model", "run_spatial_temp_model_pred"} and (last_hidden is None or int(last_hidden) < 64):
                    return AnalysisPlan(
                        tool_name=last_tool,
                        rationale=(
                            "The current temporal model plateaued, so increase model capacity before extending the sequence length. "
                            + self._literature_basis(state.task, "capacity_upgrade")
                        ),
                        params={
                            "ts_length": int(last_ts) if last_ts else 4,
                            "attn_version": last_attn_version or "v2",
                            "embedding_dim": 64,
                            "num_heads": 4,
                            "learning_rate": float(last_lr) if last_lr else 0.0001,
                        },
                    )
                if last_tool in {"run_spatial_temp_model", "run_spatial_temp_model_pred"} and (last_lr is None or float(last_lr) >= 0.0001):
                    return AnalysisPlan(
                        tool_name=last_tool,
                        rationale=(
                            "The current temporal model plateaued again, so lower the learning rate before extending the sequence length. "
                            + self._literature_basis(state.task, "lr_upgrade")
                        ),
                        params={
                            "ts_length": int(last_ts) if last_ts else 4,
                            "attn_version": last_attn_version or "v2",
                            "embedding_dim": int(last_hidden) if last_hidden else 64,
                            "num_heads": 4,
                            "learning_rate": 0.00005,
                        },
                    )
                next_ts = 6
                if last_ts is not None:
                    try:
                        next_ts = min(int(last_ts) + 2, 10)
                    except ValueError:
                        next_ts = 6
                return AnalysisPlan(
                    tool_name="dataset_gen_pred" if state.task == "pred" else "dataset_gen_afba",
                    rationale=(
                        f"The current experiment plateaued, so regenerate datasets with a larger temporal window (ts_length={next_ts}). "
                        + self._literature_basis(state.task, "longer_sequence")
                    ),
                    params={"mode": "train", "ts_length": next_ts, "interval": 1},
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
        attn_version: str | None = None,
        mode: str | None = None,
        ts_length: int | None = None,
        interval: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        num_heads: int | None = None,
        embedding_dim: int | None = None,
        epochs: int | None = None,
        sample_limit: int | None = None,
    ) -> AnalysisPlan:
        params = {}
        if model_name:
            params["model"] = model_name
        if attn_version:
            params["attn_version"] = attn_version
        if mode:
            params["mode"] = mode
        if ts_length is not None:
            params["ts_length"] = ts_length
        if interval is not None:
            params["interval"] = interval
        if batch_size is not None:
            params["batch_size"] = batch_size
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if num_heads is not None:
            params["num_heads"] = num_heads
        if embedding_dim is not None:
            params["embedding_dim"] = embedding_dim
        if epochs is not None:
            params["epochs"] = epochs
        if sample_limit is not None:
            params["sample_limit"] = sample_limit
        return AnalysisPlan(
            tool_name=tool_name,
            rationale="Direct tool selection requested by the operator.",
            params=params,
        )
