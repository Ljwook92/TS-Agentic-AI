from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.state import EvaluationResult, ExecutionResult, AnalysisState


METRIC_PATTERN = re.compile(r"(iou|dice|f1|accuracy)[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
OOM_PATTERN = re.compile(r"(out of memory|cuda oom|cudnn_status_alloc_failed)", re.IGNORECASE)
SHORT_TS_PATTERN = re.compile(r"(No enough TS|empty file list)", re.IGNORECASE)
SHAPE_PATTERN = re.compile(r"(shape|size mismatch|expected.*channels|band)", re.IGNORECASE)
DATA_AVAILABILITY_PATTERN = re.compile(r"(No valid prediction sequences were generated|empty file list)", re.IGNORECASE)
MISSING_DATASET_PATTERN = re.compile(r"(FileNotFoundError|No such file or directory).*(dataset_train|dataset_val|dataset_test)", re.IGNORECASE | re.DOTALL)
PRED_TEST_SUMMARY_PATTERNS = [
    re.compile(
        r"model\s+F1\s+Score:\s*([0-9]*\.?[0-9]+)\s*and\s*iou\s+score:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Model\s+Test\s+F1\s+Score:\s*([0-9]*\.?[0-9]+)\s+and\s+Test\s+IoU\s+Score:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    ),
]
BOUNDED_METRICS = {"f1", "iou", "dice", "accuracy"}


@dataclass
class Evaluator:
    metric_floor: float = 0.30
    target_iou: float = 0.60
    improvement_margin: float = 0.01

    def evaluate(self, state: AnalysisState, result: ExecutionResult) -> EvaluationResult:
        combined_output = result.stdout + "\n" + result.stderr

        if result.return_code != 0:
            if result.return_code == -9:
                return EvaluationResult(
                    decision="needs_resource_review",
                    summary="Run was killed by the system, likely due to memory or resource limits. Retry with a smaller scope or smaller batch.",
                    metrics={},
                )
            if DATA_AVAILABILITY_PATTERN.search(combined_output):
                return EvaluationResult(
                    decision="needs_data_filtering",
                    summary="Run failed because the selected locations do not contain usable prediction inputs. Filter for locations with actual VIIRS_Day and FirePred tif files.",
                    metrics={},
                )
            if MISSING_DATASET_PATTERN.search(combined_output):
                return EvaluationResult(
                    decision="needs_dataset_generation",
                    summary="Run failed because a prepared dataset file is missing. Generate the required train/val/test arrays before model execution.",
                    metrics={},
                )
            if OOM_PATTERN.search(combined_output):
                return EvaluationResult(
                    decision="retry_with_smaller_batch",
                    summary="Run failed with an OOM-like error. Retry with a smaller batch size.",
                    metrics={},
                )
            if SHORT_TS_PATTERN.search(combined_output):
                return EvaluationResult(
                    decision="retry_with_shorter_sequence",
                    summary="Run failed because the requested time series window is unavailable.",
                    metrics={},
                )
            if SHAPE_PATTERN.search(combined_output):
                return EvaluationResult(
                    decision="needs_debug",
                    summary="Run failed with a shape or channel mismatch. Inspect dataset generation and channel assumptions.",
                    metrics={},
                )
            return EvaluationResult(
                decision="needs_debug",
                summary="Legacy tool exited with a non-zero code. Inspect stderr and parameter compatibility.",
                metrics={},
            )

        if result.tool_name.startswith("dataset_gen_"):
            return EvaluationResult(
                decision="continue",
                summary="Dataset generation completed. The planner can proceed to model execution.",
                metrics={},
            )

        if result.tool_name == "run_spatial_temp_model_pred":
            pred_test_metrics = self._extract_pred_test_metrics(combined_output)
            if pred_test_metrics is not None:
                metrics = pred_test_metrics
            else:
                return EvaluationResult(
                    decision="needs_review",
                    summary="Prediction run completed without explicit held-out test metrics. The current script appears to expose validation logs only, so treat this run as incomplete for final comparison.",
                    metrics={},
                )
        else:
            metrics = self._extract_metrics(combined_output)
        if not metrics:
            return EvaluationResult(
                decision="needs_review",
                summary="Run completed without normalized metrics. Manual inspection or parser extension is needed.",
                metrics={},
            )

        primary_score = state.primary_metric_score(metrics) or max(metrics.values())
        previous_best = state.best_metric_for_tool(result.tool_name)
        iou_score = metrics.get("iou")

        if primary_score < self.metric_floor and result.tool_name == "run_spatial_model":
            return EvaluationResult(
                decision="retry_with_spatiotemporal",
                summary="Baseline metrics are weak. Escalate to the spatiotemporal path.",
                metrics=metrics,
            )

        if primary_score < self.metric_floor and result.tool_name in {"run_spatial_temp_model", "run_spatial_temp_model_pred", "run_seq_model"}:
            return EvaluationResult(
                decision="retry_with_longer_sequence",
                summary="Metrics remain weak after the current model choice. Retry with a longer temporal window.",
                metrics=metrics,
            )

        if iou_score is not None and iou_score < self.target_iou:
            if result.tool_name == "run_spatial_model":
                return EvaluationResult(
                    decision="retry_with_spatiotemporal",
                    summary=f"IoU remains below the target threshold ({iou_score:.4f} < {self.target_iou:.2f}). Escalate to the spatiotemporal path.",
                    metrics=metrics,
                )
            if result.tool_name in {"run_spatial_temp_model", "run_spatial_temp_model_pred", "run_seq_model"}:
                return EvaluationResult(
                    decision="needs_experiment_upgrade",
                    summary=f"IoU remains below the target threshold ({iou_score:.4f} < {self.target_iou:.2f}). Try a stronger model or a different temporal setting.",
                    metrics=metrics,
                )

        if previous_best is not None and primary_score <= previous_best + self.improvement_margin:
            return EvaluationResult(
                decision="needs_experiment_upgrade",
                summary="The latest run did not materially improve over previous results. Try a stronger model or a different temporal setting.",
                metrics=metrics,
            )

        return EvaluationResult(
            decision="complete",
            summary="Run completed with usable metrics.",
            metrics=metrics,
        )

    def _extract_metrics(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, value in METRIC_PATTERN.findall(text):
            try:
                numeric_value = float(value)
            except ValueError:
                continue
            lowered = name.lower()
            if lowered in BOUNDED_METRICS and not (0.0 <= numeric_value <= 1.0):
                continue
            metrics[lowered] = numeric_value
        return metrics

    def _extract_pred_test_metrics(self, text: str) -> dict[str, float] | None:
        for pattern in PRED_TEST_SUMMARY_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            f1_value = float(match.group(1))
            iou_value = float(match.group(2))
            metrics = {"f1": f1_value, "iou": iou_value}
            if any(not (0.0 <= value <= 1.0) for value in metrics.values()):
                return None
            return metrics
        return None
