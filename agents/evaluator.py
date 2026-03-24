from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.state import EvaluationResult, ExecutionResult, AnalysisState


METRIC_PATTERN = re.compile(r"(iou|dice|f1|accuracy)[^0-9]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
OOM_PATTERN = re.compile(r"(out of memory|cuda oom|cudnn_status_alloc_failed)", re.IGNORECASE)
SHORT_TS_PATTERN = re.compile(r"(No enough TS|empty file list)", re.IGNORECASE)
SHAPE_PATTERN = re.compile(r"(shape|size mismatch|expected.*channels|band)", re.IGNORECASE)
DATA_AVAILABILITY_PATTERN = re.compile(r"(No valid prediction sequences were generated|empty file list)", re.IGNORECASE)


@dataclass
class Evaluator:
    metric_floor: float = 0.30

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

        metrics = self._extract_metrics(combined_output)
        if not metrics:
            return EvaluationResult(
                decision="needs_review",
                summary="Run completed without normalized metrics. Manual inspection or parser extension is needed.",
                metrics={},
            )

        best_metric = max(metrics.values())
        if best_metric < self.metric_floor and result.tool_name == "run_spatial_model":
            return EvaluationResult(
                decision="retry_with_spatiotemporal",
                summary="Baseline metrics are weak. Escalate to the spatiotemporal path.",
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
                metrics[name.lower()] = float(value)
            except ValueError:
                continue
        return metrics
