from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BOUNDED_METRICS = {"f1", "iou", "dice", "accuracy"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AnalysisPlan:
    tool_name: str
    rationale: str
    params: dict[str, object]


@dataclass
class ExecutionResult:
    tool_name: str
    status: str
    return_code: int
    stdout: str
    stderr: str
    command: list[str]
    artifact_path: str | None = None
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)


@dataclass
class EvaluationResult:
    decision: str
    summary: str
    metrics: dict[str, float]


@dataclass
class DataSnapshot:
    satfire_root: str
    raw_data_root: str
    dataset_root: str
    checkpoints_root: str
    has_raw_data_root: bool
    has_prepared_train: bool
    has_prepared_val: bool
    has_firepred: bool
    firepred_count: int
    raw_fire_count: int
    prepared_files: dict[str, str]
    has_prepared_test: bool = False
    prepared_test_count: int = 0


@dataclass
class HistoryEntry:
    timestamp: str
    plan: AnalysisPlan
    result: ExecutionResult
    evaluation: EvaluationResult


@dataclass
class AnalysisState:
    task: str
    state_path: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    data_snapshot: DataSnapshot | None = None
    history: list[HistoryEntry] = field(default_factory=list)

    @classmethod
    def load_or_create(cls, state_path: Path, task: str) -> "AnalysisState":
        if state_path.exists():
            payload = json.loads(state_path.read_text())
            stored_task = payload.get("task", task)
            if stored_task != task:
                return cls(
                    task=task,
                    state_path=str(state_path),
                )
            history = [
                HistoryEntry(
                    timestamp=item["timestamp"],
                    plan=AnalysisPlan(**item["plan"]),
                    result=ExecutionResult(**item["result"]),
                    evaluation=EvaluationResult(**item["evaluation"]),
                )
                for item in payload.get("history", [])
            ]
            data_snapshot_payload = payload.get("data_snapshot")
            return cls(
                task=task,
                state_path=str(state_path),
                created_at=payload.get("created_at", utc_now()),
                updated_at=payload.get("updated_at", utc_now()),
                data_snapshot=DataSnapshot(**data_snapshot_payload) if data_snapshot_payload else None,
                history=history,
            )

        return cls(task=task, state_path=str(state_path))

    def set_data_snapshot(self, snapshot: DataSnapshot) -> None:
        self.data_snapshot = snapshot
        self.updated_at = utc_now()

    def record(
        self,
        plan: AnalysisPlan,
        result: ExecutionResult,
        evaluation: EvaluationResult,
    ) -> None:
        self.updated_at = utc_now()
        self.history.append(
            HistoryEntry(
                timestamp=self.updated_at,
                plan=plan,
                result=result,
                evaluation=evaluation,
            )
        )

    def save(self) -> None:
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task,
            "state_path": self.state_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data_snapshot": asdict(self.data_snapshot) if self.data_snapshot else None,
            "history": [asdict(item) for item in self.history],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)

    def primary_metric_score(self, metrics: dict[str, float]) -> float | None:
        for key in ("f1", "iou", "dice", "accuracy"):
            if key in metrics and self._is_valid_metric(key, metrics[key]):
                return metrics[key]
        if metrics:
            valid_values = [value for key, value in metrics.items() if self._is_valid_metric(key, value)]
            if valid_values:
                return max(valid_values)
        return None

    def _is_valid_metric(self, name: str, value: float) -> bool:
        if name in BOUNDED_METRICS:
            return 0.0 <= value <= 1.0
        return True

    def best_metric_for_tool(self, tool_name: str) -> float | None:
        scores: list[float] = []
        for entry in self.history:
            if entry.plan.tool_name != tool_name:
                continue
            score = self.primary_metric_score(entry.evaluation.metrics)
            if score is not None:
                scores.append(score)
        return max(scores) if scores else None

    def latest_metric_for_tool(self, tool_name: str) -> float | None:
        for entry in reversed(self.history):
            if entry.plan.tool_name != tool_name:
                continue
            return self.primary_metric_score(entry.evaluation.metrics)
        return None

    def experiment_memory(self) -> dict[str, object]:
        tool_scores: dict[str, float] = {}
        for entry in self.history:
            score = self.primary_metric_score(entry.evaluation.metrics)
            if score is None:
                continue
            current = tool_scores.get(entry.plan.tool_name)
            if current is None or score > current:
                tool_scores[entry.plan.tool_name] = score
        return {
            "best_scores_by_tool": tool_scores,
            "history_length": len(self.history),
        }
