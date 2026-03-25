from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from agents.rule_planner import RulePlanner
from schemas.state import AnalysisPlan, AnalysisState
from tools.knowledge_loader import load_planner_brief


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass
class Planner:
    """LLM-backed planner with deterministic fallback."""

    fallback: RulePlanner = field(default_factory=RulePlanner)

    def next_plan(self, state: AnalysisState) -> AnalysisPlan:
        if not self.is_llm_enabled():
            return self._normalize_plan(self.fallback.next_plan(state), state)

        try:
            return self._llm_plan(state)
        except Exception as exc:
            fallback_plan = self.fallback.next_plan(state)
            return self._normalize_plan(AnalysisPlan(
                tool_name=fallback_plan.tool_name,
                rationale=f"LLM planner failed and fell back to rule planner: {exc}",
                params=fallback_plan.params,
            ), state)

    def make_direct_plan(
        self,
        state: AnalysisState,
        tool_name: str,
        model_name: str | None = None,
        mode: str | None = None,
        ts_length: int | None = None,
        interval: int | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        sample_limit: int | None = None,
    ) -> AnalysisPlan:
        return self.fallback.make_direct_plan(
            state=state,
            tool_name=tool_name,
            model_name=model_name,
            mode=mode,
            ts_length=ts_length,
            interval=interval,
            batch_size=batch_size,
            epochs=epochs,
            sample_limit=sample_limit,
        )

    def is_llm_enabled(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _llm_plan(self, state: AnalysisState) -> AnalysisPlan:
        snapshot = state.data_snapshot
        if snapshot is None:
            raise ValueError("Planner requires a data snapshot before routing.")

        recent_history = [
            {
                "tool_name": entry.plan.tool_name,
                "decision": entry.evaluation.decision,
                "summary": entry.evaluation.summary,
            }
            for entry in state.history[-3:]
        ]

        system_prompt = (
            "You are an LLM planner for a remote-sensing agent system. "
            "Return strict JSON with keys: tool_name, rationale, params. "
            "Only choose from: inspect_only, dataset_gen_afba, dataset_gen_pred, "
            "run_spatial_model, run_seq_model, run_spatial_temp_model, run_spatial_temp_model_pred. "
            "Keep params minimal and only include keys needed for the next action. "
            "Prefer the smallest valid next action instead of a long-term plan. "
            "If train/val datasets exist but test arrays are missing, prefer dataset generation with mode=test before model execution. "
            "Do not restate the full context."
        )

        user_payload = {
            "task": state.task,
            "data_snapshot": {
                "has_raw_data_root": snapshot.has_raw_data_root,
                "has_prepared_train": snapshot.has_prepared_train,
                "has_prepared_val": snapshot.has_prepared_val,
                "has_prepared_test": snapshot.has_prepared_test,
                "prepared_test_count": snapshot.prepared_test_count,
                "has_firepred": snapshot.has_firepred,
                "firepred_count": snapshot.firepred_count,
                "raw_fire_count": snapshot.raw_fire_count,
            },
            "recent_history": recent_history,
            "planner_brief": load_planner_brief(),
        }

        response = self._chat_completion(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, indent=2),
        )
        parsed = json.loads(response)
        plan = AnalysisPlan(
            tool_name=parsed["tool_name"],
            rationale=parsed.get("rationale", "LLM planner selected the next action."),
            params=parsed.get("params", {}),
        )
        return self._normalize_plan(plan, state)

    def _normalize_plan(self, plan: AnalysisPlan, state: AnalysisState) -> AnalysisPlan:
        snapshot = state.data_snapshot

        if plan.tool_name == "inspect_only" and snapshot is not None:
            blocking_inspect = (not snapshot.has_raw_data_root) or (
                state.task == "pred" and not snapshot.has_firepred
            )
            if not blocking_inspect:
                return self._normalize_plan(self.fallback.next_plan(state), state)

        params = dict(plan.params)

        if plan.tool_name in {"dataset_gen_afba", "dataset_gen_pred"}:
            params.setdefault("mode", self._next_dataset_mode(state))
            params.setdefault("ts_length", 4)
            params.setdefault("interval", 1)
            if self._should_limit_dataset_generation(state, params["mode"]):
                params.setdefault("sample_limit", 3)

        if plan.tool_name == "run_spatial_model":
            params.setdefault("ts_length", 4)
            params.setdefault("interval", 1)
            params.setdefault("batch_size", 1)
            params.setdefault("epochs", 5)

        if plan.tool_name in {"run_spatial_temp_model", "run_spatial_temp_model_pred", "run_seq_model"}:
            params.setdefault("ts_length", 4)
            params.setdefault("interval", 1)
            params.setdefault("batch_size", 1)
            params.setdefault("epochs", 5)
        if plan.tool_name == "run_spatial_temp_model_pred":
            params.setdefault("channels", 43)

        return AnalysisPlan(
            tool_name=plan.tool_name,
            rationale=plan.rationale,
            params=params,
        )

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

    def _should_limit_dataset_generation(self, state: AnalysisState, mode: object) -> bool:
        snapshot = state.data_snapshot
        if snapshot is None or not isinstance(mode, str):
            return not state.history
        if mode == "train":
            return not snapshot.has_prepared_train
        if mode == "val":
            return not snapshot.has_prepared_val
        if mode == "test":
            return not snapshot.has_prepared_test
        return not state.history

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        request = urllib.request.Request(
            url=f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP error: {detail}") from exc

        return body["choices"][0]["message"]["content"]
