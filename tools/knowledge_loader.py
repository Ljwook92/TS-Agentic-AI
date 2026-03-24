from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"


def load_planner_knowledge() -> dict[str, str]:
    files = [
        "data_sources_ts_satfire.md",
        "decision_policy.md",
        "tool_catalog.md",
        "ts_adapter_policy.md",
    ]
    payload: dict[str, str] = {}
    for name in files:
        path = KNOWLEDGE_DIR / name
        if path.exists():
            payload[name] = path.read_text()[:12000]
    return payload
