from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from schemas.state import ExecutionResult


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "legacy"
CONFIG_PATH = ROOT / "configs" / "tool_defaults.json"
RUNS_DIR = ROOT / "memory" / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolSpec:
    script: str
    task_param: str | None
    params: dict[str, object]


class LegacyRunner:
    def __init__(self) -> None:
        config = json.loads(CONFIG_PATH.read_text())
        self.tool_specs = {
            name: ToolSpec(
                script=spec["script"],
                task_param=spec.get("task_param"),
                params=spec["params"],
            )
            for name, spec in config["tools"].items()
        }

    def run(self, tool_name: str, task: str, overrides: dict[str, object]) -> ExecutionResult:
        if tool_name not in self.tool_specs:
            raise ValueError(f"Unknown tool: {tool_name}")

        started_at = utc_now()
        tool = self.tool_specs[tool_name]
        params = dict(tool.params)
        if tool.task_param:
            params[tool.task_param] = task
        params.update(overrides)

        command = ["python", str(LEGACY_ROOT / "scripts" / tool.script)]
        command.extend(self._to_cli_args(params))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(LEGACY_ROOT)

        proc = subprocess.run(
            command,
            cwd=LEGACY_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        finished_at = utc_now()
        artifact_path = self._persist_run(tool_name=tool_name, params=params, proc=proc)
        return ExecutionResult(
            tool_name=tool_name,
            status="success" if proc.returncode == 0 else "failed",
            return_code=proc.returncode,
            stdout=proc.stdout[-12000:],
            stderr=proc.stderr[-12000:],
            command=command,
            artifact_path=str(artifact_path),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _persist_run(
        self,
        tool_name: str,
        params: dict[str, object],
        proc: subprocess.CompletedProcess[str],
    ) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUNS_DIR / f"{stamp}_{tool_name}.json"
        payload = {
            "tool_name": tool_name,
            "params": params,
            "return_code": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _to_cli_args(self, params: dict[str, object]) -> list[str]:
        cli_args: list[str] = []
        for key, value in params.items():
            if value is None:
                continue
            flag = f"-{self._map_arg_name(key)}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
                continue
            cli_args.extend([flag, str(value)])
        return cli_args

    def _map_arg_name(self, key: str) -> str:
        aliases = {
            "model": "m",
            "batch_size": "b",
            "run": "r",
            "learning_rate": "lr",
            "num_heads": "nh",
            "embedding_dim": "ed",
            "channels": "nc",
            "ts_length": "ts",
            "interval": "it",
            "epoch": "epoch",
            "test": "test",
            "mlp_dim": "md",
            "num_layers": "nl",
            "use_case": "uc",
        }
        return aliases.get(key, key)
