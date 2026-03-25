from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from schemas.state import ExecutionResult
from legacy.support.path_config import get_dataset_root


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

        if tool_name.startswith("dataset_gen_"):
            self._cleanup_incomplete_prepared_dataset_files(tool_name=tool_name, task=task, params=params)

        command = ["python", str(LEGACY_ROOT / "scripts" / tool.script)]
        command.extend(self._to_cli_args(params))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(LEGACY_ROOT)

        proc = subprocess.Popen(
            command,
            cwd=LEGACY_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_text, stderr_text = self._stream_process_output(proc)

        finished_at = utc_now()
        artifact_path = self._persist_run(
            tool_name=tool_name,
            params=params,
            return_code=proc.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return ExecutionResult(
            tool_name=tool_name,
            status="success" if proc.returncode == 0 else "failed",
            return_code=proc.returncode,
            stdout=stdout_text[-12000:],
            stderr=stderr_text[-12000:],
            command=command,
            artifact_path=str(artifact_path),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _cleanup_incomplete_prepared_dataset_files(self, tool_name: str, task: str, params: dict[str, object]) -> None:
        dataset_root = Path(get_dataset_root())
        ts_length = params.get("ts_length", 6)
        interval = params.get("interval", 1)
        mode = params.get("mode", "train")
        prefix = task if tool_name == "dataset_gen_pred" else str(params.get("use_case", task))

        if not isinstance(ts_length, int) or not isinstance(interval, int) or not isinstance(mode, str):
            return

        if mode in {"train", "val"}:
            target_dir = dataset_root / f"dataset_{mode}"
            image_path = target_dir / f"{prefix}_{mode}_img_seqtoseq_alll_{ts_length}i_{interval}.npy"
            label_path = target_dir / f"{prefix}_{mode}_label_seqtoseq_alll_{ts_length}i_{interval}.npy"
            image_exists = image_path.exists()
            label_exists = label_path.exists()
            if image_exists and label_exists:
                return
            if image_exists and not label_exists:
                image_path.unlink()
            if label_exists and not image_exists:
                label_path.unlink()
            return

        if mode == "test":
            target_dir = dataset_root / "dataset_test"
            matched_pairs: dict[str, set[str]] = {}
            for path in target_dir.glob(f"{prefix}_*_seqtoseql_{ts_length}i_{interval}.npy"):
                stem = path.name.replace(f"_seqtoseql_{ts_length}i_{interval}.npy", "")
                if stem.endswith("_img"):
                    matched_pairs.setdefault(stem[:-4], set()).add("img")
                elif stem.endswith("_label"):
                    matched_pairs.setdefault(stem[:-6], set()).add("label")

            if any({"img", "label"}.issubset(kinds) for kinds in matched_pairs.values()):
                return

            for sample_id, kinds in matched_pairs.items():
                if "img" in kinds and "label" not in kinds:
                    stale = target_dir / f"{sample_id}_img_seqtoseql_{ts_length}i_{interval}.npy"
                    if stale.exists():
                        stale.unlink()
                if "label" in kinds and "img" not in kinds:
                    stale = target_dir / f"{sample_id}_label_seqtoseql_{ts_length}i_{interval}.npy"
                    if stale.exists():
                        stale.unlink()

            raw_imgs = list(target_dir.glob(f"{prefix}_*_img.npy"))
            raw_labels = list(target_dir.glob(f"{prefix}_*_label.npy"))
            if raw_imgs and not raw_labels:
                for path in raw_imgs:
                    path.unlink()
            if raw_labels and not raw_imgs:
                for path in raw_labels:
                    path.unlink()

    def _persist_run(
        self,
        tool_name: str,
        params: dict[str, object],
        return_code: int,
        stdout_text: str,
        stderr_text: str,
    ) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUNS_DIR / f"{stamp}_{tool_name}.json"
        payload = {
            "tool_name": tool_name,
            "params": params,
            "return_code": return_code,
            "stdout": stdout_text[-12000:],
            "stderr": stderr_text[-12000:],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _stream_process_output(self, proc: subprocess.Popen[str]) -> tuple[str, str]:
        selector = selectors.DefaultSelector()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        if proc.stdout is not None:
            selector.register(proc.stdout, selectors.EVENT_READ, data=("stdout", stdout_chunks, sys.stdout))
        if proc.stderr is not None:
            selector.register(proc.stderr, selectors.EVENT_READ, data=("stderr", stderr_chunks, sys.stderr))

        while selector.get_map():
            for key, _ in selector.select():
                stream_name, sink, output_stream = key.data
                line = key.fileobj.readline()
                if line == "":
                    selector.unregister(key.fileobj)
                    continue
                sink.append(line)
                output_stream.write(line)
                output_stream.flush()

        proc.wait()
        return "".join(stdout_chunks), "".join(stderr_chunks)

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
            "attn_version": "av",
            "num_heads": "nh",
            "embedding_dim": "ed",
            "channels": "nc",
            "ts_length": "ts",
            "interval": "it",
            "epochs": "epochs",
            "epoch": "epoch",
            "test": "test",
            "mlp_dim": "md",
            "num_layers": "nl",
            "use_case": "uc",
            "sample_limit": "limit",
        }
        return aliases.get(key, key)
