from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from schemas.state import DataSnapshot

from legacy.support.path_config import (
    get_checkpoints_root,
    get_dataset_root,
    get_raw_data_root,
    get_satfire_root,
)


@dataclass
class DataInspector:
    def inspect(self, task: str, ts_length: int = 6, interval: int = 1) -> DataSnapshot:
        satfire_root = get_satfire_root()
        raw_root = get_raw_data_root()
        dataset_root = get_dataset_root()
        checkpoints_root = get_checkpoints_root()

        prepared_files = self._prepared_files(task=task, dataset_root=dataset_root, ts_length=ts_length, interval=interval)
        has_prepared_train = Path(prepared_files["train_image"]).exists() and Path(prepared_files["train_label"]).exists()
        has_prepared_val = Path(prepared_files["val_image"]).exists() and Path(prepared_files["val_label"]).exists()

        firepred_dirs = list(raw_root.glob("*/FirePred")) if raw_root.exists() else []
        raw_fire_dirs = [path for path in raw_root.iterdir() if path.is_dir()] if raw_root.exists() else []

        return DataSnapshot(
            satfire_root=str(satfire_root),
            raw_data_root=str(raw_root),
            dataset_root=str(dataset_root),
            checkpoints_root=str(checkpoints_root),
            has_raw_data_root=raw_root.exists(),
            has_prepared_train=has_prepared_train,
            has_prepared_val=has_prepared_val,
            has_firepred=bool(firepred_dirs),
            firepred_count=len(firepred_dirs),
            raw_fire_count=len(raw_fire_dirs),
            prepared_files=prepared_files,
        )

    def _prepared_files(self, task: str, dataset_root: Path, ts_length: int, interval: int) -> dict[str, str]:
        prefix = task
        discovered = self._discover_prepared_files(prefix=prefix, dataset_root=dataset_root)
        if discovered:
            return discovered
        return {
            "train_image": str(dataset_root / "dataset_train" / f"{prefix}_train_img_seqtoseq_alll_{ts_length}i_{interval}.npy"),
            "train_label": str(dataset_root / "dataset_train" / f"{prefix}_train_label_seqtoseq_alll_{ts_length}i_{interval}.npy"),
            "val_image": str(dataset_root / "dataset_val" / f"{prefix}_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy"),
            "val_label": str(dataset_root / "dataset_val" / f"{prefix}_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy"),
        }

    def _discover_prepared_files(self, prefix: str, dataset_root: Path) -> dict[str, str] | None:
        train_dir = dataset_root / "dataset_train"
        val_dir = dataset_root / "dataset_val"
        pattern = re.compile(rf"^{re.escape(prefix)}_(train|val)_(img|label)_seqtoseq_alll_(\d+)i_(\d+)\.npy$")

        candidates: dict[tuple[str, str], Path] = {}
        scores: dict[tuple[str, str], tuple[int, int]] = {}
        for base_dir in (train_dir, val_dir):
            if not base_dir.exists():
                continue
            for path in base_dir.glob(f"{prefix}_*_seqtoseq_alll_*.npy"):
                match = pattern.match(path.name)
                if not match:
                    continue
                split, kind, ts_value, interval_value = match.groups()
                score = (int(ts_value), int(interval_value))
                key = (split, kind)
                existing = candidates.get(key)
                if existing is None or score < scores[key]:
                    candidates[key] = path
                    scores[key] = score

        required = {
            ("train", "img"),
            ("train", "label"),
            ("val", "img"),
            ("val", "label"),
        }
        if not all(key in candidates for key in required):
            return None

        return {
            "train_image": str(candidates[("train", "img")]),
            "train_label": str(candidates[("train", "label")]),
            "val_image": str(candidates[("val", "img")]),
            "val_label": str(candidates[("val", "label")]),
        }
