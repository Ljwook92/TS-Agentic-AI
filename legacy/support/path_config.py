from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SATFIRE_ROOT = "/home/jlc3q/data/SatFire"
DEFAULT_CODE_ROOT = "/home/jlc3q/New_project/TS-SatFire"


def get_satfire_root() -> Path:
    return Path(os.environ.get("SATFIRE_ROOT", DEFAULT_SATFIRE_ROOT)).expanduser()


def get_code_root() -> Path:
    return Path(os.environ.get("TS_SATFIRE_CODE_ROOT", DEFAULT_CODE_ROOT)).expanduser()


def get_dataset_root() -> Path:
    return get_satfire_root() / "dataset"


def get_raw_data_root() -> Path:
    return get_satfire_root() / "ts-satfire"


def get_checkpoints_root() -> Path:
    return get_satfire_root() / "checkpoints"


def get_eval_root() -> Path:
    return get_satfire_root() / "evaluation_plot"
