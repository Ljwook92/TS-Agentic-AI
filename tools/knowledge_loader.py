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


def load_planner_brief() -> dict[str, object]:
    return {
        "tool_rules": [
            "Use dataset_gen_afba when task is af or ba and prepared train/val arrays are missing. Prefer mode=train, ts_length=4, interval=1 for the next action.",
            "Use dataset_gen_pred when task is pred and prepared prediction arrays are missing. Prefer mode=train, ts_length=4, interval=1 for the next action.",
            "Use run_spatial_model as the cheapest AF/BA baseline after datasets exist. Prefer ts_length=4, interval=1, batch_size=1.",
            "Use run_spatial_temp_model when AF/BA baseline metrics are weak or temporal context is needed. Prefer ts_length=4, interval=1, batch_size=1.",
            "Use run_spatial_temp_model_pred for prediction once pred datasets exist. Prefer ts_length=4, interval=1, batch_size=1.",
            "Use inspect_only when raw data root is missing or prediction task lacks FirePred folders.",
        ],
        "ts_policy": {
            "run_spatial_model": "spatial_framewise: flatten [B,C,T,H,W] into [B*T,C,H,W].",
            "run_spatial_temp_model": "spatiotemporal_native: keep [B,C,T,H,W].",
            "run_spatial_temp_model_pred": "spatiotemporal_native: keep [B,C,T,H,W].",
            "run_seq_model": "temporal_sequence: use sequence representation.",
        },
        "failure_rules": [
            "If evaluation says retry_with_smaller_batch, rerun same tool with batch_size 1.",
            "If evaluation says retry_with_shorter_sequence, regenerate dataset with smaller ts_length.",
            "If evaluation says retry_with_spatiotemporal, escalate from spatial baseline to run_spatial_temp_model.",
            "If evaluation says needs_data_filtering, keep current task but prefer dataset generation or inspection over model execution.",
        ],
        "data_facts": [
            "Prediction input uses 27 channels: VIIRS_Day 6 + VIIRS_Night 2 + FirePred 19.",
            "Prediction requires FirePred folders and usable GeoTIFF files.",
            "Prepared dataset files live under SATFIRE_ROOT/dataset/dataset_train and dataset_val.",
        ],
        "output_contract": {
            "required_keys": ["tool_name", "rationale", "params"],
            "allowed_tools": [
                "inspect_only",
                "dataset_gen_afba",
                "dataset_gen_pred",
                "run_spatial_model",
                "run_seq_model",
                "run_spatial_temp_model",
                "run_spatial_temp_model_pred",
            ],
        },
    }
