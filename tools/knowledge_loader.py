from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"


def load_planner_knowledge() -> dict[str, str]:
    files = [
        "data_sources_ts_satfire.md",
        "decision_policy.md",
        "paper_priorities.json",
        "heuristics.md",
        "reasoning_templates.md",
        "tool_catalog.md",
        "ts_adapter_policy.md",
        "feature_sets.json",
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
            "Use dataset_gen_afba when task is af or ba and prepared arrays are missing. Prefer mode=train, then mode=val, then mode=test as missing splits are discovered. Prefer ts_length=4, interval=1 for the next action.",
            "Use dataset_gen_pred when task is pred and prepared prediction arrays are missing. Prefer mode=train, then mode=val, then mode=test as missing splits are discovered. Prefer ts_length=4, interval=1 for the next action.",
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
            "If evaluation says retry_with_longer_sequence, regenerate datasets with a longer ts_length before retrying model execution.",
            "If evaluation says needs_dataset_generation, generate the missing split before retrying model execution.",
            "If evaluation says retry_with_spatiotemporal, escalate from spatial baseline to run_spatial_temp_model.",
            "If evaluation says needs_experiment_upgrade, choose a stronger model family or a different temporal setting instead of repeating the same run.",
            "If evaluation says needs_data_filtering, keep current task but prefer dataset generation or inspection over model execution.",
        ],
        "experiment_upgrade_policy": [
            "When a temporal model plateaus, prefer changing one major factor at a time: attention version, then hidden size, then learning rate, then sequence length.",
            "Use reduced feature-set ablations only after a stable model path exists for the task.",
            "Treat feature-set changes as scientific ablations and compare them directly against the current best configuration.",
        ],
        "data_facts": [
            "Prediction input uses 27 channels: VIIRS_Day 6 + VIIRS_Night 2 + FirePred 19.",
            "Prediction requires FirePred folders and usable GeoTIFF files.",
            "Prepared dataset files live under SATFIRE_ROOT/dataset/dataset_train, dataset_val, and dataset_test.",
        ],
        "feature_set_policy": [
            "Current implemented defaults are af_core8 for AF/BA and pred_full43 for prediction.",
            "Planned ablations include af_day_only, af_night_heavy, pred_no_landcover, pred_no_firepred, and pred_remote_only.",
            "Do not recommend a planned feature set unless the current implemented baseline has already been run successfully.",
        ],
        "paper_priority_policy": [
            "Treat TS-SatFire as the highest-priority source for task semantics, split conventions, and variable assumptions.",
            "When proposing AF/BA/pred experiments, prefer high-priority papers over generic supporting literature.",
            "Use supporting papers to widen the search space only after TS-SatFire-specific evidence has been respected.",
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
