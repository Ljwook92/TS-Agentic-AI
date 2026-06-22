from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path('/home/jlc3q/data/SatFire/event_candidates')
RESULT_PATTERNS = [
    'pred_event_mask_eval_summary*.csv',
    'pred_event_eval_compare_*global.csv',
    'pred_event_eval_compare_*topk.csv',
    'pred_event_history_comparison*.csv',
    'pred_event_candidate_radius_oracle_summary*.csv',
]
PREFERRED_COLUMNS = [
    'model',
    'split',
    'policy',
    'history_a_days',
    'history_b_days',
    'candidate_radius',
    'source_candidate_radius',
    'target_scope',
    'selected_method',
    'selected_value',
    'pr_auc',
    'roc_auc',
    'mean_iou',
    'mean_f1',
    'fire_mean_iou',
    'fire_mean_f1',
    'fire_micro_iou',
    'fire_micro_f1',
    'fire_full_mean_iou',
    'fire_full_mean_f1',
    'fire_full_micro_iou',
    'fire_full_micro_f1',
    'fire_local_positive_mean_iou',
    'fire_local_positive_mean_f1',
    'fire_local_micro_iou',
    'fire_local_micro_f1',
    'fire_local_candidate_coverage_micro',
    'local_oracle_iou_mean',
    'local_oracle_iou_micro',
    'candidate_cost_vs_min_radius',
    'local_candidate_prevalence_micro',
    'remote_growth_fraction_micro',
    'delta_b_minus_a',
    'ci95_low',
    'ci95_high',
    'wilcoxon_p',
]


def format_value(value) -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.6f}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '_No rows._\n'
    columns = list(frame.columns)
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(format_value(value) for value in row) + ' |')
    return '\n'.join(lines) + '\n'


def select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    selected = [column for column in PREFERRED_COLUMNS if column in frame.columns]
    if not selected:
        selected = list(frame.columns[:20])
    return frame[selected]


def result_files(root: Path) -> list[Path]:
    paths = set()
    for pattern in RESULT_PATTERNS:
        paths.update(root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def candidate_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob('pred_event_candidates_*.csv')):
        name = path.name
        if '_r12p0_' in name and '_h4_partial_' in name and name.endswith(
            '_goes_frp_motion_recent_firepred.csv'
        ):
            retention = 'KEEP: selected maximum-radius enriched source'
        elif '_r12p0_' in name and '_h4_partial_' in name:
            retention = 'TEMP: remove after final enriched files and sweep are verified'
        elif '_r5p0_' in name and '_h4_partial_' in name and name.endswith(
            '_goes_frp_motion_recent_firepred.csv'
        ):
            retention = 'KEEP UNTIL SWEEP VERIFIED: previous best comparison source'
        else:
            retention = 'ARCHIVE RESULT, THEN DELETE: superseded candidate dataset'
        rows.append({
            'file': name,
            'size_gib': path.stat().st_size / (1024 ** 3),
            'retention': retention,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Archive compact candidate experiment summaries and data-retention decisions to Markdown.'
    )
    parser.add_argument('--candidate-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--max-rows-per-table', type=int, default=200)
    args = parser.parse_args()

    root = args.candidate_root
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.out is None:
        args.out = root / 'pred_event_experiment_archive.md'

    paths = result_files(root)
    if not paths:
        raise RuntimeError(f'No compact result CSV files found under {root}')

    lines = [
        '# Prediction Event Experiment Archive',
        '',
        f'- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`',
        f'- Candidate root: `{root}`',
        '- Scope: H1/H2/H4/H6 candidate experiments, GOES/FirePred ablations, local/full-growth metrics, and radius oracle results.',
        '',
        '## Retention Decision',
        '',
        '- Preserve raw VIIRS/FirePred and raw GOES sources.',
        '- Preserve the final H4-partial r12 GOES-recent-motion + FirePred CSV for train/val/test.',
        '- Preserve compact CSV summaries and this Markdown archive.',
        '- Preserve the previous H4-partial r5 final CSV only until the r12-source radius sweep is verified.',
        '- Intermediate and superseded candidate CSVs may be removed after their row counts and archived results are verified.',
        '',
        '## Candidate Dataset Inventory',
        '',
    ]
    inventory = candidate_inventory(root)
    if inventory.empty:
        lines.extend(['_No candidate CSV files found._', ''])
    else:
        lines.extend([markdown_table(inventory), ''])

    lines.extend(['## Result Tables', ''])
    for path in paths:
        frame = pd.read_csv(path)
        shown = select_columns(frame)
        truncated = len(shown) > args.max_rows_per_table
        if truncated:
            shown = shown.head(args.max_rows_per_table)
        lines.extend([
            f'### `{path.name}`',
            '',
            f'- Rows in source: `{len(frame)}`',
            markdown_table(shown),
        ])
        if truncated:
            lines.extend([
                f'_Displayed first {args.max_rows_per_table} rows; source CSV remains authoritative._',
                '',
            ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {args.out}')
    print(f'Archived result files: {len(paths)}')
    print(f'Candidate inventory rows: {len(inventory)}')


if __name__ == '__main__':
    main()
