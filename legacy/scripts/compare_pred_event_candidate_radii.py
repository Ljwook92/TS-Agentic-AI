from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path('/home/jlc3q/data/SatFire/event_candidates')


def parse_radii(value: str) -> list[float]:
    radii = sorted({float(item.strip()) for item in value.split(',') if item.strip()})
    if not radii or radii[0] <= 0:
        raise ValueError('--candidate-radii must contain positive values')
    return radii


def number_tag(value: float) -> str:
    return str(float(value)).replace('.', 'p')


def evaluation_output_tag(args: argparse.Namespace, candidate_radius: float) -> str:
    if args.goes_variant == 'goes_frp':
        output_tag = ''
    else:
        partial_tag = '_partial' if args.allow_partial_history and args.history_days > 1 else ''
        history_tag = f'_h{args.history_days}{partial_tag}' if args.history_days > 1 else ''
        output_tag = f'{history_tag}_{args.goes_variant}'
    if args.fixed_threshold is not None:
        output_tag += f'_thr{number_tag(args.fixed_threshold)}'
    if args.full_growth_metrics:
        output_tag += f'_localr{number_tag(args.local_spread_radius)}_fullgrowth'
    if args.target_scope == 'local':
        output_tag += '_targetlocal'
    if args.source_candidate_radius != 5.0 or candidate_radius != 5.0:
        output_tag += (
            f'_srcR{number_tag(args.source_candidate_radius)}'
            f'_candR{number_tag(candidate_radius)}'
        )
    return output_tag


def run_evaluation(
    script: Path,
    args: argparse.Namespace,
    candidate_radius: float,
) -> None:
    command = [
        sys.executable,
        str(script),
        '--candidate-root',
        str(args.candidate_root),
        '--candidate-radius',
        str(candidate_radius),
        '--source-candidate-radius',
        str(args.source_candidate_radius),
        '--connectivity',
        str(args.connectivity),
        '--min-component-pixels',
        str(args.min_component_pixels),
        '--target-scope',
        args.target_scope,
        '--history-days',
        str(args.history_days),
        '--goes-variant',
        args.goes_variant,
        '--feature-sets',
        args.feature_set,
        '--train-sample',
        str(args.train_sample),
        '--read-chunksize',
        str(args.read_chunksize),
        '--selection-objective',
        args.selection_objective,
        '--local-spread-radius',
        str(args.local_spread_radius),
    ]
    if args.allow_partial_history:
        command.append('--allow-partial-history')
    if args.fixed_threshold is not None:
        command.extend(['--fixed-threshold', str(args.fixed_threshold)])
    if args.full_growth_metrics:
        command.append('--full-growth-metrics')
    print('\nRunning:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def add_deltas(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    frame = frame.sort_values(['split', 'candidate_radius']).copy()
    for metric in metrics:
        if metric not in frame:
            continue
        baseline = frame.groupby('split')[metric].transform('first')
        frame[f'{metric}_delta_vs_min_radius'] = frame[metric] - baseline
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Run or collect fixed candidate-radius model evaluations from one '
            'maximum-radius enriched candidate CSV.'
        )
    )
    parser.add_argument('--candidate-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--candidate-radii', default='5,8,10,12')
    parser.add_argument('--source-candidate-radius', type=float, default=None)
    parser.add_argument('--connectivity', type=int, default=8)
    parser.add_argument('--min-component-pixels', type=int, default=1)
    parser.add_argument('--target-scope', choices=['all', 'local'], default='local')
    parser.add_argument('--history-days', type=int, default=4)
    parser.add_argument('--allow-partial-history', action='store_true')
    parser.add_argument('--goes-variant', default='goes_frp_motion_recent_firepred')
    parser.add_argument(
        '--feature-set',
        default='geometry_plus_goes_frp_motion_recent_firepred',
    )
    parser.add_argument('--train-sample', type=int, default=1_000_000)
    parser.add_argument('--read-chunksize', type=int, default=200_000)
    parser.add_argument('--fixed-threshold', type=float, default=0.8)
    parser.add_argument('--selection-objective', default='fire_local_positive_mean_iou')
    parser.add_argument('--local-spread-radius', type=float, default=5.0)
    parser.add_argument(
        '--full-growth-metrics',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--run-evaluation',
        action='store_true',
        help='Run eval_pred_event_candidate_masks.py for every radius before collecting results.',
    )
    parser.add_argument(
        '--skip-existing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Do not rerun a radius when its expected summary CSV already exists.',
    )
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    radii = parse_radii(args.candidate_radii)
    if args.source_candidate_radius is None:
        args.source_candidate_radius = max(radii)
    if args.source_candidate_radius < max(radii):
        raise ValueError('--source-candidate-radius must cover every requested radius')

    script = Path(__file__).with_name('eval_pred_event_candidate_masks.py')
    if args.run_evaluation:
        for radius in radii:
            summary_path = args.candidate_root / (
                f'pred_event_mask_eval_summary{evaluation_output_tag(args, radius)}.csv'
            )
            if args.skip_existing and summary_path.exists():
                print(f'\nSkipping radius={radius}: {summary_path} already exists', flush=True)
                continue
            run_evaluation(script, args, radius)

    frames = []
    missing = []
    for radius in radii:
        output_tag = evaluation_output_tag(args, radius)
        path = args.candidate_root / f'pred_event_mask_eval_summary{output_tag}.csv'
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        frame = frame[frame['model'] == args.feature_set].copy()
        if frame.empty:
            raise ValueError(f'{path} does not contain model={args.feature_set}')
        frame['candidate_radius'] = radius
        frame['source_candidate_radius'] = args.source_candidate_radius
        frame['summary_path'] = str(path)
        frames.append(frame)

    if missing:
        formatted = '\n'.join(str(path) for path in missing)
        raise FileNotFoundError(
            'Missing radius evaluation summaries. Run with --run-evaluation first:\n'
            f'{formatted}'
        )

    comparison = pd.concat(frames, ignore_index=True)
    metrics = [
        'pr_auc',
        'roc_auc',
        'fire_local_positive_mean_iou',
        'fire_local_positive_mean_f1',
        'fire_local_micro_iou',
        'fire_local_micro_f1',
        'fire_local_candidate_coverage_mean',
        'fire_local_candidate_coverage_micro',
        'fire_full_mean_iou',
        'fire_full_mean_f1',
    ]
    comparison = add_deltas(comparison, metrics)

    if args.out is None:
        partial_tag = '_partial' if args.allow_partial_history and args.history_days > 1 else ''
        threshold_tag = (
            f'_thr{number_tag(args.fixed_threshold)}'
            if args.fixed_threshold is not None else '_tuned'
        )
        args.out = args.candidate_root / (
            f'pred_event_candidate_radius_actual_comparison_h{args.history_days}'
            f'{partial_tag}_{args.goes_variant}_{args.target_scope}{threshold_tag}.csv'
        )
    comparison.to_csv(args.out, index=False)

    display = [
        'split',
        'candidate_radius',
        'pr_auc',
        'fire_local_positive_mean_iou',
        'fire_local_positive_mean_f1',
        'fire_local_micro_iou',
        'fire_local_micro_f1',
        'fire_local_candidate_coverage_micro',
    ]
    display = [column for column in display if column in comparison]
    print('\nRadius comparison:')
    print(comparison[display].sort_values(['split', 'candidate_radius']).to_string(index=False))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
