from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


def load_data_functions():
    try:
        from analyze_pred_event_windows import (
            has_prediction_inputs,
            load_daily_masks,
            resolve_locations,
        )
    except ModuleNotFoundError:
        from scripts.analyze_pred_event_windows import (
            has_prediction_inputs,
            load_daily_masks,
            resolve_locations,
        )
    return has_prediction_inputs, load_daily_masks, resolve_locations


def parse_radii(value: str) -> list[float]:
    radii = sorted({float(item.strip()) for item in value.split(',') if item.strip()})
    if not radii or radii[0] <= 0:
        raise ValueError('--candidate-radii must contain positive values')
    return radii


def radius_tag(value: float) -> str:
    return str(float(value)).replace('.', 'p')


def structure_for_connectivity(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError('connectivity must be 4 or 8')


def split_local_remote_growth(
    current: np.ndarray,
    growth: np.ndarray,
    local_spread_radius: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_components = ndimage.label(growth, structure=structure)
    if n_components == 0:
        empty = np.zeros_like(growth, dtype=bool)
        return empty, empty.copy(), 0, 0

    distance_to_current = ndimage.distance_transform_edt(~current)
    component_ids = np.arange(1, n_components + 1)
    minimum_distances = np.asarray(
        ndimage.minimum(distance_to_current, labels=labels, index=component_ids),
        dtype=np.float64,
    )
    local_ids = component_ids[minimum_distances <= local_spread_radius]
    remote_ids = component_ids[minimum_distances > local_spread_radius]
    return (
        np.isin(labels, local_ids),
        np.isin(labels, remote_ids),
        int(local_ids.size),
        int(remote_ids.size),
    )


def candidate_geometry(
    current: np.ndarray,
    connectivity: int,
    min_component_pixels: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    labels, n_components = ndimage.label(
        current,
        structure=structure_for_connectivity(connectivity),
    )
    if n_components == 0:
        return None

    component_ids = np.arange(1, n_components + 1)
    areas = np.asarray(
        ndimage.sum(current, labels=labels, index=component_ids),
        dtype=np.int64,
    )
    valid_ids = component_ids[areas >= min_component_pixels]
    if not valid_ids.size:
        return None

    background = ~current
    distances, nearest_indices = ndimage.distance_transform_edt(
        background,
        return_indices=True,
    )
    nearest_components = labels[nearest_indices[0], nearest_indices[1]]
    valid_background = background & np.isin(nearest_components, valid_ids)
    return distances, valid_background


def transition_spread_stats(
    previous: np.ndarray,
    current: np.ndarray,
    local_spread_radius: float,
) -> tuple[int, float, float]:
    if not previous.any():
        return 0, np.nan, np.nan
    growth = current & ~previous
    local_growth, _, _, _ = split_local_remote_growth(
        previous,
        growth,
        local_spread_radius,
    )
    count = int(local_growth.sum())
    if not count:
        return count, np.nan, np.nan
    distances = ndimage.distance_transform_edt(~previous)[local_growth]
    return count, float(np.percentile(distances, 95)), float(distances.max())


def history_spread_signals(
    masks: list[np.ndarray],
    day_idx: int,
    history_days: int,
    local_spread_radius: float,
) -> dict[str, float]:
    first_end_idx = max(1, day_idx - history_days + 2)
    stats = [
        transition_spread_stats(
            masks[end_idx - 1].astype(bool),
            masks[end_idx].astype(bool),
            local_spread_radius,
        )
        for end_idx in range(first_end_idx, day_idx + 1)
    ]
    valid_p95 = [item[1] for item in stats if np.isfinite(item[1])]
    valid_max = [item[2] for item in stats if np.isfinite(item[2])]
    latest = stats[-1] if stats else (0, np.nan, np.nan)
    return {
        'previous_local_growth_pixels': int(latest[0]),
        'previous_spread_distance_p95': latest[1],
        'previous_spread_distance_max': latest[2],
        'history_spread_distance_p95_max': max(valid_p95) if valid_p95 else np.nan,
        'history_spread_distance_max': max(valid_max) if valid_max else np.nan,
        'available_spread_transitions': len(stats),
    }


def select_radius(signal: float, radii: list[float]) -> float:
    if not np.isfinite(signal):
        return radii[0]
    for radius in radii:
        if signal <= radius:
            return radius
    return radii[-1]


def coverage_row(
    base: dict,
    policy: str,
    selected_radius: float,
    distances: np.ndarray,
    valid_background: np.ndarray,
    full_growth: np.ndarray,
    local_growth: np.ndarray,
    remote_growth: np.ndarray,
) -> dict:
    support = valid_background & (distances <= selected_radius)
    local_pixels = int(local_growth.sum())
    full_pixels = int(full_growth.sum())
    candidate_pixels = int(support.sum())
    local_supported = int((support & local_growth).sum())
    full_supported = int((support & full_growth).sum())
    return {
        **base,
        'policy': policy,
        'selected_radius': selected_radius,
        'candidate_pixels': candidate_pixels,
        'local_true_pixels': local_pixels,
        'local_supported_pixels': local_supported,
        'local_oracle_iou': local_supported / local_pixels if local_pixels else np.nan,
        'local_candidate_prevalence': (
            local_supported / candidate_pixels if candidate_pixels else np.nan
        ),
        'full_true_pixels': full_pixels,
        'full_supported_pixels': full_supported,
        'full_oracle_iou': full_supported / full_pixels if full_pixels else np.nan,
        'remote_true_pixels': int(remote_growth.sum()),
    }


def summarize(rows: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    output = []
    minimum_fixed_candidates = rows.loc[
        rows['policy'] == f'fixed_r{radius_tag(radii[0])}',
        'candidate_pixels',
    ].sum()
    for policy, group in rows.groupby('policy', sort=False):
        positive = group[group['local_true_pixels'] > 0]
        local_true = int(group['local_true_pixels'].sum())
        local_supported = int(group['local_supported_pixels'].sum())
        full_true = int(group['full_true_pixels'].sum())
        full_supported = int(group['full_supported_pixels'].sum())
        remote_true = int(group['remote_true_pixels'].sum())
        candidate_pixels = int(group['candidate_pixels'].sum())
        item = {
            'policy': policy,
            'n_dates': int(len(group)),
            'n_local_positive_dates': int(len(positive)),
            'mean_selected_radius': float(group['selected_radius'].mean()),
            'local_oracle_iou_mean': float(positive['local_oracle_iou'].mean()),
            'local_oracle_iou_micro': (
                local_supported / local_true if local_true else np.nan
            ),
            'full_oracle_iou_micro': full_supported / full_true if full_true else np.nan,
            'candidate_pixels_mean': float(group['candidate_pixels'].mean()),
            'candidate_pixels_total': candidate_pixels,
            'candidate_cost_vs_min_radius': (
                candidate_pixels / minimum_fixed_candidates
                if minimum_fixed_candidates else np.nan
            ),
            'local_candidate_prevalence_micro': (
                local_supported / candidate_pixels if candidate_pixels else np.nan
            ),
            'remote_growth_fraction_micro': (
                remote_true / full_true if full_true else 0.0
            ),
        }
        for radius in radii:
            item[f'selected_r{radius_tag(radius)}_fraction'] = float(
                (group['selected_radius'] == radius).mean()
            )
        output.append(item)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Measure candidate-radius oracle coverage and causal adaptive-radius '
            'signals from prior VIIRS spread.'
        )
    )
    parser.add_argument('-mode', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--candidate-radii', default='5,8,10,12')
    parser.add_argument('--connectivity', type=int, choices=[4, 8], default=8)
    parser.add_argument('--min-component-pixels', type=int, default=1)
    parser.add_argument('--history-days', type=int, default=4)
    parser.add_argument('--allow-partial-history', action='store_true')
    parser.add_argument('--local-spread-radius', type=float, default=5.0)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('/home/jlc3q/data/SatFire/event_candidates'),
    )
    parser.add_argument('-limit', type=int, default=None)
    parser.add_argument('-start', type=int, default=0)
    args = parser.parse_args()

    radii = parse_radii(args.candidate_radii)
    if args.history_days < 1:
        raise ValueError('--history-days must be >= 1')
    if args.local_spread_radius < 0:
        raise ValueError('--local-spread-radius must be >= 0')

    has_prediction_inputs, load_daily_masks, resolve_locations = load_data_functions()
    pairs = [
        (fire_id, label_sel)
        for fire_id, label_sel in resolve_locations(args.mode)
        if has_prediction_inputs(fire_id)
    ]
    pairs = pairs[max(args.start, 0):]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError(f'No valid fires found for mode={args.mode}')

    rows = []
    start_idx = 0 if args.allow_partial_history else max(args.history_days - 1, 0)
    for fire_number, (fire_id, label_sel) in enumerate(pairs, start=1):
        print(f'[{fire_number}/{len(pairs)}] {fire_id}')
        dates, masks = load_daily_masks(fire_id, label_sel)
        for day_idx in range(start_idx, len(masks) - 1):
            current = masks[day_idx].astype(bool)
            if not current.any():
                continue
            geometry = candidate_geometry(
                current,
                args.connectivity,
                args.min_component_pixels,
            )
            if geometry is None:
                continue
            distances, valid_background = geometry
            future = masks[day_idx + 1].astype(bool)
            full_growth = future & ~current
            local_growth, remote_growth, n_local, n_remote = split_local_remote_growth(
                current,
                full_growth,
                args.local_spread_radius,
            )
            signals = history_spread_signals(
                masks,
                day_idx,
                args.history_days,
                args.local_spread_radius,
            )
            base = {
                'fire_id': fire_id,
                'date': dates[day_idx],
                'next_date': dates[day_idx + 1],
                'day_idx': day_idx,
                'history_days': args.history_days,
                'available_history_days': min(args.history_days, day_idx + 1),
                'local_component_count': n_local,
                'remote_component_count': n_remote,
                **signals,
            }
            for radius in radii:
                rows.append(coverage_row(
                    base,
                    f'fixed_r{radius_tag(radius)}',
                    radius,
                    distances,
                    valid_background,
                    full_growth,
                    local_growth,
                    remote_growth,
                ))
            adaptive_signals = {
                'adaptive_previous_p95': signals['previous_spread_distance_p95'],
                'adaptive_history_p95_max': signals['history_spread_distance_p95_max'],
            }
            for policy, signal in adaptive_signals.items():
                rows.append(coverage_row(
                    base,
                    policy,
                    select_radius(signal, radii),
                    distances,
                    valid_background,
                    full_growth,
                    local_growth,
                    remote_growth,
                ))

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError('No prediction dates were analyzed')
    summary = summarize(detail, radii)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_tag = '_partial' if args.allow_partial_history and args.history_days > 1 else ''
    local_tag = radius_tag(args.local_spread_radius)
    suffix = (
        f'{args.mode}_conn{args.connectivity}_h{args.history_days}{partial_tag}'
        f'_localr{local_tag}'
    )
    detail_path = args.out_dir / f'pred_event_candidate_radius_oracle_detail_{suffix}.csv'
    summary_path = args.out_dir / f'pred_event_candidate_radius_oracle_summary_{suffix}.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f'Wrote {detail_path}')
    print(f'Wrote {summary_path}')
    print('\nSummary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
