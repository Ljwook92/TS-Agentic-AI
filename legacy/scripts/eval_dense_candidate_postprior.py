from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-5


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def safe_name(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '-_' else '_' for char in str(value))


def logit(probability: np.ndarray | float) -> np.ndarray:
    value = np.clip(probability, EPS, 1.0 - EPS)
    return np.log(value) - np.log1p(-value)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def iou_f1(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, int, int, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    union = tp + fp + fn
    f1_denom = 2 * tp + fp + fn
    return (
        tp / union if union else 1.0,
        2 * tp / f1_denom if f1_denom else 1.0,
        tp,
        fp,
        fn,
    )


def expected_windows(split: str, ts_length: int, interval: int) -> list[dict]:
    try:
        from analyze_pred_event_windows import has_prediction_inputs, load_daily_masks, resolve_locations
    except ModuleNotFoundError:
        from scripts.analyze_pred_event_windows import has_prediction_inputs, load_daily_masks, resolve_locations

    rows = []
    for fire_id, label_sel in resolve_locations(split):
        if not has_prediction_inputs(fire_id):
            continue
        dates, masks = load_daily_masks(fire_id, label_sel)
        for start in range(0, len(dates), interval):
            target_idx = start + ts_length
            if target_idx >= len(dates):
                break
            current_idx = target_idx - 1
            rows.append({
                'fire_id': fire_id,
                'date': dates[current_idx],
                'day_idx': current_idx,
                'target': masks[target_idx] & ~masks[current_idx],
            })
    return rows


def load_candidate_maps(root: Path, feature_set: str, split: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    directory = root / safe_name(feature_set) / split
    if not directory.is_dir():
        raise FileNotFoundError(f'Candidate score-map directory does not exist: {directory}')
    maps = {}
    for path in sorted(directory.glob('*.npz')):
        data = np.load(path)
        fire_id = path.stem
        for index, date in enumerate(data['dates'].astype(str)):
            maps[(fire_id, str(date))] = (
                data['probability'][index].astype(np.float32),
                data['support'][index].astype(bool),
            )
    return maps


def dense_probability_for_rows(root: Path, split: str, rows: list[dict]) -> list[np.ndarray]:
    if split == 'val':
        path = root / 'val.npy'
        dense = np.load(path, mmap_mode='r')
        if dense.shape[0] != len(rows):
            raise ValueError(
                f'Validation dense rows ({dense.shape[0]}) do not match reconstructed windows ({len(rows)}). '
                'Check dataset filtering and TS/interval.'
            )
        return [dense[index].astype(np.float32) for index in range(len(rows))]

    by_fire: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_fire.setdefault(row['fire_id'], []).append(index)
    output: list[np.ndarray | None] = [None] * len(rows)
    for fire_id, indices in by_fire.items():
        path = root / f'test_{safe_name(fire_id)}.npy'
        dense = np.load(path, mmap_mode='r')
        if dense.shape[0] != len(indices):
            raise ValueError(
                f'{fire_id}: dense rows ({dense.shape[0]}) do not match reconstructed windows ({len(indices)})'
            )
        for local_index, global_index in enumerate(indices):
            output[global_index] = dense[local_index].astype(np.float32)
    return output  # type: ignore[return-value]


def evaluate(
    rows: list[dict],
    dense_probabilities: list[np.ndarray],
    candidate_maps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    alpha: float,
    dense_threshold: float,
    candidate_neutral_probability: float,
) -> tuple[dict, pd.DataFrame]:
    date_rows = []
    neutral_logit = float(logit(candidate_neutral_probability))
    for row, dense_probability in zip(rows, dense_probabilities):
        combined_logit = logit(dense_probability)
        candidate = candidate_maps.get((row['fire_id'], row['date']))
        support_pixels = 0
        if candidate is not None and alpha != 0:
            candidate_probability, support = candidate
            support_pixels = int(support.sum())
            combined_logit = combined_logit.copy()
            combined_logit[support] += alpha * (logit(candidate_probability[support]) - neutral_logit)
        prediction = sigmoid(combined_logit) >= dense_threshold
        iou, f1, tp, fp, fn = iou_f1(prediction, row['target'])
        date_rows.append({
            'fire_id': row['fire_id'],
            'date': row['date'],
            'day_idx': row['day_idx'],
            'iou': iou,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'candidate_support_pixels': support_pixels,
        })

    dates = pd.DataFrame(date_rows)
    fires = dates.groupby('fire_id', sort=False).agg(
        iou=('iou', 'mean'),
        f1=('f1', 'mean'),
        tp=('tp', 'sum'),
        fp=('fp', 'sum'),
        fn=('fn', 'sum'),
        n_dates=('date', 'size'),
    )
    tp = int(dates['tp'].sum())
    fp = int(dates['fp'].sum())
    fn = int(dates['fn'].sum())
    summary = {
        'alpha': alpha,
        'dense_threshold': dense_threshold,
        'candidate_neutral_probability': candidate_neutral_probability,
        'fire_macro_iou': float(fires['iou'].mean()),
        'fire_macro_f1': float(fires['f1'].mean()),
        'date_macro_iou': float(dates['iou'].mean()),
        'date_macro_f1': float(dates['f1'].mean()),
        'micro_iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        'n_fires': int(len(fires)),
        'n_dates': int(len(dates)),
    }
    return summary, dates


def main() -> None:
    parser = argparse.ArgumentParser(description='Fuse dense TS-SatFire predictions with candidate probabilities as a post-prior.')
    parser.add_argument('--dense-probability-root', type=Path, required=True)
    parser.add_argument('--candidate-score-root', type=Path, required=True)
    parser.add_argument('--feature-set', default='geometry_plus_goes_frp_motion_recent_firepred')
    parser.add_argument('--ts', type=int, required=True)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--alphas', default='0,0.25,0.5,1,2')
    parser.add_argument('--dense-thresholds', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--candidate-neutral-probability', type=float, default=0.8)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    if not 0 < args.candidate_neutral_probability < 1:
        raise ValueError('--candidate-neutral-probability must be strictly between 0 and 1')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = expected_windows('val', args.ts, args.interval)
    test_rows = expected_windows('test', args.ts, args.interval)
    val_dense = dense_probability_for_rows(args.dense_probability_root, 'val', val_rows)
    test_dense = dense_probability_for_rows(args.dense_probability_root, 'test', test_rows)
    val_candidate = load_candidate_maps(args.candidate_score_root, args.feature_set, 'val')
    test_candidate = load_candidate_maps(args.candidate_score_root, args.feature_set, 'test')

    tuning_rows = []
    for alpha in parse_float_list(args.alphas):
        for threshold in parse_float_list(args.dense_thresholds):
            summary, _ = evaluate(
                val_rows,
                val_dense,
                val_candidate,
                alpha,
                threshold,
                args.candidate_neutral_probability,
            )
            tuning_rows.append(summary)
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ['fire_macro_iou', 'fire_macro_f1'], ascending=False
    )
    best = tuning.iloc[0]
    test_summary, test_dates = evaluate(
        test_rows,
        test_dense,
        test_candidate,
        float(best['alpha']),
        float(best['dense_threshold']),
        args.candidate_neutral_probability,
    )
    test_summary['split'] = 'test'
    test_summary['feature_set'] = args.feature_set
    test_summary['ts'] = args.ts

    tuning_path = args.out_dir / 'dense_candidate_postprior_tuning.csv'
    summary_path = args.out_dir / 'dense_candidate_postprior_summary.csv'
    dates_path = args.out_dir / 'dense_candidate_postprior_test_dates.csv'
    tuning.to_csv(tuning_path, index=False)
    pd.DataFrame([test_summary]).to_csv(summary_path, index=False)
    test_dates.to_csv(dates_path, index=False)
    print('Best validation setting:')
    print(best.to_string())
    print('\nTest result:')
    print(pd.Series(test_summary).to_string())
    print(f'Wrote {tuning_path}')
    print(f'Wrote {summary_path}')
    print(f'Wrote {dates_path}')


if __name__ == '__main__':
    main()
