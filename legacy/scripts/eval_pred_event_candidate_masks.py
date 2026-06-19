from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_CANDIDATE_ROOT = Path('/home/jlc3q/data/SatFire/event_candidates')
TARGET = 'label_ignited_next_day'
KEY_COLS = ['fire_id', 'date']
ID_LOAD_COLS = ['fire_id', 'date', 'component_id']
MASK_SIZE = 256

GEOMETRY_NUM = [
    'distance_px',
    'component_area',
    'component_front_pixels',
    'current_fire_pixels',
]
GEOMETRY_CAT = ['direction_bin_8']
GOES_FRP = [
    'goes_frp_sum_at_candidate',
    'goes_frp_max_at_candidate',
    'goes_frp_sum_5x5_max',
    'goes_frp_sum_5x5_mean',
    'goes_frp_max_5x5_max',
    'goes_weighted_active_at_candidate',
    'goes_weighted_active_5x5_max',
    'goes_weighted_active_5x5_mean',
    'goes_distance_to_high_frp',
]
GOES_SUBDAILY_MOTION = [
    'goes_subdaily_early_frp_at_candidate',
    'goes_subdaily_late_frp_at_candidate',
    'goes_subdaily_frp_late_minus_early_at_candidate',
    'goes_subdaily_early_active_at_candidate',
    'goes_subdaily_late_active_at_candidate',
    'goes_subdaily_new_active_late_at_candidate',
    'goes_subdaily_frp_late_minus_early_5x5_mean',
    'goes_subdaily_new_active_late_5x5_max',
    'goes_subdaily_frame_count',
    'goes_subdaily_valid_active_fraction',
    'goes_subdaily_valid_frp_fraction',
    'goes_subdaily_early_frame_count',
    'goes_subdaily_late_frame_count',
    'goes_subdaily_peak_frp_hour_sin',
    'goes_subdaily_peak_frp_hour_cos',
    'goes_subdaily_peak_frp_total_log1p',
    'goes_subdaily_frp_motion_distance',
    'goes_subdaily_active_motion_distance',
    'goes_subdaily_candidate_frp_motion_alignment',
    'goes_subdaily_candidate_active_motion_alignment',
    'goes_subdaily_component_frp_motion_distance',
    'goes_subdaily_component_frp_motion_alignment',
    'goes_subdaily_component_active_motion_distance',
    'goes_subdaily_component_active_motion_alignment',
]


def suffix(
    split: str,
    connectivity: int,
    candidate_radius: float,
    min_component_pixels: int,
    history_days: int = 1,
    allow_partial_history: bool = False,
) -> str:
    radius_tag = str(candidate_radius).replace('.', 'p')
    if history_days <= 1:
        history_tag = ''
    else:
        partial_tag = '_partial' if allow_partial_history else ''
        history_tag = f'_h{history_days}{partial_tag}'
    return f'{split}_conn{connectivity}_r{radius_tag}_mincomp{min_component_pixels}{history_tag}'


def candidate_path(
    root: Path,
    split: str,
    connectivity: int,
    candidate_radius: float,
    min_component_pixels: int,
    history_days: int = 1,
    allow_partial_history: bool = False,
    goes_variant: str = 'goes_frp',
) -> Path:
    return root / f'pred_event_candidates_{suffix(split, connectivity, candidate_radius, min_component_pixels, history_days, allow_partial_history)}_{goes_variant}.csv'


def history_num_features(history_days: int, allow_partial_history: bool = False) -> list[str]:
    if history_days <= 1:
        return []
    cols = [
        'history_days',
        'history_fire_pixels_mean',
        'history_fire_pixels_max',
        'history_growth_pixels_sum',
        'history_growth_adjacent_pixels_sum',
        'history_candidate_active_days',
        'history_nearest_active_days',
    ]
    if allow_partial_history:
        cols.extend(['available_history_days', 'history_coverage_fraction'])
    for i in range(history_days):
        cols.extend([
            f'history_fire_pixels_lag{i}',
            f'history_growth_pixels_lag{i}',
            f'history_growth_adjacent_pixels_lag{i}',
        ])
    return cols


def goes_history_features(history_days: int, allow_partial_history: bool = False) -> list[str]:
    if history_days <= 1:
        return []
    bases = [
        'goes_frp_sum_at_candidate',
        'goes_frp_max_at_candidate',
        'goes_active_frequency_at_candidate',
        'goes_weighted_active_at_candidate',
        'goes_frp_sum_5x5_max',
        'goes_weighted_active_5x5_max',
    ]
    cols = []
    for base in bases:
        for i in range(history_days):
            cols.append(f'{base}_lag{i}')
        cols.extend([f'{base}_hist_mean', f'{base}_hist_max', f'{base}_hist_sum'])
    cols.extend(['goes_frp_total_log1p_hist_sum', 'goes_weighted_active_total_hist_sum'])
    if allow_partial_history:
        cols.extend(['available_goes_history_days', 'goes_history_coverage_fraction'])
    return cols


def parse_feature_sets(value: str) -> list[str]:
    aliases = {
        'default': ['geometry_only', 'geometry_plus_goes_frp'],
        'ablation': [
            'geometry_no_history',
            'geometry_plus_viirs_history',
            'geometry_viirs_history_plus_goes_current',
            'full_viirs_goes_history',
        ],
        'all': [
            'geometry_no_history',
            'geometry_plus_viirs_history',
            'geometry_viirs_history_plus_goes_current',
            'full_viirs_goes_history',
        ],
    }
    requested = [item.strip() for item in value.split(',') if item.strip()]
    if len(requested) == 1 and requested[0] in aliases:
        return aliases[requested[0]]
    allowed = {
        'geometry_only',
        'geometry_plus_goes_frp',
        'geometry_no_history',
        'geometry_plus_viirs_history',
        'geometry_viirs_history_plus_goes_current',
        'full_viirs_goes_history',
        'geometry_plus_goes_frp_motion',
    }
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise ValueError(f'Unknown feature set(s): {unknown}. Allowed: {sorted(allowed | set(aliases))}')
    return requested


def load_split(path: Path, features: list[str], sample: int | None = None, include_keys: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cols = [TARGET, 'candidate_row', 'candidate_col'] + features
    if include_keys:
        cols += ID_LOAD_COLS
    df = pd.read_csv(path, usecols=list(dict.fromkeys(cols)))
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)
    y = df[TARGET].astype(np.int8)
    X = df[features]
    return X, y, df


def build_model(num_features: list[str], cat_features: list[str]):
    pre = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ],
        remainder='drop',
    )
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
        class_weight='balanced',
    )
    return make_pipeline(pre, clf)


def confusion_iou_f1(pred: np.ndarray, true: np.ndarray) -> tuple[int, int, int, float, float]:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    union = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    iou = tp / union if union else 1.0
    f1 = (2 * tp) / denom_f1 if denom_f1 else 1.0
    return tp, fp, fn, iou, f1


def date_mask_metrics(df: pd.DataFrame, prob: np.ndarray, method: str, value: float) -> pd.DataFrame:
    extra_cols = ['component_id'] if method == 'component_top_frac' else []
    work = df[KEY_COLS + extra_cols + ['candidate_row', 'candidate_col', TARGET]].copy()
    work['prob'] = prob
    rows = []

    for (fire_id, date), g in work.groupby(KEY_COLS, sort=False):
        true_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        pred_score = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)

        rr = g['candidate_row'].to_numpy(dtype=np.int64)
        cc = g['candidate_col'].to_numpy(dtype=np.int64)
        labels = g[TARGET].to_numpy(dtype=bool)
        scores = g['prob'].to_numpy(dtype=np.float32)

        true_mask[rr[labels], cc[labels]] = True
        # If multiple components propose the same candidate, keep max probability.
        np.maximum.at(pred_score, (rr, cc), scores)

        if method == 'threshold':
            pred_mask = pred_score >= value
        elif method == 'top_frac':
            candidate_mask = pred_score > 0
            candidate_scores = pred_score[candidate_mask]
            pred_mask = np.zeros_like(true_mask)
            if candidate_scores.size:
                k = max(1, int(np.ceil(candidate_scores.size * value)))
                cutoff = np.partition(candidate_scores, -k)[-k]
                pred_mask = candidate_mask & (pred_score >= cutoff)
        elif method == 'component_top_frac':
            pred_mask = np.zeros_like(true_mask)
            for _, cg in g.groupby('component_id', sort=False):
                crr = cg['candidate_row'].to_numpy(dtype=np.int64)
                ccc = cg['candidate_col'].to_numpy(dtype=np.int64)
                cscores = cg['prob'].to_numpy(dtype=np.float32)
                if cscores.size == 0:
                    continue
                k = max(1, int(np.ceil(cscores.size * value)))
                order = np.argpartition(cscores, -k)[-k:]
                pred_mask[crr[order], ccc[order]] = True
        else:
            raise ValueError(method)

        tp, fp, fn, iou, f1 = confusion_iou_f1(pred_mask, true_mask)
        rows.append({
            'fire_id': fire_id,
            'date': date,
            'method': method,
            'value': value,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'true_pixels': int(true_mask.sum()),
            'pred_pixels': int(pred_mask.sum()),
            'iou': iou,
            'f1': f1,
        })
    return pd.DataFrame(rows)



def firewise_metrics(date_metrics: pd.DataFrame, model: str, split: str, method: str, value: float) -> pd.DataFrame:
    rows = []
    for fire_id, g in date_metrics.groupby('fire_id', sort=False):
        tp = int(g['tp'].sum())
        fp = int(g['fp'].sum())
        fn = int(g['fn'].sum())
        union = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        rows.append({
            'model': model,
            'split': split,
            'fire_id': fire_id,
            'selected_method': method,
            'selected_value': value,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'true_pixels': int(g['true_pixels'].sum()),
            'pred_pixels': int(g['pred_pixels'].sum()),
            'n_fire_dates': int(len(g)),
            'iou': tp / union if union else 1.0,
            'f1': (2 * tp) / denom_f1 if denom_f1 else 1.0,
        })
    return pd.DataFrame(rows)


def summarize_firewise(fire_metrics: pd.DataFrame) -> dict[str, float]:
    tp = int(fire_metrics['tp'].sum())
    fp = int(fire_metrics['fp'].sum())
    fn = int(fire_metrics['fn'].sum())
    union = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    return {
        'fire_mean_iou': float(fire_metrics['iou'].mean()),
        'fire_mean_f1': float(fire_metrics['f1'].mean()),
        'fire_micro_iou': tp / union if union else 1.0,
        'fire_micro_f1': (2 * tp) / denom_f1 if denom_f1 else 1.0,
        'n_fires': int(len(fire_metrics)),
    }

def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    tp = int(metrics['tp'].sum())
    fp = int(metrics['fp'].sum())
    fn = int(metrics['fn'].sum())
    union = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    return {
        'mean_iou': float(metrics['iou'].mean()),
        'mean_f1': float(metrics['f1'].mean()),
        'micro_iou': tp / union if union else 1.0,
        'micro_f1': (2 * tp) / denom_f1 if denom_f1 else 1.0,
        'mean_pred_pixels': float(metrics['pred_pixels'].mean()),
        'mean_true_pixels': float(metrics['true_pixels'].mean()),
        'n_fire_dates': int(len(metrics)),
    }


def tune_thresholds(
    df_val: pd.DataFrame,
    prob_val: np.ndarray,
    thresholds: list[float],
    top_fracs: list[float],
    objective: str,
) -> pd.DataFrame:
    rows = []
    methods = [('threshold', thresholds), ('top_frac', top_fracs), ('component_top_frac', top_fracs)]
    for method, values in methods:
        for value in values:
            metrics = date_mask_metrics(df_val, prob_val, method, value)
            fire_metrics = firewise_metrics(metrics, model='tuning', split='val', method=method, value=value)
            rows.append({
                'method': method,
                'value': value,
                **summarize(metrics),
                **summarize_firewise(fire_metrics),
            })
    if objective not in rows[0]:
        raise ValueError(f'Unknown selection objective: {objective}')
    secondary = 'fire_mean_f1' if objective == 'fire_mean_iou' else 'mean_f1'
    if secondary not in rows[0]:
        secondary = 'mean_f1'
    return pd.DataFrame(rows).sort_values([objective, secondary], ascending=False)


def evaluate_model(name: str, root: Path, args: argparse.Namespace, num_features: list[str], cat_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = num_features + cat_features
    train_path = candidate_path(root, 'train', args.connectivity, args.candidate_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)
    val_path = candidate_path(root, 'val', args.connectivity, args.candidate_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)
    test_path = candidate_path(root, 'test', args.connectivity, args.candidate_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)

    X_train, y_train, _ = load_split(train_path, features, sample=args.train_sample)
    X_val, y_val, df_val = load_split(val_path, features, include_keys=True)
    X_test, y_test, df_test = load_split(test_path, features, include_keys=True)

    model = build_model(num_features, cat_features)
    model.fit(X_train, y_train)

    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    print(f'\n=== {name} global ===')
    print('val PR-AUC', average_precision_score(y_val, prob_val), 'ROC-AUC', roc_auc_score(y_val, prob_val))
    print('test PR-AUC', average_precision_score(y_test, prob_test), 'ROC-AUC', roc_auc_score(y_test, prob_test))

    thresholds = [float(x) for x in np.linspace(0.05, 0.95, 19)]
    top_fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    tuning = tune_thresholds(df_val, prob_val, thresholds, top_fracs, args.selection_objective)
    tuning['model'] = name
    best = tuning.iloc[0]
    print(f"best val: method={best['method']} value={best['value']} mean_iou={best['mean_iou']:.6f} mean_f1={best['mean_f1']:.6f}")

    method = str(best['method'])
    value = float(best['value'])
    val_metrics = date_mask_metrics(df_val, prob_val, method, value)
    test_metrics = date_mask_metrics(df_test, prob_test, method, value)

    val_fire = firewise_metrics(val_metrics, name, 'val', method, value)
    test_fire = firewise_metrics(test_metrics, name, 'test', method, value)
    fire_metrics = pd.concat([val_fire, test_fire], ignore_index=True)

    val_summary = pd.DataFrame([{
        'model': name,
        'split': 'val',
        'selected_method': method,
        'selected_value': value,
        **summarize(val_metrics),
        **summarize_firewise(val_fire),
    }])
    test_summary = pd.DataFrame([{
        'model': name,
        'split': 'test',
        'selected_method': method,
        'selected_value': value,
        **summarize(test_metrics),
        **summarize_firewise(test_fire),
    }])
    summary = pd.concat([val_summary, test_summary], ignore_index=True)
    return tuning, summary, fire_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Reconstruct 256x256 masks from event candidate probabilities and evaluate IoU/F1.')
    parser.add_argument('--candidate-root', type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument('--connectivity', type=int, default=8)
    parser.add_argument('--candidate-radius', type=float, default=5.0)
    parser.add_argument('--min-component-pixels', type=int, default=1)
    parser.add_argument('--train-sample', type=int, default=1_000_000)
    parser.add_argument('--history-days', type=int, default=1, help='Candidate/GOES history window. Use 2/4/6 for TS-matched experiments.')
    parser.add_argument(
        '--allow-partial-history',
        action='store_true',
        help='Use _partial candidate files where early windows keep fewer than history-days inputs.',
    )
    parser.add_argument(
        '--feature-sets',
        default='default',
        help=(
            'Comma-separated feature sets or alias: default, ablation, all. '
            'Useful sets: geometry_plus_viirs_history, '
            'geometry_viirs_history_plus_goes_current, full_viirs_goes_history.'
        ),
    )
    parser.add_argument(
        '--goes-variant',
        choices=['goes_frp', 'goes_frp_motion'],
        default='goes_frp',
        help='GOES-enriched candidate file suffix to evaluate.',
    )
    parser.add_argument(
        '--selection-objective',
        choices=['mean_iou', 'mean_f1', 'micro_iou', 'micro_f1', 'fire_mean_iou', 'fire_mean_f1', 'fire_micro_iou', 'fire_micro_f1'],
        default='fire_mean_iou',
        help='Validation metric used to select threshold/top-fraction mask reconstruction.',
    )
    args = parser.parse_args()

    if args.history_days < 1:
        raise ValueError('--history-days must be >= 1')
    root = args.candidate_root
    outputs = []
    tunings = []
    fire_outputs = []

    geom_num = GEOMETRY_NUM + history_num_features(args.history_days, args.allow_partial_history)
    goes_num = GOES_FRP + goes_history_features(args.history_days, args.allow_partial_history)
    feature_specs = {
        'geometry_only': (geom_num, GEOMETRY_CAT),
        'geometry_plus_goes_frp': (geom_num + goes_num, GEOMETRY_CAT),
        'geometry_no_history': (GEOMETRY_NUM, GEOMETRY_CAT),
        'geometry_plus_viirs_history': (geom_num, GEOMETRY_CAT),
        'geometry_viirs_history_plus_goes_current': (geom_num + GOES_FRP, GEOMETRY_CAT),
        'full_viirs_goes_history': (geom_num + goes_num, GEOMETRY_CAT),
        'geometry_plus_goes_frp_motion': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION,
            GEOMETRY_CAT,
        ),
    }

    for feature_set in parse_feature_sets(args.feature_sets):
        num_features, cat_features = feature_specs[feature_set]
        tuning, summary, fire_metrics = evaluate_model(feature_set, root, args, num_features, cat_features)
        tunings.append(tuning)
        outputs.append(summary)
        fire_outputs.append(fire_metrics)

    tuning_df = pd.concat(tunings, ignore_index=True)
    summary_df = pd.concat(outputs, ignore_index=True)
    fire_df = pd.concat(fire_outputs, ignore_index=True)

    if args.goes_variant == 'goes_frp':
        output_tag = ''
    else:
        partial_tag = '_partial' if args.allow_partial_history and args.history_days > 1 else ''
        history_tag = f'_h{args.history_days}{partial_tag}' if args.history_days > 1 else ''
        output_tag = f'{history_tag}_{args.goes_variant}'
    tuning_out = root / f'pred_event_mask_eval_threshold_tuning{output_tag}.csv'
    summary_out = root / f'pred_event_mask_eval_summary{output_tag}.csv'
    fire_out = root / f'pred_event_mask_eval_firewise{output_tag}.csv'
    tuning_df.to_csv(tuning_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    fire_df.to_csv(fire_out, index=False)
    print('\nWrote', tuning_out)
    print('Wrote', summary_out)
    print('Wrote', fire_out)
    print('\nSummary:')
    print(summary_df)


if __name__ == '__main__':
    main()
