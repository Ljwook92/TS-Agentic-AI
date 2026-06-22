from __future__ import annotations

import argparse
import gc
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
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


def split_local_remote_growth(
    current: np.ndarray,
    growth: np.ndarray,
    local_spread_radius: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if local_spread_radius < 0:
        raise ValueError('local_spread_radius must be >= 0')
    structure = ndimage.generate_binary_structure(2, 2)
    growth_labels, n_components = ndimage.label(growth, structure=structure)
    if n_components == 0:
        empty = np.zeros_like(growth, dtype=bool)
        return empty, empty.copy(), 0, 0

    distance_to_current = ndimage.distance_transform_edt(~current)
    component_ids = np.arange(1, n_components + 1)
    minimum_distances = np.asarray(
        ndimage.minimum(distance_to_current, labels=growth_labels, index=component_ids),
        dtype=np.float64,
    )
    local_ids = component_ids[minimum_distances <= local_spread_radius]
    remote_ids = component_ids[minimum_distances > local_spread_radius]
    local_growth = np.isin(growth_labels, local_ids)
    remote_growth = np.isin(growth_labels, remote_ids)
    return local_growth, remote_growth, int(local_ids.size), int(remote_ids.size)


class FullGrowthTruth:
    def __init__(self, split: str, local_spread_radius: float = 5.0):
        try:
            from analyze_pred_event_windows import load_daily_masks, resolve_locations
        except ModuleNotFoundError:
            from scripts.analyze_pred_event_windows import load_daily_masks, resolve_locations

        self.split = split
        self.local_spread_radius = float(local_spread_radius)
        if self.local_spread_radius < 0:
            raise ValueError('local_spread_radius must be >= 0')
        self.label_selectors = dict(resolve_locations(split))
        self.load_daily_masks = load_daily_masks

    @lru_cache(maxsize=None)
    def event_masks(self, fire_id: str) -> tuple[list[str], list[np.ndarray]]:
        fire_id = str(fire_id)
        if fire_id not in self.label_selectors:
            raise KeyError(f'{fire_id} is not present in the {self.split} ROI table')
        return self.load_daily_masks(fire_id, int(self.label_selectors[fire_id]))

    @lru_cache(maxsize=None)
    def growth_partition(
        self,
        fire_id: str,
        date: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        dates, masks = self.event_masks(str(fire_id))
        date = str(date)
        if date not in dates:
            raise KeyError(f'{fire_id} date {date} is not present in VIIRS masks')
        day_idx = dates.index(date)
        if day_idx + 1 >= len(masks):
            raise IndexError(f'{fire_id} date {date} has no following VIIRS mask')
        current = masks[day_idx].astype(bool)
        growth = masks[day_idx + 1].astype(bool) & ~current
        local_growth, remote_growth, n_local, n_remote = split_local_remote_growth(
            current,
            growth,
            self.local_spread_radius,
        )
        return current, growth, local_growth, remote_growth, n_local, n_remote

    def current_and_growth(self, fire_id: str, date: str) -> tuple[np.ndarray, np.ndarray]:
        current, growth, _, _, _, _ = self.growth_partition(str(fire_id), str(date))
        return current, growth

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
GOES_RECENT_MOTION = [
    'goes_recent3_frp_at_candidate',
    'goes_recent6_frp_at_candidate',
    'goes_previous6_frp_at_candidate',
    'goes_recent6_frp_delta_at_candidate',
    'goes_recent3_active_at_candidate',
    'goes_recent6_active_at_candidate',
    'goes_previous6_active_at_candidate',
    'goes_new_active_recent6_at_candidate',
    'goes_recent6_frp_delta_5x5_mean',
    'goes_new_active_recent6_5x5_max',
    'goes_recent3_frame_count',
    'goes_recent6_frame_count',
    'goes_previous6_frame_count',
    'goes_recent_frp_motion_distance',
    'goes_recent_active_motion_distance',
    'goes_recent_candidate_frp_motion_alignment',
    'goes_recent_candidate_active_motion_alignment',
    'goes_recent_component_frp_motion_distance',
    'goes_recent_component_frp_motion_alignment',
    'goes_recent_component_active_motion_distance',
    'goes_recent_component_active_motion_alignment',
]
FIREPRED_NUM = [
    'firepred_available',
    'firepred_ndvi_candidate',
    'firepred_ndvi_5x5_mean',
    'firepred_evi2_candidate',
    'firepred_evi2_5x5_mean',
    'firepred_precip_candidate',
    'firepred_precip_5x5_mean',
    'firepred_wind_speed_candidate',
    'firepred_wind_speed_5x5_mean',
    'firepred_tmin_candidate',
    'firepred_tmax_candidate',
    'firepred_erc_candidate',
    'firepred_erc_5x5_mean',
    'firepred_specific_humidity_candidate',
    'firepred_specific_humidity_5x5_mean',
    'firepred_slope_candidate',
    'firepred_slope_5x5_mean',
    'firepred_elevation_candidate',
    'firepred_elevation_5x5_mean',
    'firepred_pdsi_candidate',
    'firepred_pdsi_5x5_mean',
    'firepred_forecast_precip_candidate',
    'firepred_forecast_precip_5x5_mean',
    'firepred_forecast_wind_speed_candidate',
    'firepred_forecast_wind_speed_5x5_mean',
    'firepred_forecast_temperature_candidate',
    'firepred_forecast_temperature_5x5_mean',
    'firepred_forecast_specific_humidity_candidate',
    'firepred_forecast_specific_humidity_5x5_mean',
    'firepred_elevation_delta_from_fire',
    'firepred_directional_elevation_gradient',
    'firepred_landcover_same_as_fire',
    'firepred_wind_candidate_alignment',
    'firepred_wind_candidate_cross_alignment',
    'firepred_wind_candidate_push',
    'firepred_wind_candidate_crosswind',
    'firepred_forecast_wind_candidate_alignment',
    'firepred_forecast_wind_candidate_cross_alignment',
    'firepred_forecast_wind_candidate_push',
    'firepred_forecast_wind_candidate_crosswind',
    'firepred_uphill_candidate_alignment',
]
FIREPRED_CAT = [
    'firepred_landcover_candidate',
    'firepred_landcover_nearest_fire',
    'firepred_landcover_transition',
]
FIREPRED_OBSERVED_WIND = [
    'firepred_available',
    'firepred_wind_speed_candidate',
    'firepred_wind_speed_5x5_mean',
    'firepred_wind_candidate_alignment',
    'firepred_wind_candidate_cross_alignment',
    'firepred_wind_candidate_push',
    'firepred_wind_candidate_crosswind',
]
FIREPRED_FORECAST_WIND = [
    'firepred_available',
    'firepred_forecast_wind_speed_candidate',
    'firepred_forecast_wind_speed_5x5_mean',
    'firepred_forecast_wind_candidate_alignment',
    'firepred_forecast_wind_candidate_cross_alignment',
    'firepred_forecast_wind_candidate_push',
    'firepred_forecast_wind_candidate_crosswind',
]
FIREPRED_FUEL_WEATHER = [
    'firepred_available',
    'firepred_ndvi_candidate',
    'firepred_ndvi_5x5_mean',
    'firepred_evi2_candidate',
    'firepred_evi2_5x5_mean',
    'firepred_precip_candidate',
    'firepred_precip_5x5_mean',
    'firepred_tmin_candidate',
    'firepred_tmax_candidate',
    'firepred_erc_candidate',
    'firepred_erc_5x5_mean',
    'firepred_specific_humidity_candidate',
    'firepred_specific_humidity_5x5_mean',
    'firepred_pdsi_candidate',
    'firepred_pdsi_5x5_mean',
    'firepred_forecast_precip_candidate',
    'firepred_forecast_precip_5x5_mean',
    'firepred_forecast_temperature_candidate',
    'firepred_forecast_temperature_5x5_mean',
    'firepred_forecast_specific_humidity_candidate',
    'firepred_forecast_specific_humidity_5x5_mean',
]
FIREPRED_TERRAIN = [
    'firepred_available',
    'firepred_slope_candidate',
    'firepred_slope_5x5_mean',
    'firepred_elevation_candidate',
    'firepred_elevation_5x5_mean',
    'firepred_elevation_delta_from_fire',
    'firepred_directional_elevation_gradient',
    'firepred_landcover_same_as_fire',
    'firepred_uphill_candidate_alignment',
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
        'recent_ablation': [
            'geometry_plus_goes_frp_motion',
            'geometry_plus_goes_frp_motion_firepred',
            'geometry_plus_goes_frp_motion_recent',
            'geometry_plus_goes_frp_motion_recent_observed_wind',
            'geometry_plus_goes_frp_motion_recent_forecast_wind',
            'geometry_plus_goes_frp_motion_recent_fuel_weather',
            'geometry_plus_goes_frp_motion_recent_terrain',
            'geometry_plus_goes_frp_motion_recent_firepred',
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
        'geometry_plus_firepred',
        'geometry_plus_goes_frp_firepred',
        'geometry_plus_goes_frp_motion_firepred',
        'geometry_plus_goes_frp_motion_recent',
        'geometry_plus_goes_frp_motion_recent_observed_wind',
        'geometry_plus_goes_frp_motion_recent_forecast_wind',
        'geometry_plus_goes_frp_motion_recent_fuel_weather',
        'geometry_plus_goes_frp_motion_recent_terrain',
        'geometry_plus_goes_frp_motion_recent_firepred',
    }
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise ValueError(f'Unknown feature set(s): {unknown}. Allowed: {sorted(allowed | set(aliases))}')
    return requested


def load_split(
    path: Path,
    features: list[str],
    sample: int | None = None,
    include_keys: bool = False,
    max_candidate_distance: float | None = None,
    read_chunksize: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cols = [TARGET, 'candidate_row', 'candidate_col'] + features
    if max_candidate_distance is not None:
        cols.append('distance_px')
    if include_keys:
        cols += ID_LOAD_COLS
    usecols = list(dict.fromkeys(cols))
    if read_chunksize is None:
        df = pd.read_csv(path, usecols=usecols)
        if max_candidate_distance is not None:
            df = df[df['distance_px'] <= max_candidate_distance].copy()
        if sample is not None and len(df) > sample:
            df = df.sample(sample, random_state=42)
    else:
        pieces = []
        reservoir = None
        rng = np.random.default_rng(42)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=read_chunksize):
            if max_candidate_distance is not None:
                chunk = chunk[chunk['distance_px'] <= max_candidate_distance].copy()
            if chunk.empty:
                continue
            if sample is None:
                pieces.append(chunk)
                continue

            chunk['_sample_key'] = rng.random(len(chunk))
            reservoir = (
                chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
            )
            if len(reservoir) > sample:
                reservoir = reservoir.nsmallest(sample, '_sample_key')
        if sample is None:
            df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)
        elif reservoir is None:
            df = pd.DataFrame(columns=usecols)
        else:
            df = reservoir.drop(columns='_sample_key').reset_index(drop=True)
    df = df.reset_index(drop=True)
    y = df[TARGET].astype(np.int8)
    X = df[features]
    return X, y, df


def replace_target_with_local_growth(
    df: pd.DataFrame,
    truth: FullGrowthTruth,
) -> pd.Series:
    required = set(KEY_COLS + ['candidate_row', 'candidate_col'])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Local target construction requires columns: {sorted(missing)}')

    labels = np.zeros(len(df), dtype=np.int8)
    for (fire_id, date), positions in df.groupby(KEY_COLS, sort=False).indices.items():
        local_growth = truth.growth_partition(str(fire_id), str(date))[2]
        group = df.iloc[positions]
        rr = group['candidate_row'].to_numpy(dtype=np.int64)
        cc = group['candidate_col'].to_numpy(dtype=np.int64)
        labels[positions] = local_growth[rr, cc].astype(np.int8)
    df[TARGET] = labels
    return df[TARGET].astype(np.int8)


def predict_split_chunked(
    path: Path,
    model,
    features: list[str],
    max_candidate_distance: float,
    read_chunksize: int,
    target_scope: str,
    truth: FullGrowthTruth | None,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    cols = list(dict.fromkeys(
        [TARGET, 'candidate_row', 'candidate_col', *ID_LOAD_COLS, 'distance_px', *features]
    ))
    metadata_parts = []
    targets = []
    probabilities = []
    metadata_cols = list(dict.fromkeys(
        [*ID_LOAD_COLS, 'candidate_row', 'candidate_col', TARGET]
    ))

    for chunk in pd.read_csv(path, usecols=cols, chunksize=read_chunksize):
        chunk = chunk[chunk['distance_px'] <= max_candidate_distance].reset_index(drop=True)
        if chunk.empty:
            continue
        if target_scope == 'local':
            if truth is None:
                raise ValueError('Local target scope requires full growth truth')
            y_chunk = replace_target_with_local_growth(chunk, truth)
        else:
            y_chunk = chunk[TARGET].astype(np.int8)
        probability = model.predict_proba(chunk[features])[:, 1].astype(np.float32)

        metadata = chunk[metadata_cols].copy()
        metadata['component_id'] = metadata['component_id'].astype(np.int32)
        metadata['candidate_row'] = metadata['candidate_row'].astype(np.int16)
        metadata['candidate_col'] = metadata['candidate_col'].astype(np.int16)
        metadata[TARGET] = y_chunk.to_numpy(dtype=np.int8)
        metadata_parts.append(metadata)
        targets.append(y_chunk.to_numpy(dtype=np.int8))
        probabilities.append(probability)

    if not metadata_parts:
        raise ValueError(
            f'No candidate rows remain at radius={max_candidate_distance} in {path}'
        )
    metadata = pd.concat(metadata_parts, ignore_index=True)
    for column in KEY_COLS:
        metadata[column] = metadata[column].astype('category')
    target = pd.Series(np.concatenate(targets), name=TARGET)
    probability = np.concatenate(probabilities)
    return target, metadata, probability


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


def date_mask_metrics(
    df: pd.DataFrame,
    prob: np.ndarray,
    method: str,
    value: float,
    full_growth_truth: FullGrowthTruth | None = None,
    target_scope: str = 'all',
) -> pd.DataFrame:
    extra_cols = ['component_id'] if method == 'component_top_frac' else []
    work = df[KEY_COLS + extra_cols + ['candidate_row', 'candidate_col', TARGET]].copy()
    work['prob'] = prob
    rows = []

    for (fire_id, date), g in work.groupby(KEY_COLS, sort=False):
        true_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        pred_score = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
        candidate_support = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)

        rr = g['candidate_row'].to_numpy(dtype=np.int64)
        cc = g['candidate_col'].to_numpy(dtype=np.int64)
        labels = g[TARGET].to_numpy(dtype=bool)
        scores = g['prob'].to_numpy(dtype=np.float32)

        true_mask[rr[labels], cc[labels]] = True
        candidate_support[rr, cc] = True
        # If multiple components propose the same candidate, keep max probability.
        np.maximum.at(pred_score, (rr, cc), scores)

        if method == 'threshold':
            pred_mask = pred_score >= value
        elif method == 'top_frac':
            candidate_scores = pred_score[candidate_support]
            pred_mask = np.zeros_like(true_mask)
            if candidate_scores.size:
                k = max(1, int(np.ceil(candidate_scores.size * value)))
                cutoff = np.partition(candidate_scores, -k)[-k]
                pred_mask = candidate_support & (pred_score >= cutoff)
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
        row = {
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
        }
        if full_growth_truth is not None:
            (
                _,
                full_true_mask,
                local_true_mask,
                remote_true_mask,
                local_component_count,
                remote_component_count,
            ) = full_growth_truth.growth_partition(str(fire_id), str(date))
            target_true_mask = local_true_mask if target_scope == 'local' else full_true_mask
            covered_true_mask = target_true_mask & candidate_support
            mismatch = int(np.logical_xor(true_mask, covered_true_mask).sum())
            if mismatch:
                raise ValueError(
                    f'{fire_id} {date}: candidate labels disagree with raw VIIRS growth '
                    f'at {mismatch} pixels'
                )
            full_tp, full_fp, full_fn, full_iou, full_f1 = confusion_iou_f1(pred_mask, full_true_mask)
            full_true_pixels = int(full_true_mask.sum())
            covered_true_pixels = int(covered_true_mask.sum())
            local_tp, local_fp, local_fn, local_iou, local_f1 = confusion_iou_f1(
                pred_mask,
                local_true_mask,
            )
            local_true_pixels = int(local_true_mask.sum())
            local_covered_true_pixels = int((local_true_mask & candidate_support).sum())
            remote_true_pixels = int(remote_true_mask.sum())
            row.update({
                'full_tp': full_tp,
                'full_fp': full_fp,
                'full_fn': full_fn,
                'full_true_pixels': full_true_pixels,
                'candidate_covered_true_pixels': covered_true_pixels,
                'candidate_coverage': covered_true_pixels / full_true_pixels if full_true_pixels else 1.0,
                'full_iou': full_iou,
                'full_f1': full_f1,
                'local_tp': local_tp,
                'local_fp': local_fp,
                'local_fn': local_fn,
                'local_true_pixels': local_true_pixels,
                'local_candidate_covered_true_pixels': local_covered_true_pixels,
                'local_candidate_coverage': (
                    local_covered_true_pixels / local_true_pixels if local_true_pixels else 1.0
                ),
                'local_iou': local_iou,
                'local_f1': local_f1,
                'local_component_count': local_component_count,
                'remote_true_pixels': remote_true_pixels,
                'remote_component_count': remote_component_count,
                'remote_growth_fraction': (
                    remote_true_pixels / full_true_pixels if full_true_pixels else 0.0
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)



def firewise_metrics(date_metrics: pd.DataFrame, model: str, split: str, method: str, value: float) -> pd.DataFrame:
    rows = []
    for fire_id, g in date_metrics.groupby('fire_id', sort=False):
        tp = int(g['tp'].sum())
        fp = int(g['fp'].sum())
        fn = int(g['fn'].sum())
        union = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        row = {
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
        }
        if 'full_tp' in g:
            full_tp = int(g['full_tp'].sum())
            full_fp = int(g['full_fp'].sum())
            full_fn = int(g['full_fn'].sum())
            full_union = full_tp + full_fp + full_fn
            full_f1_denom = 2 * full_tp + full_fp + full_fn
            full_true_pixels = int(g['full_true_pixels'].sum())
            covered_true_pixels = int(g['candidate_covered_true_pixels'].sum())
            row.update({
                'full_tp': full_tp,
                'full_fp': full_fp,
                'full_fn': full_fn,
                'full_true_pixels': full_true_pixels,
                'candidate_covered_true_pixels': covered_true_pixels,
                'candidate_coverage': covered_true_pixels / full_true_pixels if full_true_pixels else 1.0,
                'full_iou': full_tp / full_union if full_union else 1.0,
                'full_f1': (2 * full_tp) / full_f1_denom if full_f1_denom else 1.0,
            })
            local_tp = int(g['local_tp'].sum())
            local_fp = int(g['local_fp'].sum())
            local_fn = int(g['local_fn'].sum())
            local_union = local_tp + local_fp + local_fn
            local_f1_denom = 2 * local_tp + local_fp + local_fn
            local_true_pixels = int(g['local_true_pixels'].sum())
            local_covered_true_pixels = int(g['local_candidate_covered_true_pixels'].sum())
            remote_true_pixels = int(g['remote_true_pixels'].sum())
            row.update({
                'local_tp': local_tp,
                'local_fp': local_fp,
                'local_fn': local_fn,
                'local_true_pixels': local_true_pixels,
                'local_candidate_covered_true_pixels': local_covered_true_pixels,
                'local_candidate_coverage': (
                    local_covered_true_pixels / local_true_pixels if local_true_pixels else 1.0
                ),
                'local_iou': local_tp / local_union if local_union else 1.0,
                'local_f1': (2 * local_tp) / local_f1_denom if local_f1_denom else 1.0,
                'local_component_count': int(g['local_component_count'].sum()),
                'remote_true_pixels': remote_true_pixels,
                'remote_component_count': int(g['remote_component_count'].sum()),
                'remote_growth_fraction': (
                    remote_true_pixels / full_true_pixels if full_true_pixels else 0.0
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_firewise(fire_metrics: pd.DataFrame) -> dict[str, float]:
    tp = int(fire_metrics['tp'].sum())
    fp = int(fire_metrics['fp'].sum())
    fn = int(fire_metrics['fn'].sum())
    union = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    summary = {
        'fire_mean_iou': float(fire_metrics['iou'].mean()),
        'fire_mean_f1': float(fire_metrics['f1'].mean()),
        'fire_micro_iou': tp / union if union else 1.0,
        'fire_micro_f1': (2 * tp) / denom_f1 if denom_f1 else 1.0,
        'n_fires': int(len(fire_metrics)),
    }
    if 'full_tp' in fire_metrics:
        full_tp = int(fire_metrics['full_tp'].sum())
        full_fp = int(fire_metrics['full_fp'].sum())
        full_fn = int(fire_metrics['full_fn'].sum())
        full_union = full_tp + full_fp + full_fn
        full_f1_denom = 2 * full_tp + full_fp + full_fn
        full_true_pixels = int(fire_metrics['full_true_pixels'].sum())
        covered_true_pixels = int(fire_metrics['candidate_covered_true_pixels'].sum())
        summary.update({
            'fire_full_mean_iou': float(fire_metrics['full_iou'].mean()),
            'fire_full_mean_f1': float(fire_metrics['full_f1'].mean()),
            'fire_full_micro_iou': full_tp / full_union if full_union else 1.0,
            'fire_full_micro_f1': (2 * full_tp) / full_f1_denom if full_f1_denom else 1.0,
            'fire_candidate_coverage_mean': float(fire_metrics['candidate_coverage'].mean()),
            'fire_candidate_coverage_micro': covered_true_pixels / full_true_pixels if full_true_pixels else 1.0,
        })
        local_tp = int(fire_metrics['local_tp'].sum())
        local_fp = int(fire_metrics['local_fp'].sum())
        local_fn = int(fire_metrics['local_fn'].sum())
        local_union = local_tp + local_fp + local_fn
        local_f1_denom = 2 * local_tp + local_fp + local_fn
        local_true_pixels = int(fire_metrics['local_true_pixels'].sum())
        local_covered_true_pixels = int(fire_metrics['local_candidate_covered_true_pixels'].sum())
        remote_true_pixels = int(fire_metrics['remote_true_pixels'].sum())
        positive_local = fire_metrics[fire_metrics['local_true_pixels'] > 0]
        summary.update({
            'fire_local_positive_mean_iou': (
                float(positive_local['local_iou'].mean()) if len(positive_local) else np.nan
            ),
            'fire_local_positive_mean_f1': (
                float(positive_local['local_f1'].mean()) if len(positive_local) else np.nan
            ),
            'fire_local_micro_iou': local_tp / local_union if local_union else 1.0,
            'fire_local_micro_f1': (
                (2 * local_tp) / local_f1_denom if local_f1_denom else 1.0
            ),
            'fire_local_candidate_coverage_mean': (
                float(positive_local['local_candidate_coverage'].mean())
                if len(positive_local) else np.nan
            ),
            'fire_local_candidate_coverage_micro': (
                local_covered_true_pixels / local_true_pixels if local_true_pixels else 1.0
            ),
            'fire_remote_growth_fraction_micro': (
                remote_true_pixels / full_true_pixels if full_true_pixels else 0.0
            ),
            'n_local_positive_fires': int(len(positive_local)),
        })
    return summary

def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    tp = int(metrics['tp'].sum())
    fp = int(metrics['fp'].sum())
    fn = int(metrics['fn'].sum())
    union = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    summary = {
        'mean_iou': float(metrics['iou'].mean()),
        'mean_f1': float(metrics['f1'].mean()),
        'micro_iou': tp / union if union else 1.0,
        'micro_f1': (2 * tp) / denom_f1 if denom_f1 else 1.0,
        'mean_pred_pixels': float(metrics['pred_pixels'].mean()),
        'mean_true_pixels': float(metrics['true_pixels'].mean()),
        'n_fire_dates': int(len(metrics)),
    }
    if 'full_tp' in metrics:
        full_tp = int(metrics['full_tp'].sum())
        full_fp = int(metrics['full_fp'].sum())
        full_fn = int(metrics['full_fn'].sum())
        full_union = full_tp + full_fp + full_fn
        full_f1_denom = 2 * full_tp + full_fp + full_fn
        full_true_pixels = int(metrics['full_true_pixels'].sum())
        covered_true_pixels = int(metrics['candidate_covered_true_pixels'].sum())
        summary.update({
            'full_mean_iou': float(metrics['full_iou'].mean()),
            'full_mean_f1': float(metrics['full_f1'].mean()),
            'full_micro_iou': full_tp / full_union if full_union else 1.0,
            'full_micro_f1': (2 * full_tp) / full_f1_denom if full_f1_denom else 1.0,
            'candidate_coverage_mean': float(metrics['candidate_coverage'].mean()),
            'candidate_coverage_micro': covered_true_pixels / full_true_pixels if full_true_pixels else 1.0,
        })
        local_tp = int(metrics['local_tp'].sum())
        local_fp = int(metrics['local_fp'].sum())
        local_fn = int(metrics['local_fn'].sum())
        local_union = local_tp + local_fp + local_fn
        local_f1_denom = 2 * local_tp + local_fp + local_fn
        local_true_pixels = int(metrics['local_true_pixels'].sum())
        local_covered_true_pixels = int(metrics['local_candidate_covered_true_pixels'].sum())
        remote_true_pixels = int(metrics['remote_true_pixels'].sum())
        positive_local = metrics[metrics['local_true_pixels'] > 0]
        empty_local = metrics[metrics['local_true_pixels'] == 0]
        summary.update({
            'local_positive_mean_iou': (
                float(positive_local['local_iou'].mean()) if len(positive_local) else np.nan
            ),
            'local_positive_mean_f1': (
                float(positive_local['local_f1'].mean()) if len(positive_local) else np.nan
            ),
            'local_micro_iou': local_tp / local_union if local_union else 1.0,
            'local_micro_f1': (2 * local_tp) / local_f1_denom if local_f1_denom else 1.0,
            'local_candidate_coverage_mean': (
                float(positive_local['local_candidate_coverage'].mean())
                if len(positive_local) else np.nan
            ),
            'local_candidate_coverage_micro': (
                local_covered_true_pixels / local_true_pixels if local_true_pixels else 1.0
            ),
            'local_empty_correct_rate': (
                float((empty_local['pred_pixels'] == 0).mean()) if len(empty_local) else np.nan
            ),
            'remote_growth_fraction_micro': (
                remote_true_pixels / full_true_pixels if full_true_pixels else 0.0
            ),
            'n_local_positive_dates': int(len(positive_local)),
            'n_local_empty_dates': int(len(empty_local)),
        })
    return summary


def tune_thresholds(
    df_val: pd.DataFrame,
    prob_val: np.ndarray,
    thresholds: list[float],
    top_fracs: list[float],
    objective: str,
    full_growth_truth: FullGrowthTruth | None = None,
    target_scope: str = 'all',
) -> pd.DataFrame:
    rows = []
    methods = [('threshold', thresholds), ('top_frac', top_fracs), ('component_top_frac', top_fracs)]
    for method, values in methods:
        for value in values:
            metrics = date_mask_metrics(
                df_val,
                prob_val,
                method,
                value,
                full_growth_truth,
                target_scope,
            )
            fire_metrics = firewise_metrics(metrics, model='tuning', split='val', method=method, value=value)
            rows.append({
                'method': method,
                'value': value,
                **summarize(metrics),
                **summarize_firewise(fire_metrics),
            })
    if objective not in rows[0]:
        raise ValueError(f'Unknown selection objective: {objective}')
    if objective == 'fire_full_mean_iou':
        secondary = 'fire_full_mean_f1'
    elif objective == 'full_mean_iou':
        secondary = 'full_mean_f1'
    elif objective == 'fire_local_positive_mean_iou':
        secondary = 'fire_local_positive_mean_f1'
    elif objective == 'local_positive_mean_iou':
        secondary = 'local_positive_mean_f1'
    else:
        secondary = 'fire_mean_f1' if objective == 'fire_mean_iou' else 'mean_f1'
    if secondary not in rows[0]:
        secondary = 'mean_f1'
    return pd.DataFrame(rows).sort_values([objective, secondary], ascending=False)


def evaluate_model(
    name: str,
    root: Path,
    args: argparse.Namespace,
    num_features: list[str],
    cat_features: list[str],
    full_growth_truth: dict[str, FullGrowthTruth] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = num_features + cat_features
    source_radius = args.source_candidate_radius
    train_path = candidate_path(root, 'train', args.connectivity, source_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)
    val_path = candidate_path(root, 'val', args.connectivity, source_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)
    test_path = candidate_path(root, 'test', args.connectivity, source_radius, args.min_component_pixels, args.history_days, args.allow_partial_history, args.goes_variant)

    X_train, y_train, df_train = load_split(
        train_path,
        features,
        sample=args.train_sample,
        include_keys=args.target_scope == 'local',
        max_candidate_distance=args.candidate_radius,
        read_chunksize=args.read_chunksize,
    )
    if not len(X_train):
        raise ValueError(
            f'No train candidate rows remain at radius={args.candidate_radius}; '
            f'source radius={source_radius}'
        )
    if args.target_scope == 'local':
        if full_growth_truth is None or 'train' not in full_growth_truth:
            raise ValueError('Local target scope requires train/val/test full growth truth')
        y_train = replace_target_with_local_growth(df_train, full_growth_truth['train'])

    model = build_model(num_features, cat_features)
    model.fit(X_train, y_train)
    del X_train, y_train, df_train
    if full_growth_truth is not None:
        FullGrowthTruth.growth_partition.cache_clear()
        FullGrowthTruth.event_masks.cache_clear()
    gc.collect()

    def predict_evaluation_split(
        path: Path,
        split: str,
    ) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
        truth = full_growth_truth.get(split) if full_growth_truth else None
        if args.read_chunksize is not None:
            return predict_split_chunked(
                path,
                model,
                features,
                args.candidate_radius,
                args.read_chunksize,
                args.target_scope,
                truth,
            )
        X, y, frame = load_split(
            path,
            features,
            include_keys=True,
            max_candidate_distance=args.candidate_radius,
        )
        if args.target_scope == 'local':
            y = replace_target_with_local_growth(frame, truth)
        probability = model.predict_proba(X)[:, 1].astype(np.float32)
        del X
        gc.collect()
        return y, frame, probability

    y_val, df_val, prob_val = predict_evaluation_split(val_path, 'val')
    val_pr_auc = average_precision_score(y_val, prob_val)
    val_roc_auc = roc_auc_score(y_val, prob_val)
    print(f'\n=== {name} global ===')
    print('val PR-AUC', val_pr_auc, 'ROC-AUC', val_roc_auc)

    val_metrics = None
    if args.fixed_threshold is None:
        thresholds = [float(x) for x in np.linspace(0.05, 0.95, 19)]
        top_fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        tuning = tune_thresholds(
            df_val,
            prob_val,
            thresholds,
            top_fracs,
            args.selection_objective,
            full_growth_truth.get('val') if full_growth_truth else None,
            args.target_scope,
        )
        tuning['model'] = name
        best = tuning.iloc[0]
        method = str(best['method'])
        value = float(best['value'])
        print(f"best val: method={method} value={value} mean_iou={best['mean_iou']:.6f} mean_f1={best['mean_f1']:.6f}")
    else:
        method = 'threshold'
        value = float(args.fixed_threshold)
        fixed_metrics = date_mask_metrics(
            df_val,
            prob_val,
            method,
            value,
            full_growth_truth.get('val') if full_growth_truth else None,
            args.target_scope,
        )
        fixed_fire = firewise_metrics(fixed_metrics, name, 'val', method, value)
        fixed_summary = summarize(fixed_metrics)
        val_metrics = fixed_metrics
        tuning = pd.DataFrame([{
            'method': method,
            'value': value,
            **fixed_summary,
            **summarize_firewise(fixed_fire),
            'model': name,
        }])
        print(
            f"fixed threshold: value={value} "
            f"mean_iou={fixed_summary['mean_iou']:.6f} "
            f"mean_f1={fixed_summary['mean_f1']:.6f}"
        )
    if val_metrics is None:
        val_metrics = date_mask_metrics(
            df_val,
            prob_val,
            method,
            value,
            full_growth_truth.get('val') if full_growth_truth else None,
            args.target_scope,
        )
    val_fire = firewise_metrics(val_metrics, name, 'val', method, value)
    val_summary_values = {
        'model': name,
        'split': 'val',
        'selected_method': method,
        'selected_value': value,
        'candidate_radius': args.candidate_radius,
        'source_candidate_radius': source_radius,
        'target_scope': args.target_scope,
        'pr_auc': val_pr_auc,
        'roc_auc': val_roc_auc,
        **summarize(val_metrics),
        **summarize_firewise(val_fire),
    }
    del y_val, df_val, prob_val, val_metrics
    if full_growth_truth is not None:
        FullGrowthTruth.growth_partition.cache_clear()
        FullGrowthTruth.event_masks.cache_clear()
    gc.collect()

    y_test, df_test, prob_test = predict_evaluation_split(test_path, 'test')
    test_pr_auc = average_precision_score(y_test, prob_test)
    test_roc_auc = roc_auc_score(y_test, prob_test)
    print('test PR-AUC', test_pr_auc, 'ROC-AUC', test_roc_auc)
    test_metrics = date_mask_metrics(
        df_test,
        prob_test,
        method,
        value,
        full_growth_truth.get('test') if full_growth_truth else None,
        args.target_scope,
    )

    test_fire = firewise_metrics(test_metrics, name, 'test', method, value)
    fire_metrics = pd.concat([val_fire, test_fire], ignore_index=True)
    fire_metrics['candidate_radius'] = args.candidate_radius
    fire_metrics['source_candidate_radius'] = source_radius
    fire_metrics['target_scope'] = args.target_scope
    if full_growth_truth is not None:
        fire_metrics['local_spread_radius'] = args.local_spread_radius

    val_summary = pd.DataFrame([val_summary_values])
    test_summary = pd.DataFrame([{
        'model': name,
        'split': 'test',
        'selected_method': method,
        'selected_value': value,
        'candidate_radius': args.candidate_radius,
        'source_candidate_radius': source_radius,
        'target_scope': args.target_scope,
        'pr_auc': test_pr_auc,
        'roc_auc': test_roc_auc,
        **summarize(test_metrics),
        **summarize_firewise(test_fire),
    }])
    summary = pd.concat([val_summary, test_summary], ignore_index=True)
    tuning['candidate_radius'] = args.candidate_radius
    tuning['source_candidate_radius'] = source_radius
    tuning['target_scope'] = args.target_scope
    if full_growth_truth is not None:
        tuning['local_spread_radius'] = args.local_spread_radius
        summary['local_spread_radius'] = args.local_spread_radius
    return tuning, summary, fire_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Reconstruct 256x256 masks from event candidate probabilities and evaluate IoU/F1.')
    parser.add_argument('--candidate-root', type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument('--connectivity', type=int, default=8)
    parser.add_argument('--candidate-radius', type=float, default=5.0)
    parser.add_argument(
        '--source-candidate-radius',
        type=float,
        default=None,
        help=(
            'Radius encoded in the input CSV. When larger than --candidate-radius, '
            'reuse that file and filter rows by distance_px instead of rebuilding it.'
        ),
    )
    parser.add_argument('--min-component-pixels', type=int, default=1)
    parser.add_argument(
        '--target-scope',
        choices=['all', 'local'],
        default='all',
        help='Train against all next-day growth or only local spread components.',
    )
    parser.add_argument('--train-sample', type=int, default=1_000_000)
    parser.add_argument(
        '--read-chunksize',
        type=int,
        default=200_000,
        help='CSV rows read per chunk before radius filtering; use 0 to disable chunked reads.',
    )
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
        choices=[
            'goes_frp',
            'goes_frp_motion',
            'goes_frp_motion_firepred',
            'goes_frp_motion_recent',
            'goes_frp_motion_recent_firepred',
        ],
        default='goes_frp',
        help='GOES-enriched candidate file suffix to evaluate.',
    )
    parser.add_argument(
        '--selection-objective',
        choices=[
            'mean_iou', 'mean_f1', 'micro_iou', 'micro_f1',
            'fire_mean_iou', 'fire_mean_f1', 'fire_micro_iou', 'fire_micro_f1',
            'full_mean_iou', 'full_mean_f1', 'full_micro_iou', 'full_micro_f1',
            'fire_full_mean_iou', 'fire_full_mean_f1',
            'fire_full_micro_iou', 'fire_full_micro_f1',
            'local_positive_mean_iou', 'local_positive_mean_f1',
            'local_micro_iou', 'local_micro_f1',
            'fire_local_positive_mean_iou', 'fire_local_positive_mean_f1',
            'fire_local_micro_iou', 'fire_local_micro_f1',
        ],
        default='fire_mean_iou',
        help='Validation metric used to select threshold/top-fraction mask reconstruction.',
    )
    parser.add_argument(
        '--fixed-threshold',
        type=float,
        default=None,
        help='Use one fixed probability threshold for every model instead of validation selection.',
    )
    parser.add_argument(
        '--full-growth-metrics',
        action='store_true',
        help='Load raw VIIRS masks and report full next-day growth metrics plus candidate coverage.',
    )
    parser.add_argument(
        '--local-spread-radius',
        type=float,
        default=5.0,
        help='Maximum current-perimeter distance in VIIRS-grid pixels for a new component to count as local spread.',
    )
    args = parser.parse_args()

    if args.source_candidate_radius is None:
        args.source_candidate_radius = args.candidate_radius

    if args.history_days < 1:
        raise ValueError('--history-days must be >= 1')
    if args.read_chunksize < 0:
        raise ValueError('--read-chunksize must be >= 0')
    if args.read_chunksize == 0:
        args.read_chunksize = None
    if args.candidate_radius <= 0:
        raise ValueError('--candidate-radius must be > 0')
    if args.source_candidate_radius < args.candidate_radius:
        raise ValueError('--source-candidate-radius must be >= --candidate-radius')
    if args.fixed_threshold is not None and not 0.0 <= args.fixed_threshold <= 1.0:
        raise ValueError('--fixed-threshold must be between 0 and 1')
    if args.local_spread_radius < 0:
        raise ValueError('--local-spread-radius must be >= 0')
    if (
        'full_' in args.selection_objective or 'local_' in args.selection_objective
    ) and not args.full_growth_metrics:
        raise ValueError('Full/local growth selection objectives require --full-growth-metrics')
    if args.target_scope == 'local' and not args.full_growth_metrics:
        raise ValueError('--target-scope local requires --full-growth-metrics')
    root = args.candidate_root
    outputs = []
    tunings = []
    fire_outputs = []
    full_growth_truth = None
    if args.full_growth_metrics:
        full_growth_truth = {
            'val': FullGrowthTruth('val', args.local_spread_radius),
            'test': FullGrowthTruth('test', args.local_spread_radius),
        }
        if args.target_scope == 'local':
            full_growth_truth['train'] = FullGrowthTruth('train', args.local_spread_radius)

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
        'geometry_plus_firepred': (
            geom_num + FIREPRED_NUM,
            GEOMETRY_CAT + FIREPRED_CAT,
        ),
        'geometry_plus_goes_frp_firepred': (
            geom_num + goes_num + FIREPRED_NUM,
            GEOMETRY_CAT + FIREPRED_CAT,
        ),
        'geometry_plus_goes_frp_motion_firepred': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + FIREPRED_NUM,
            GEOMETRY_CAT + FIREPRED_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION,
            GEOMETRY_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent_observed_wind': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION + FIREPRED_OBSERVED_WIND,
            GEOMETRY_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent_forecast_wind': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION + FIREPRED_FORECAST_WIND,
            GEOMETRY_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent_fuel_weather': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION + FIREPRED_FUEL_WEATHER,
            GEOMETRY_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent_terrain': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION + FIREPRED_TERRAIN,
            GEOMETRY_CAT + FIREPRED_CAT,
        ),
        'geometry_plus_goes_frp_motion_recent_firepred': (
            geom_num + goes_num + GOES_SUBDAILY_MOTION + GOES_RECENT_MOTION + FIREPRED_NUM,
            GEOMETRY_CAT + FIREPRED_CAT,
        ),
    }

    for feature_set in parse_feature_sets(args.feature_sets):
        num_features, cat_features = feature_specs[feature_set]
        tuning, summary, fire_metrics = evaluate_model(
            feature_set,
            root,
            args,
            num_features,
            cat_features,
            full_growth_truth,
        )
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
    if args.fixed_threshold is not None:
        threshold_tag = str(args.fixed_threshold).replace('.', 'p')
        output_tag += f'_thr{threshold_tag}'
    if args.full_growth_metrics:
        radius_tag = str(args.local_spread_radius).replace('.', 'p')
        output_tag += f'_localr{radius_tag}_fullgrowth'
    if args.target_scope == 'local':
        output_tag += '_targetlocal'
    if args.source_candidate_radius != 5.0 or args.candidate_radius != 5.0:
        source_tag = str(args.source_candidate_radius).replace('.', 'p')
        candidate_tag = str(args.candidate_radius).replace('.', 'p')
        output_tag += f'_srcR{source_tag}_candR{candidate_tag}'
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
