import math
from typing import Dict, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, jaccard_score


def nonconformity_score(prob_fire: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Binary nonconformity score: 1-p for positives, p for negatives."""
    prob = np.asarray(prob_fire, dtype=np.float64)
    y = (np.asarray(y_true) > 0).astype(np.int64)
    return np.where(y == 1, 1.0 - prob, prob)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split conformal quantile with finite-sample correction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("Calibration scores are empty.")

    n = s.size
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(s, q_level, method="higher"))


def prediction_set(
    prob_fire: np.ndarray, qhat: Union[float, Dict[str, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return membership masks for {no-fire}, {fire}, and uncertain={both}."""
    prob = np.asarray(prob_fire, dtype=np.float64)

    if isinstance(qhat, dict):
        q0 = float(qhat["q0"])
        q1 = float(qhat["q1"])
        include_fire = (1.0 - prob) <= q1
        include_no_fire = prob <= q0
    else:
        include_fire = (1.0 - prob) <= qhat
        include_no_fire = prob <= qhat

    uncertain = include_fire & include_no_fire
    return include_no_fire, include_fire, uncertain


def class_conditional_quantiles(prob_fire: np.ndarray, y_true: np.ndarray, alpha: float) -> Dict[str, float]:
    """Class-conditional split conformal quantiles."""
    prob = np.asarray(prob_fire, dtype=np.float64).ravel()
    y = (np.asarray(y_true).ravel() > 0).astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        # Fallback to global quantile if one class is absent.
        q_global = conformal_quantile(nonconformity_score(prob, y), alpha=alpha)
        return {"q0": q_global, "q1": q_global, "n_pos": n_pos, "n_neg": n_neg}

    # y=1 -> score is 1-p ; y=0 -> score is p
    q1 = conformal_quantile(1.0 - prob[y == 1], alpha=alpha)
    q0 = conformal_quantile(prob[y == 0], alpha=alpha)
    return {"q0": float(q0), "q1": float(q1), "n_pos": n_pos, "n_neg": n_neg}


def evaluate_cp(
    prob_fire: np.ndarray,
    y_true: np.ndarray,
    qhat: Union[float, Dict[str, float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute baseline + conformal reliability metrics."""
    prob = np.asarray(prob_fire, dtype=np.float64).ravel()
    y = (np.asarray(y_true).ravel() > 0).astype(np.int64)

    baseline_pred = (prob >= threshold).astype(np.int64)

    include_no_fire, include_fire, uncertain = prediction_set(prob, qhat)
    covered = np.where(y == 1, include_fire, include_no_fire)

    confident_fire = include_fire & (~include_no_fire)
    confident_no_fire = include_no_fire & (~include_fire)
    confident = confident_fire | confident_no_fire
    cp_singleton_pred = confident_fire.astype(np.int64)

    errors = baseline_pred != y
    error_count = int(errors.sum())
    if error_count == 0:
        error_capture_rate = 0.0
    else:
        error_capture_rate = float((uncertain & errors).sum() / error_count)

    confident_count = int(confident.sum())
    if confident_count == 0:
        selective_f1 = float("nan")
        selective_iou = float("nan")
    else:
        selective_f1 = float(
            f1_score(y[confident], cp_singleton_pred[confident], zero_division=1.0)
        )
        selective_iou = float(
            jaccard_score(y[confident], cp_singleton_pred[confident], zero_division=1.0)
        )

    baseline_f1 = float(f1_score(y, baseline_pred, zero_division=1.0))
    baseline_iou = float(jaccard_score(y, baseline_pred, zero_division=1.0))

    out = {
        "qhat": qhat if isinstance(qhat, dict) else float(qhat),
        "coverage": float(np.mean(covered)),
        "uncertainty_rate": float(np.mean(uncertain)),
        "error_capture_rate": float(error_capture_rate),
        "baseline_f1": baseline_f1,
        "baseline_iou": baseline_iou,
        "selective_f1": selective_f1,
        "selective_iou": selective_iou,
        "num_pixels": int(y.size),
        "num_confident_pixels": confident_count,
        "num_uncertain_pixels": int(uncertain.sum()),
        "num_errors_baseline": error_count,
    }
    return out
