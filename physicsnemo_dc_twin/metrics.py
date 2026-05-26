import json
from pathlib import Path

import numpy as np


def _valid_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    mask = valid > 0.5
    return values[mask]


def _max_location(values: np.ndarray, valid: np.ndarray, x, y, z) -> dict:
    masked = np.where(valid > 0.5, values, np.nan)
    flat_idx = int(np.nanargmax(masked))
    i, j, k = np.unravel_index(flat_idx, masked.shape)
    return {
        "index": [int(i), int(j), int(k)],
        "xyz": [float(x[i]), float(y[j]), float(z[k])],
        "value_C": float(masked[i, j, k]),
    }


def compute_validation_metrics(true_c, pred_c, valid, x, y, z) -> dict:
    true_valid = _valid_values(true_c, valid)
    pred_valid = _valid_values(pred_c, valid)
    err = pred_valid - true_valid
    abs_err = np.abs(err)

    true_hotspot = _max_location(true_c, valid, x, y, z)
    pred_hotspot = _max_location(pred_c, valid, x, y, z)
    hotspot_delta = np.array(pred_hotspot["xyz"]) - np.array(true_hotspot["xyz"])

    return {
        "valid_point_count": int(true_valid.size),
        "mae_C": float(np.mean(abs_err)),
        "rmse_C": float(np.sqrt(np.mean(err**2))),
        "bias_C": float(np.mean(err)),
        "max_abs_error_C": float(np.max(abs_err)),
        "p95_abs_error_C": float(np.percentile(abs_err, 95)),
        "p99_abs_error_C": float(np.percentile(abs_err, 99)),
        "true_temperature": {
            "min_C": float(np.min(true_valid)),
            "mean_C": float(np.mean(true_valid)),
            "max_C": float(np.max(true_valid)),
            "p95_C": float(np.percentile(true_valid, 95)),
            "p99_C": float(np.percentile(true_valid, 99)),
        },
        "pred_temperature": {
            "min_C": float(np.min(pred_valid)),
            "mean_C": float(np.mean(pred_valid)),
            "max_C": float(np.max(pred_valid)),
            "p95_C": float(np.percentile(pred_valid, 95)),
            "p99_C": float(np.percentile(pred_valid, 99)),
        },
        "hotspot": {
            "true": true_hotspot,
            "pred": pred_hotspot,
            "temperature_error_C": float(pred_hotspot["value_C"] - true_hotspot["value_C"]),
            "location_error_m": float(np.linalg.norm(hotspot_delta)),
            "location_delta_xyz_m": [float(v) for v in hotspot_delta],
        },
    }


def save_metrics(metrics: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
