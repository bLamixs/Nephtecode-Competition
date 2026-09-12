"""
Метрики качества моделей и валидация.
"""
import numpy as np
from typing import Dict

def calc_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "count": 0}
    
    diff = y_p - y_t
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "count": int(len(y_t))
    }
