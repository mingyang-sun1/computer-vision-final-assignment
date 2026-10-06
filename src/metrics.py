"""Evaluation metrics.

Prices are in units of $1000 USD. MSE is the Kaggle metric, so it is the
primary number every experiment is judged on. MAE and R^2 are reported
alongside it because MSE alone hides how the errors are distributed -- a
model can score well on MSE while being badly wrong on expensive houses,
which is exactly what the error analysis in the report needs to catch.

Used by every experiment in the project, so the comparison between
configurations is always apples-to-apples.
"""

from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    """Mean squared error -- the competition metric."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in the same units as the price."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination against the mean predictor.

    R^2 = 0 means the model is no better than always predicting the mean;
    R^2 = 1 is perfect. This makes it directly comparable to the B0 anchor.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def summary(y_true, y_pred) -> dict:
    """All metrics at once, ready to be written to a log file."""
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def format_summary(name: str, values: dict) -> str:
    return f"{name:<24} MSE {values['mse']:>10.1f}   MAE {values['mae']:>7.1f}   R2 {values['r2']:>6.3f}"
