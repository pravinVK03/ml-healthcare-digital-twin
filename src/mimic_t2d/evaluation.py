"""Grouped cross-validation and patient-level uncertainty summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .models import model_specifications


def run_grouped_cross_validation(points: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate out-of-fold predictions without putting a patient in train and test."""
    groups = points["subject_id"].to_numpy()
    n_subjects = pd.Series(groups).nunique()
    n_splits = min(int(config["n_folds"]), n_subjects)
    if n_splits < 2:
        raise ValueError("At least two patients with eligible points are required for grouped evaluation.")
    splitter = GroupKFold(n_splits=n_splits)
    specs = model_specifications()
    outputs: list[pd.DataFrame] = []
    split_rows: list[dict] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(points, groups=groups), start=1):
        train = points.iloc[train_index]
        test = points.iloc[test_index]
        train_subjects = set(train["subject_id"])
        test_subjects = set(test["subject_id"])
        if train_subjects & test_subjects:
            raise AssertionError("Patient leakage detected in GroupKFold split.")
        for model_name, (features, estimator) in specs.items():
            if model_name == "persistence":
                prediction = test["current_glucose"].to_numpy(dtype=float)
            else:
                fitted = clone(estimator).fit(train[features], train["target_glucose"])
                prediction = fitted.predict(test[features])
            prediction = np.clip(prediction, config["min_glucose_mg_dl"], config["max_glucose_mg_dl"])
            output = test[["subject_id", "hadm_id", "timestamp", "target_timestamp", "current_glucose", "target_glucose", "outcome_gap_hours"]].copy()
            output["fold"] = fold
            output["model"] = model_name
            output["prediction"] = prediction
            output["absolute_error"] = np.abs(output["prediction"] - output["target_glucose"])
            output["squared_error"] = (output["prediction"] - output["target_glucose"]) ** 2
            outputs.append(output)
        split_rows.append({"fold": fold, "train_subjects": len(train_subjects), "test_subjects": len(test_subjects), "train_points": len(train), "test_points": len(test), "subject_overlap": 0})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(split_rows)


def _bootstrap_interval(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise_predictions(predictions: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate patient-macro metrics and paired changes from persistence."""
    errors = predictions.assign(error=lambda d: d["prediction"] - d["target_glucose"])
    patient_metrics = (
        errors.groupby(["model", "subject_id"], as_index=False)
        .agg(
            n_predictions=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            bias=("error", "mean"),
        )
    )
    rng = np.random.default_rng(int(config["random_seed"]))
    rows: list[dict] = []
    for model, frame in patient_metrics.groupby("model", sort=True):
        for metric in ("mae", "rmse", "bias"):
            values = frame[metric].to_numpy(dtype=float)
            low, high = _bootstrap_interval(values, int(config["bootstrap_iterations"]), rng)
            rows.append({"model": model, "aggregation": "subject_macro", "metric": metric, "estimate": float(values.mean()), "ci_2_5": low, "ci_97_5": high, "n_subjects": int(len(values)), "n_predictions": int(frame["n_predictions"].sum())})
        pooled = predictions.loc[predictions["model"].eq(model)]
        rows.append({"model": model, "aggregation": "pooled", "metric": "mae", "estimate": float(pooled["absolute_error"].mean()), "ci_2_5": np.nan, "ci_97_5": np.nan, "n_subjects": int(frame.shape[0]), "n_predictions": int(len(pooled))})
    metrics = pd.DataFrame(rows)

    pivot = patient_metrics.pivot(index="subject_id", columns="model", values="mae")
    comparison_rows: list[dict] = []
    if "persistence" in pivot:
        baseline = pivot["persistence"]
        for model in pivot.columns:
            if model == "persistence":
                continue
            delta = (pivot[model] - baseline).dropna().to_numpy(dtype=float)
            low, high = _bootstrap_interval(delta, int(config["bootstrap_iterations"]), rng)
            comparison_rows.append({"comparison": f"{model} minus persistence", "metric": "subject_macro_mae_difference", "estimate": float(delta.mean()), "ci_2_5": low, "ci_97_5": high, "probability_improved": float(np.mean(delta < 0)), "n_subjects": int(len(delta))})
    return metrics, patient_metrics, pd.DataFrame(comparison_rows)
