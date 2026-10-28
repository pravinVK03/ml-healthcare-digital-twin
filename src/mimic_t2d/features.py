"""Past-only feature construction for irregular glucose trajectories."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


SNAPSHOT_FEATURES = ["current_glucose", "age_at_admission", "sex_male", "hours_from_admission"]
HISTORY_FEATURES = SNAPSHOT_FEATURES + [
    "previous_glucose", "hours_since_previous", "history_count", "history_mean",
    "history_std", "history_min", "history_max", "history_slope",
]
ADAPTIVE_FEATURES = SNAPSHOT_FEATURES + [
    "state_mean", "state_std", "state_trend", "state_effective_count", "state_age_hours",
]


def _slope(hours: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.ptp(hours) == 0:
        return 0.0
    return float(np.polyfit(hours, values, 1)[0])


def build_prediction_points(glucose: pd.DataFrame, patients: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create one next-measurement prediction point using observations available at its index time.

    Every feature is computed before assigning the later glucose outcome. The output
    is deliberately row-oriented so all model families use exactly the same points.
    """
    demographics = patients[["subject_id", "gender", "anchor_age"]].copy()
    demographics["sex_male"] = demographics["gender"].eq("M").astype(int)
    frame = glucose.merge(demographics[["subject_id", "anchor_age", "sex_male"]], on="subject_id", how="left", validate="many_to_one")
    rows: list[dict] = []
    history_hours = float(config["history_window_hours"])
    min_count = int(config["minimum_history_measurements"])
    half_life = float(config["adaptive_state_half_life_hours"])

    for (subject_id, hadm_id), episode in frame.groupby(["subject_id", "hadm_id"], sort=False):
        episode = episode.sort_values("charttime").reset_index(drop=True)
        times = episode["charttime"]
        values = episode["glucose_mg_dl"].to_numpy(dtype=float)
        for index in range(len(episode) - 1):
            timestamp = times.iloc[index]
            outcome_time = times.iloc[index + 1]
            gap_hours = (outcome_time - timestamp).total_seconds() / 3600
            if not (config["minimum_outcome_gap_hours"] <= gap_hours <= config["maximum_outcome_gap_hours"]):
                continue
            prior_mask = (times.iloc[: index + 1] >= timestamp - pd.Timedelta(hours=history_hours))
            prior = episode.iloc[: index + 1].loc[prior_mask]
            if len(prior) < min_count:
                continue

            prior_values = prior["glucose_mg_dl"].to_numpy(dtype=float)
            prior_hours = (prior["charttime"] - timestamp).dt.total_seconds().to_numpy(dtype=float) / 3600
            previous_time = times.iloc[index - 1]
            previous_glucose = float(values[index - 1])
            since_previous = (timestamp - previous_time).total_seconds() / 3600

            # Exponential sufficient statistics are updated from current and earlier values only.
            age = -prior_hours
            weights = np.exp(-math.log(2) * age / half_life)
            weight_sum = weights.sum()
            state_mean = float(np.average(prior_values, weights=weights))
            state_variance = float(np.average((prior_values - state_mean) ** 2, weights=weights))
            state_trend = _slope(prior_hours, prior_values)
            current = float(values[index])
            row = {
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "timestamp": timestamp,
                "target_timestamp": outcome_time,
                "current_glucose": current,
                "target_glucose": float(values[index + 1]),
                "outcome_gap_hours": gap_hours,
                "age_at_admission": float(episode.loc[index, "anchor_age"]),
                "sex_male": int(episode.loc[index, "sex_male"]),
                "hours_from_admission": (timestamp - episode.loc[index, "admittime"]).total_seconds() / 3600,
                "previous_glucose": previous_glucose,
                "hours_since_previous": since_previous,
                "history_count": len(prior_values),
                "history_mean": float(prior_values.mean()),
                "history_std": float(prior_values.std(ddof=0)),
                "history_min": float(prior_values.min()),
                "history_max": float(prior_values.max()),
                "history_slope": _slope(prior_hours, prior_values),
                "state_mean": state_mean,
                "state_std": math.sqrt(max(state_variance, 0.0)),
                "state_trend": state_trend,
                "state_effective_count": float(weight_sum),
                "state_age_hours": float(np.average(age, weights=weights)),
            }
            rows.append(row)
    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("No eligible glucose prediction points were created. Check the configured gaps and history rules.")
    return points.sort_values(["subject_id", "hadm_id", "timestamp"]).reset_index(drop=True)
