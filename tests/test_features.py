from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimic_t2d.features import build_prediction_points


class FeatureLeakageTest(unittest.TestCase):
    def test_future_values_do_not_change_index_features(self) -> None:
        base = pd.Timestamp("2100-01-01 00:00:00")
        glucose = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "hadm_id": [10, 10, 10, 10],
                "charttime": [base, base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=4), base + pd.Timedelta(hours=6)],
                "glucose_mg_dl": [100.0, 120.0, 140.0, 160.0],
                "admittime": [base - pd.Timedelta(hours=1)] * 4,
                "dischtime": [base + pd.Timedelta(hours=12)] * 4,
            }
        )
        patients = pd.DataFrame({"subject_id": [1], "gender": ["M"], "anchor_age": [60]})
        config = {"history_window_hours": 48, "minimum_history_measurements": 2, "minimum_outcome_gap_hours": 1, "maximum_outcome_gap_hours": 24, "adaptive_state_half_life_hours": 12}
        original = build_prediction_points(glucose, patients, config)
        changed = glucose.copy()
        changed.loc[3, "glucose_mg_dl"] = 999.0
        modified = build_prediction_points(changed, patients, config)
        before_last = original.loc[original["timestamp"] < base + pd.Timedelta(hours=6)].reset_index(drop=True)
        after_last = modified.loc[modified["timestamp"] < base + pd.Timedelta(hours=6)].reset_index(drop=True)
        columns = ["current_glucose", "history_mean", "history_std", "state_mean", "state_std", "state_trend"]
        pd.testing.assert_frame_equal(before_last[columns], after_last[columns])
