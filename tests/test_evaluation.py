from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimic_t2d.evaluation import run_grouped_cross_validation
from mimic_t2d.features import ADAPTIVE_FEATURES, HISTORY_FEATURES, SNAPSHOT_FEATURES


class GroupSplitTest(unittest.TestCase):
    def test_each_model_has_one_out_of_fold_prediction_per_point(self) -> None:
        rows = []
        for subject in range(1, 7):
            for observation in range(3):
                row = {"subject_id": subject, "hadm_id": subject * 10, "timestamp": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=observation), "target_timestamp": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=observation + 2), "current_glucose": 100 + subject + observation, "target_glucose": 105 + subject + observation, "outcome_gap_hours": 2.0}
                for feature in set(SNAPSHOT_FEATURES + HISTORY_FEATURES + ADAPTIVE_FEATURES):
                    row.setdefault(feature, float(subject + observation))
                rows.append(row)
        points = pd.DataFrame(rows)
        predictions, folds = run_grouped_cross_validation(points, {"n_folds": 3, "min_glucose_mg_dl": 20, "max_glucose_mg_dl": 1000})
        self.assertEqual(int(folds["subject_overlap"].sum()), 0)
        counts = predictions.groupby("model").size()
        self.assertTrue((counts == len(points)).all())
        self.assertTrue(np.isfinite(predictions["prediction"]).all())
