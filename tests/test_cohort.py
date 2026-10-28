from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimic_t2d.data import is_type_2_diabetes


class Type2CohortTest(unittest.TestCase):
    def test_icd10_and_icd9_type_2_codes_are_selected(self) -> None:
        codes = pd.Series(["E11.9", "E10.9", "250.00", "250.02", "250.01", "250.03", "I10"])
        versions = pd.Series([10, 10, 9, 9, 9, 9, 10])
        selected = is_type_2_diabetes(codes, versions).tolist()
        self.assertEqual(selected, [True, False, True, True, False, False, False])
