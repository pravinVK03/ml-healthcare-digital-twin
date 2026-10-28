"""Run the MIMIC-IV demo T2D patient-state study from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimic_t2d.config import load_config  # noqa: E402
from mimic_t2d.pipeline import run_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leakage-safe MIMIC-IV demo T2D study.")
    parser.add_argument("--config", default=ROOT / "configs" / "study.json", type=Path)
    args = parser.parse_args()
    result = run_study(load_config(args.config))
    audit = result["audit"]
    print(f"Study outputs written to: {result['output_path']}")
    print(f"HbA1c feasibility gate: {audit['hba1c_primary_study_gate']} ({audit['hba1c_eligible_subjects']} eligible patients)")
    print(f"Exploratory next-glucose points: {audit['eligible_glucose_prediction_points']} from {audit['subjects_with_prediction_points']} patients")


if __name__ == "__main__":
    main()
