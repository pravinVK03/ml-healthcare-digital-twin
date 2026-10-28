"""End-to-end reproducible study runner and evidence artifact writer."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_glucose_measurements, load_hba1c_measurements, load_t2d_cohort
from .evaluation import run_grouped_cross_validation, summarise_predictions
from .features import build_prediction_points


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def build_feasibility_audit(patients: pd.DataFrame, admissions: pd.DataFrame, t2d_diagnoses: pd.DataFrame, glucose: pd.DataFrame, a1c: pd.DataFrame, points: pd.DataFrame, config: dict) -> dict:
    a1c = a1c.copy()
    a1c["next_time"] = a1c.groupby("subject_id")["charttime"].shift(-1)
    a1c["next_hba1c"] = a1c.groupby("subject_id")["valuenum"].shift(-1)
    a1c["follow_up_days"] = (a1c["next_time"] - a1c["charttime"]).dt.total_seconds() / 86400
    eligible = a1c.loc[a1c["follow_up_days"].between(config["hba1c_minimum_gap_days"], config["hba1c_maximum_gap_days"])]
    eligible_subjects = int(eligible["subject_id"].nunique())
    gate_passes = eligible_subjects >= int(config["hba1c_minimum_subjects_for_primary_study"])
    return {
        "cohort_definition": "At least one inpatient ICD-10 E11* or ICD-9 250.x0/250.x2 diagnosis.",
        "demo_only": True,
        "t2d_subjects": int(patients["subject_id"].nunique()),
        "t2d_admissions": int(admissions["hadm_id"].nunique()),
        "t2d_diagnosis_records": int(len(t2d_diagnoses)),
        "valid_glucose_measurements": int(len(glucose)),
        "subjects_with_glucose": int(glucose["subject_id"].nunique()),
        "eligible_glucose_prediction_points": int(len(points)),
        "subjects_with_prediction_points": int(points["subject_id"].nunique()),
        "hba1c_measurements": int(len(a1c)),
        "subjects_with_hba1c": int(a1c["subject_id"].nunique()),
        "hba1c_eligible_pairs": int(len(eligible)),
        "hba1c_eligible_subjects": eligible_subjects,
        "hba1c_primary_study_minimum_subjects": int(config["hba1c_minimum_subjects_for_primary_study"]),
        "hba1c_primary_study_gate": "PASS" if gate_passes else "NO_GO",
        "hba1c_gate_reason": "Adequate repeated HbA1c follow-up is present." if gate_passes else "The demo has too few patients with eligible repeated HbA1c measurements for the proposed primary endpoint.",
        "exploratory_endpoint": "Next in-hospital laboratory glucose measurement, 1 to 24 hours after the prediction time, within the same admission.",
        "key_limitations": [
            "MIMIC-IV demo is a small, de-identified sample and is not a paper-scale cohort.",
            "Glucose laboratory values are irregularly sampled and are not continuous glucose monitoring data.",
            "This analysis predicts observed care; it does not estimate medication effects or prescribe treatment.",
            "Patient-held-out cross-validation is internal validation only and is not external validation."
        ]
    }


def _write_summary(path: Path, audit: dict, metrics: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    primary = metrics.loc[(metrics["aggregation"] == "subject_macro") & (metrics["metric"] == "mae")].sort_values("estimate")
    lines = [
        "# MIMIC-IV Demo T2D Patient State Study",
        "",
        "This automatically generated report describes an exploratory, retrospective next-glucose prediction experiment. It is not a clinical decision-support result.",
        "",
        "## Feasibility decision",
        "",
        f"The next-HbA1c primary endpoint gate is **{audit['hba1c_primary_study_gate']}**: {audit['hba1c_eligible_subjects']} eligible patients are available, versus the pre-specified minimum of {audit['hba1c_primary_study_minimum_subjects']}.",
        "",
        "## Exploratory acute-care endpoint",
        "",
        f"Cohort: {audit['t2d_subjects']} T2D-coded patients, {audit['valid_glucose_measurements']} valid glucose measurements, and {audit['eligible_glucose_prediction_points']} eligible prediction points from {audit['subjects_with_prediction_points']} patients.",
        "",
        "Outcome: next in-hospital laboratory glucose measured 1-24 hours after the prediction time. Evaluation uses patient-held-out folds, so no patient contributes records to both training and test data in a fold.",
        "",
        "## Subject-macro MAE",
        "",
        "| Model | MAE mg/dL | 95% CI | Patients |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.estimate:.2f} | {row.ci_2_5:.2f} to {row.ci_97_5:.2f} | {row.n_subjects} |")
    if not comparisons.empty:
        lines.extend(["", "## Paired patient-level comparison with persistence", "", "| Comparison | Mean MAE difference mg/dL | 95% CI |", "| --- | ---: | --- |"])
        for row in comparisons.itertuples(index=False):
            lines.append(f"| {row.comparison} | {row.estimate:.2f} | {row.ci_2_5:.2f} to {row.ci_97_5:.2f} |")
    lines.extend(["", "## Interpretation boundary", "", "The adaptive state representation is a transparent exponentially weighted summary of historical laboratory glucose. A better internal score would support further evaluation of this feature design, not a claim of a clinically validated digital twin."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(config: dict) -> dict:
    """Execute the demo study and write only reproducible, reviewable outputs."""
    output = Path(config["output_path"])
    output.mkdir(parents=True, exist_ok=True)
    patients, admissions, t2d_diagnoses = load_t2d_cohort(Path(config["data_path"]))
    glucose = load_glucose_measurements(Path(config["data_path"]), admissions, config)
    a1c = load_hba1c_measurements(Path(config["data_path"]), patients["subject_id"], config)
    points = build_prediction_points(glucose, patients, config)
    predictions, folds = run_grouped_cross_validation(points, config)
    metrics, patient_metrics, comparisons = summarise_predictions(predictions, config)
    audit = build_feasibility_audit(patients, admissions, t2d_diagnoses, glucose, a1c, points, config)

    points.to_csv(output / "prediction_points.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    patient_metrics.to_csv(output / "patient_metrics.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    folds.to_csv(output / "fold_audit.csv", index=False)
    _write_json(output / "feasibility_audit.json", audit)
    _write_json(output / "study_config.json", {key: value for key, value in config.items() if key not in {"project_root", "data_path", "output_path"}})
    _write_json(output / "run_environment.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__})
    _write_summary(output / "summary.md", audit, metrics, comparisons)
    return {"output_path": output, "audit": audit, "metrics": metrics}
