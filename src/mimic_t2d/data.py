"""MIMIC-IV demo cohort extraction with explicit T2D ascertainment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalise_icd_codes(codes: pd.Series) -> pd.Series:
    return codes.fillna("").astype(str).str.upper().str.replace(".", "", regex=False)


def is_type_2_diabetes(icd_code: pd.Series, icd_version: pd.Series) -> pd.Series:
    """Return the documented ICD-9/10 code families used for the T2D cohort."""
    code = normalise_icd_codes(icd_code)
    version = pd.to_numeric(icd_version, errors="coerce")
    icd10 = version.eq(10) & code.str.startswith("E11")
    # ICD-9 fifth digits 0 and 2 denote type II/unspecified diabetes.
    icd9 = (
        version.eq(9)
        & code.str.startswith("250")
        & code.str.len().ge(5)
        & code.str[-1].isin(["0", "2"])
    )
    return icd10 | icd9


def load_t2d_cohort(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return patients, T2D admissions, and qualifying diagnosis records."""
    diagnoses = pd.read_csv(data_path / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    diagnoses["is_t2d"] = is_type_2_diabetes(diagnoses["icd_code"], diagnoses["icd_version"])
    t2d_diagnoses = diagnoses.loc[diagnoses["is_t2d"]].copy()
    t2d_admission_ids = t2d_diagnoses["hadm_id"].dropna().unique()

    admissions = pd.read_csv(data_path / "core" / "admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    admissions = admissions.loc[admissions["hadm_id"].isin(t2d_admission_ids)].copy()
    patients = pd.read_csv(data_path / "core" / "patients.csv.gz")
    patients = patients.loc[patients["subject_id"].isin(t2d_diagnoses["subject_id"].unique())].copy()
    return patients, admissions, t2d_diagnoses


def load_glucose_measurements(data_path: Path, cohort_admissions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Load plausible, time-stamped blood glucose labs and aggregate duplicate timestamps.

    The outcome is intentionally a laboratory glucose measurement, not a CGM value.
    This prevents the dashboard from implying that MIMIC-IV provides continuous monitoring.
    """
    columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    labs = pd.read_csv(data_path / "hosp" / "labevents.csv.gz", usecols=columns, parse_dates=["charttime"])
    allowed_admissions = cohort_admissions[["subject_id", "hadm_id", "admittime", "dischtime"]]
    glucose = labs.loc[
        labs["itemid"].isin(config["glucose_lab_itemids"])
        & labs["valuenum"].between(config["min_glucose_mg_dl"], config["max_glucose_mg_dl"])
    ].copy()
    glucose = glucose.merge(allowed_admissions, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    glucose = glucose.loc[glucose["charttime"].notna()].copy()
    glucose = (
        glucose.groupby(["subject_id", "hadm_id", "charttime"], as_index=False)
        .agg(glucose_mg_dl=("valuenum", "median"), source_count=("itemid", "size"), admittime=("admittime", "first"), dischtime=("dischtime", "first"))
        .sort_values(["subject_id", "hadm_id", "charttime"])
        .reset_index(drop=True)
    )
    return glucose


def load_hba1c_measurements(data_path: Path, t2d_subject_ids: pd.Series, config: dict) -> pd.DataFrame:
    """Load numeric HbA1c values for the protocol feasibility gate only."""
    columns = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    labs = pd.read_csv(data_path / "hosp" / "labevents.csv.gz", usecols=columns, parse_dates=["charttime"])
    a1c = labs.loc[
        labs["subject_id"].isin(t2d_subject_ids)
        & labs["itemid"].isin(config["hba1c_lab_itemids"])
        & labs["valuenum"].between(3.0, 20.0)
    ].copy()
    return a1c.sort_values(["subject_id", "charttime"]).reset_index(drop=True)
