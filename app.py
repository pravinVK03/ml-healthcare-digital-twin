"""Research dashboard for the leakage-safe MIMIC-IV demo T2D study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_LABELS = {
    "persistence": "Persistence baseline",
    "snapshot_ridge": "Snapshot ridge",
    "history_ridge": "Fixed-history ridge",
    "adaptive_ridge": "Adaptive-state ridge",
    "adaptive_hgb": "Adaptive-state gradient boosting",
}

st.set_page_config(page_title="T2D Patient State Study", page_icon="research", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    date_columns = ["timestamp", "target_timestamp"] if name == "predictions.csv" else None
    return pd.read_csv(RESULTS / name, parse_dates=date_columns)


@st.cache_data(show_spinner=False)
def load_audit() -> dict:
    with (RESULTS / "feasibility_audit.json").open(encoding="utf-8") as handle:
        return json.load(handle)


required = ["feasibility_audit.json", "metrics.csv", "predictions.csv", "patient_metrics.csv", "fold_audit.csv"]
st.title("Type 2 Diabetes Patient State Study")
st.caption("MIMIC-IV demo | exploratory retrospective research | not a medical device or clinical decision-support system")
if not all((RESULTS / name).exists() for name in required):
    st.error("Study outputs are not available. From the project folder, run: python scripts/run_study.py")
    st.stop()

audit = load_audit()
metrics = load_csv("metrics.csv")
predictions = load_csv("predictions.csv")
patient_metrics = load_csv("patient_metrics.csv")
folds = load_csv("fold_audit.csv")

with st.sidebar:
    st.header("Study scope")
    st.write("T2D-coded hospital cohort; next laboratory glucose prediction over 1-24 hours.")
    st.warning("All patient displays use saved, held-out research predictions. They must not guide care.")
    st.divider()
    st.caption("The demo data are small. Re-run against an approved full MIMIC-IV extract before drawing paper-level conclusions.")

state_tab, monitoring_tab, forecast_tab, evidence_tab, safeguards_tab = st.tabs(
    ["Patient state", "Trajectory monitoring", "Held-out forecasting", "Study evidence", "Method safeguards"]
)

subjects = sorted(predictions["subject_id"].astype(str).unique())
with st.sidebar:
    selected_subject = st.selectbox("Research participant", subjects)
    selected_model = st.selectbox("Forecast model", list(MODEL_LABELS), index=4, format_func=MODEL_LABELS.get)

patient = predictions.loc[predictions["subject_id"].astype(str).eq(selected_subject)].copy()
patient_model = patient.loc[patient["model"].eq(selected_model)].sort_values("timestamp")

with state_tab:
    st.subheader("Personalized patient-state representation")
    st.write("This is a transparent representation of the latest observed laboratory glucose trajectory. It is not a physiological simulation and does not contain live patient data.")
    latest = patient_model.iloc[-1]
    recent = patient_model.drop_duplicates("timestamp", keep="last").tail(8)
    change = float(recent["current_glucose"].iloc[-1] - recent["current_glucose"].iloc[0]) if len(recent) > 1 else 0.0
    one, two, three, four = st.columns(4)
    one.metric("Latest recorded glucose", f"{latest['current_glucose']:.0f} mg/dL")
    two.metric("Change over displayed history", f"{change:+.0f} mg/dL")
    three.metric("Next measurement gap", f"{float(latest['outcome_gap_hours']):.1f} hours")
    four.metric("Held-out fold", f"{int(latest['fold'])}")
    st.caption("The adaptive-state models internally summarize recent values with time-decay. Exact feature values are available in `results/prediction_points.csv` for audit.")

with monitoring_tab:
    st.subheader("Disease trajectory monitoring")
    st.write("Each point is an observed laboratory glucose value at a prediction time. Sampling is irregular; gaps are clinical measurement intervals, not missing CGM samples.")
    history = patient_model.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    st.line_chart(history[["current_glucose"]], y_label="Observed glucose (mg/dL)", height=330)
    display = history.reset_index()[["timestamp", "current_glucose", "outcome_gap_hours"]].tail(20)
    display.columns = ["Observation time", "Observed glucose mg/dL", "Time to next lab hours"]
    st.dataframe(display.round({"Observed glucose mg/dL": 1, "Time to next lab hours": 2}), hide_index=True, width="stretch")

with forecast_tab:
    st.subheader("Held-out next-glucose forecasting")
    st.write("These forecasts were made by models trained without this participant in the relevant evaluation fold. The later laboratory value is displayed only to evaluate saved predictions.")
    origins = patient_model["timestamp"].drop_duplicates().sort_values().tolist()
    chosen_time = st.select_slider("Prediction time", options=origins, value=origins[-1], format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y %H:%M"))
    selected = patient_model.loc[patient_model["timestamp"].eq(chosen_time)].iloc[0]
    prior = patient_model.loc[patient_model["timestamp"].le(chosen_time)].drop_duplicates("timestamp", keep="last").tail(12)
    chart = prior[["timestamp", "current_glucose"]].rename(columns={"timestamp": "time", "current_glucose": "Observed history"})
    target = pd.DataFrame({"time": [selected["target_timestamp"]], "Held-out forecast": [selected["prediction"]], "Later observed glucose": [selected["target_glucose"]]})
    chart = chart.merge(target, on="time", how="outer").sort_values("time").set_index("time")
    st.line_chart(chart, y_label="Glucose (mg/dL)", height=340)
    left, middle, right = st.columns(3)
    left.metric("Forecast", f"{selected['prediction']:.1f} mg/dL")
    middle.metric("Later observed", f"{selected['target_glucose']:.1f} mg/dL")
    right.metric("Absolute error", f"{selected['absolute_error']:.1f} mg/dL")

with evidence_tab:
    st.subheader("Cohort and model evidence")
    a, b, c, d = st.columns(4)
    a.metric("T2D-coded patients", audit["t2d_subjects"])
    b.metric("Glucose measurements", audit["valid_glucose_measurements"])
    c.metric("Eligible prediction points", audit["eligible_glucose_prediction_points"])
    d.metric("Patients evaluated", audit["subjects_with_prediction_points"])
    st.markdown(f"**Next-HbA1c primary study gate: {audit['hba1c_primary_study_gate']}.** {audit['hba1c_gate_reason']}")
    mae = metrics.loc[(metrics["aggregation"] == "subject_macro") & (metrics["metric"] == "mae")].copy()
    mae["Model"] = mae["model"].map(MODEL_LABELS)
    mae = mae.sort_values("estimate")
    figure, axis = plt.subplots(figsize=(8, 4.2))
    errors = [mae["estimate"] - mae["ci_2_5"], mae["ci_97_5"] - mae["estimate"]]
    axis.barh(mae["Model"], mae["estimate"], xerr=errors, color="#2463a6", capsize=4)
    axis.set_xlabel("Subject-macro MAE (mg/dL), lower is better")
    axis.set_title("Patient-held-out next-glucose performance")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    table = mae[["Model", "estimate", "ci_2_5", "ci_97_5", "n_subjects", "n_predictions"]].rename(columns={"estimate": "MAE", "ci_2_5": "95% CI lower", "ci_97_5": "95% CI upper", "n_subjects": "Patients", "n_predictions": "Predictions"})
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    st.caption(f"Fold audit: {len(folds)} patient-held-out folds; recorded subject overlap is {int(folds['subject_overlap'].sum())}.")

with safeguards_tab:
    st.subheader("Interpretation and safeguards")
    st.markdown("""
    - **No future leakage:** each feature uses the index glucose value and earlier observations only; the later glucose is assigned after feature construction.
    - **Patient-held-out evaluation:** a patient's records are never shared between training and test within a fold.
    - **Fair comparison:** persistence, snapshot, fixed-history, and adaptive-state models receive the identical eligible prediction points.
    - **Uncertainty:** confidence intervals resample patients, rather than treating repeated measurements from one patient as independent people.
    - **Bounded claims:** this predicts a later observed laboratory measurement under recorded care. It does not infer medication effects, recommend therapy, or constitute a validated digital twin.
    """)
    st.write("Key limitations from the feasibility audit:")
    for limitation in audit["key_limitations"]:
        st.write(f"- {limitation}")
