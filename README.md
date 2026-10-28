# MIMIC IV Demo T2D Patient State Study

This project turns the bundled MIMIC-IV demo CSV files into a reproducible, research-only prototype for personalized type-2 diabetes state monitoring and next-glucose forecasting. You do not need to operate a database: the study runner reads the compressed CSV files directly.

## What this project can and cannot claim

The code first tests the proposal's preferred next-HbA1c endpoint. The demo cohort has too few people with repeated HbA1c measurements to support it, so the runner records a **NO_GO** decision rather than inventing a paper result.

It then runs a separate, exploratory acute-care study: predict the **next in-hospital laboratory glucose** 1-24 hours after each eligible observation. This is not continuous glucose monitoring, medication-effect estimation, clinical decision support, or a validated digital twin. The demo validates the pipeline; full MIMIC-IV or another approved longitudinal cohort is required for a paper-scale study.

## Run it

From this project folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python scripts/run_study.py
python -m streamlit run app.py --server.port 8503
```

Open `http://localhost:8503` if the browser does not open automatically.

## What the dashboard shows

1. **Patient state** - latest observed glucose context for a selected research participant.
2. **Trajectory monitoring** - irregularly sampled historical laboratory glucose values.
3. **Held-out forecasting** - saved predictions from a fold in which that participant was excluded from model training.
4. **Study evidence** - cohort counts, feasibility gate, patient-macro MAE, confidence intervals, and fold audit.
5. **Method safeguards** - leakage, validation, uncertainty, and interpretation boundaries.

## Study design

| Component | Implementation |
| --- | --- |
| T2D cohort | Any qualifying inpatient ICD-10 `E11*` or ICD-9 `250.x0`/`250.x2` diagnosis. |
| Endpoint | Next valid blood glucose laboratory result in the same admission, 1-24 hours later. |
| State inputs | Current glucose, demographics, 48-hour fixed-history summaries, or explicitly time-decayed adaptive-state summaries. |
| Models | Persistence; snapshot ridge; fixed-history ridge; adaptive-state ridge; adaptive-state gradient boosting. |
| Evaluation | Five-fold grouped cross-validation, holding out whole patients. |
| Primary reported metric | Subject-macro MAE (each evaluated patient contributes equally), with patient-bootstrap 95% confidence intervals. |
| Reliability checks | Fold-overlap audit, saved out-of-fold predictions, fixed configuration, and unit tests for cohort logic and future-information leakage. |

The “adaptive state” is an honest feature representation: exponentially time-decayed mean, variability, trend, effective observation count, and state age. It is deliberately not described as a novel physiological simulator.

## Outputs

Running the study replaces `results/` with reproducible evidence artifacts:

```text
results/
  feasibility_audit.json    Cohort counts and HbA1c go/no-go decision
  prediction_points.csv     Features and outcomes for every eligible index time
  predictions.csv           Out-of-fold predictions used by the UI
  metrics.csv               Subject-macro and pooled accuracy summaries
  patient_metrics.csv       Per-patient MAE/RMSE/bias
  model_comparisons.csv     Paired patient-level comparisons with persistence
  fold_audit.csv            Verification of no patient overlap by fold
  summary.md                Automatically generated research summary
  study_config.json         Frozen study settings
  run_environment.json      Software/runtime record
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Scale to full MIMIC-IV safely

Use the same MIMIC-IV folder layout and update only `configs/study.json` if a protocol change is approved before evaluation. Before making conference-paper claims, pre-specify the endpoint and eligibility rules, rerun the feasibility audit, use a larger cohort, perform temporal and external validation when feasible, assess subgroup performance, and complete TRIPOD+AI reporting.
