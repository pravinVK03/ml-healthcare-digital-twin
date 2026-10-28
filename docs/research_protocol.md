# Research Protocol for the Demo Implementation

## Research objective

Evaluate whether transparent, time-aware summaries of past laboratory glucose improve prediction of the next observed in-hospital glucose measurement in a T2D-coded cohort, compared with persistence, snapshot, and fixed-history models.

## Two independent sections

### Longitudinal HbA1c feasibility gate

The intended longitudinal endpoint is the next HbA1c, with a 30-730 day follow-up interval. The demo must contain at least 50 distinct patients with an eligible pair before that endpoint may be used as the primary analysis. The runner writes a pass/no-go decision before glucose model results are interpreted.

### Exploratory acute-care glucose study

The implementation uses a valid laboratory glucose at time *t* as the index observation and the next valid laboratory glucose in the same admission as the outcome, when the interval is 1-24 hours. It uses a 48-hour lookback and requires at least two prior/current glucose observations.

## Patient state

The state representation is updated at each observed glucose time. It includes a current measurement and a time-decayed summary of prior/current measurements: weighted mean, weighted standard deviation, linear trend, effective count, and weighted state age. The decay half-life is 12 hours. It is feature engineering for a patient-specific state representation, not a physiological or causal model.

## Evaluation safeguards

- Split by patient with GroupKFold; all records from a held-out patient remain out of that fold's training set.
- Fit imputation, scaling, and each model only within a training fold.
- Use the same eligible prediction points for every model.
- Calculate confidence intervals by resampling patients, not individual observations.
- Report the full cohort flow, patient denominators, and stored out-of-fold predictions.

## Paper boundary

The MIMIC-IV demo has 100 patients and is only a pipeline demonstration. It does not establish generalizable performance, medication response, causal treatment benefit, clinical efficacy, or a deployable digital twin. A full study needs a protocol freeze, a full approved data extract, temporal/external validation, and reporting aligned with TRIPOD+AI.
