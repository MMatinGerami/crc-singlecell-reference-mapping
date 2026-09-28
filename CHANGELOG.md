# Changelog

## 0.2.0 (2026-09-28)

- Per-patient and per-tissue accuracy on the query cohort (`scripts/08_per_patient.py`).
- Conformal prediction sets calibrated on held-out reference patients and evaluated on the query, with per-patient coverage (`scmap.conformal`, `scripts/09_conformal_shift.py`).
- Model card (`MODEL_CARD.md`).
- README shortened; results unchanged.

## 0.1.0 (2026-09-24)

- Cross-cohort label transfer (logistic regression, kNN on PCA, scVI + kNN, scANVI with scArches) between the SMC and KUL3 cohorts of Lee et al. 2020.
- Open-set experiments, calibrated abstention, integration metrics, second external cohort (Pelka et al. 2021).
