# Changelog

## Unreleased

- Text checked against the result tables: 33 shared fine subtypes (not 31; also in Fig 2), close-relative open-set AUROC 0.66 to 0.83, PCA + kNN ties scANVI as abstention signal, unfiltered accuracy of the calibration model 83.5%, Pelka subsample is random and from two hospitals, doublet column is the share caught, two per-patient ranges, `__version__` 0.5.0. Conformal text no longer presents in-sample calibration coverage as a result, and the label-shift reading is stated as a hypothesis.
- Residual doublets (`scripts/17_doublets.py`, `make doublets`): Scrublet calls 0.2% of query cells doublets; they are mislabelled 56 to 88 times more often than singlets, and the scANVI latent-distance novelty score ranks them far better (AUROC 0.86) than softmax confidence (0.72). Adds `scikit-image` for Scrublet's automatic threshold.

## 0.5.0 (2026-09-29)

- Calibration of scANVI confidence under the cohort shift, with temperature scaling fitted on reference patients (`scmap.calibration`, `scripts/16_calibration_shift.py`).
- Two-hospital site-shift experiment repeated with scANVI (`scripts/15_site_shift_pelka_scanvi.py`): a mixed reference raises macro-F1 and restores 90% coverage on both hospitals.
- `scmap annotate` command: maps a new query onto the saved scANVI reference and labels its cells with confidence and the pre-committed abstention flag.
- Reference-size curve repeated with scANVI and scArches mapping (`scripts/14_reference_size_scanvi.py`); same plateau from six patients as the baseline.

## 0.4.0 (2026-09-29)

- Weighted conformal prediction under covariate shift, tested and found not to recover the coverage loss (`scripts/12_weighted_conformal.py`); mapped probabilities and embeddings cached.
- Two-hospital site-shift experiment on the Pelka atlas with single-site and mixed references (`scripts/13_site_shift_pelka.py`).
- Dockerfile, `make docker`, pre-commit configuration.

## 0.3.0 (2026-09-28)

- Per-lineage novelty thresholds tested and rejected (`scripts/10_per_lineage_novelty.py`).
- Transfer accuracy vs number of reference patients (`scripts/11_reference_size.py`).
- Citation file.

## 0.2.0 (2026-09-28)

- Per-patient and per-tissue accuracy on the query cohort (`scripts/08_per_patient.py`).
- Conformal prediction sets calibrated on held-out reference patients and evaluated on the query, with per-patient coverage (`scmap.conformal`, `scripts/09_conformal_shift.py`).
- Model card (`MODEL_CARD.md`).
- README shortened; results unchanged.

## 0.1.0 (2026-09-24)

- Cross-cohort label transfer (logistic regression, kNN on PCA, scVI + kNN, scANVI with scArches) between the SMC and KUL3 cohorts of Lee et al. 2020.
- Open-set experiments, calibrated abstention, integration metrics, second external cohort (Pelka et al. 2021).
