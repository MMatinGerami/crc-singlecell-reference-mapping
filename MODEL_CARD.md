# Model card: cross-cohort cell-type annotation (scANVI reference)

Research prototype. Not intended for clinical use.

## Model

- **Task:** assign each cell of a new colorectal cancer scRNA-seq cohort one of 6 major cell types and one of 31 fine subtypes, and flag cells whose type is not in the reference.
- **Model:** scVI (30 latent dimensions, 2 layers, 128 hidden units) trained on reference counts with sample as batch, then scANVI initialised from it. New cohorts are mapped with scArches: reference weights frozen, only query-batch parameters trained. Three baselines (logistic regression, kNN on PCA, scVI + kNN) are reported alongside.
- **Input:** raw UMI counts for 3,000 highly variable genes selected on the reference.
- **Output:** per-cell probabilities over the reference subtypes; a confidence score (max probability); a latent-distance novelty score; with `scripts/09_conformal_shift.py`, a set of candidate subtypes with a stated error rate.

## Data

- **Reference:** SMC cohort, Lee et al. 2020 (Korea): 23 patients, ~63k cells, tumour and normal tissue.
- **Query:** KUL3 cohort, Lee et al. 2020 (Belgium): 6 patients, ~27k cells, tumour core, border and normal mucosa. Query labels are used only for scoring.
- **Second query:** Pelka et al. 2021 (USA): 40,000-cell subsample, 62 patients, 10x v2 and v3.
- Five reference patients (SMC06, 08, 11, 15, 19) are held out to calibrate thresholds; the models used for calibration are retrained without them.

## Performance on the query (patient-level bootstrap, 95% CI)

| Level | scANVI | Logistic regression |
|---|---|---|
| Coarse (6 types), macro-F1 | 0.997 | 0.995 |
| Fine (31 subtypes), macro-F1 | 0.754 (0.71 to 0.75) | 0.742 (0.70 to 0.74) |
| Fine accuracy | 0.82 | 0.82 |

## Performance by patient and tissue (fine subtypes, accuracy)

| Subgroup | scANVI | Range across methods |
|---|---|---|
| KUL01 (4,307 cells) | 0.79 | 0.69 to 0.79 |
| KUL19 (7,704) | 0.85 | 0.84 to 0.85 |
| KUL21 (4,293) | 0.76 | 0.66 to 0.76 |
| KUL28 (1,688) | 0.85 | 0.83 to 0.86 |
| KUL30 (3,157) | 0.87 | 0.86 to 0.90 |
| KUL31 (2,069) | 0.83 | 0.80 to 0.83 |
| Normal mucosa (9,095) | 0.90 | 0.85 to 0.91 |
| Tumour border (6,230) | 0.79 | 0.77 to 0.79 |
| Tumour core (7,893) | 0.76 | 0.69 to 0.77 |

Accuracy varies by about 11 points between patients and by 14 points between tissues. Tumour cells are the hardest, largely because the tumour-intrinsic CMS subtypes are continuous states discretised differently in each cohort.

## Uncertainty

**Abstention.** Rejecting the 20% least-confident scANVI calls raises accuracy on the rest from 0.82 to 0.89. Thresholds fixed on the held-out reference patients keep their coverage on the query but deliver about 4 points less accuracy than promised.

**Conformal sets** (`scripts/09_conformal_shift.py`, score 1 - p(true subtype), calibrated on the held-out reference patients):

| Nominal coverage | Threshold | Calibration patients | Query (KUL3) | Query per-patient range | Mean set size (query) |
|---|---|---|---|---|---|
| 90% | global | 0.900 | 0.864 | 0.82 to 0.91 | 1.1 |
| 90% | per subtype | 0.904 | 0.854 | 0.83 to 0.87 | 5.1 |
| 95% | global | 0.950 | 0.914 | 0.89 to 0.94 | 1.3 |
| 95% | per subtype | 0.954 | 0.919 | 0.90 to 0.93 | 8.4 |

The guarantee holds exactly within the reference cohort and loses 3 to 4 points on the Belgian cohort. Neither per-subtype thresholds nor weighted conformal prediction with density-ratio weights (0.864 to 0.867 at 90%) recover it, which points to a shift in the labels rather than in the inputs.

**Novelty.** Per-lineage thresholds on latent distance were tested and rejected (AUROC 0.373 vs 0.674 for the global score; novel cells are not outliers within their assigned lineage). A deleted lineage with no close relative is detected by latent distance (AUROC 0.99); a deleted subtype with a close relative is mostly absorbed by it (AUROC 0.60 to 0.83). Populations missing from the reference annotation are detected at best at AUROC 0.67.

## Site shift with consistent labels (Pelka et al., two hospitals)

Logistic-regression baseline, 12 reference patients. Reference from the other hospital only: macro-F1 0.81 on DFCI, 0.91 on MGH. Reference of the same size drawn from both hospitals: 0.89 on both. Conformal coverage at nominal 90%: 0.91 (DFCI) and 0.87 (MGH) for either design. scANVI with scArches mapping, one draw: single-site reference 0.82 (DFCI) and 0.88 (MGH), mixed reference 0.92 and 0.93; coverage 0.90 and 0.87 single-site, 0.94 and 0.94 mixed.

## Reference size

With the logistic-regression baseline, fine-subtype macro-F1 on the query is 0.42 to 0.72 with 3 reference patients, 0.72 with 6, and 0.73 to 0.74 from 10 patients upwards (0.74 with all 23). Coarse accuracy exceeds 0.98 with 3 patients. scANVI with scArches mapping, one draw per size: 0.39, 0.73, 0.73, 0.75, 0.74 for 3, 6, 10, 15 and 23 patients.

## Limitations

- One tissue, one sequencing platform, six query patients; both main cohorts annotated by the same group.
- Fine labels for continuous states (CMS subtypes, macrophage polarisation, proliferation) are partly conventions, and transfer poorly for that reason.
- Coverage and abstention guarantees assume the query is exchangeable with the calibration cells; the results above show what a cohort shift costs.
- A reference mapping cannot detect a subtype that resembles a known one, so novel populations inside heterogeneous compartments will be silently absorbed.
- Open-set runs use a shorter training schedule than the main run.

## Intended use

Method evaluation and teaching. Annotation of a new cohort with this pipeline should keep the abstention and novelty flags on and review flagged cells manually.
