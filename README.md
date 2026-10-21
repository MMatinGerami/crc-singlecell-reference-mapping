# crc-singlecell-reference-mapping

How well do cell-type labels transfer from one colorectal cancer scRNA-seq cohort to another, and
can we flag cells whose type is missing from the reference?

Reference: SMC cohort (Korea, 23 patients, about 63k cells). Query: KUL3 cohort (Belgium,
6 patients, about 27k cells). Both from [Lee et al., 2020](https://doi.org/10.1038/s41588-020-0636-z).
A second query, [Pelka et al., 2021](https://doi.org/10.1016/j.cell.2021.08.003), tests a change of
hospital. Query labels are only used for scoring.

<p align="center"><img src="results/figures/fig1_umap_scanvi.png" width="85%"></p>

## Methods compared

Logistic regression and kNN on PCA (no batch correction), scVI + kNN, and scANVI. The query is
mapped into scVI and scANVI with scArches, keeping the reference weights frozen.

## Main results

- Major lineages transfer almost perfectly (macro-F1 0.995 to 0.997). The 33 shared fine subtypes
  reach 0.75 with scANVI and 0.74 with logistic regression.
- Discrete cell types transfer well (mast cells, enteric glia); continuous states do not
  (proliferating myeloid cells, CMS tumour subtypes).
- A whole lineage missing from the reference is easy to flag by latent distance (AUROC 0.99). A
  missing subtype with a close relative is mostly absorbed by it: 96% of tip-like endothelial
  cells were called stalk-like.
- Abstaining on the 20% least confident cells raises accuracy from 0.82 to 0.89.
- A reference drawn from both Pelka hospitals instead of one raises macro-F1 on the smaller
  hospital from 0.82 to 0.92.

More analyses (calibration, conformal sets, reference size, doublets) are in
[docs/results.md](docs/results.md).

## Usage

```bash
uv sync
make data     # both cohorts from GEO, about 190 MB
make all
make test
```

## Limitations

- One tissue, one platform, and only six query patients, so intervals are wide.
- "Novel" means absent from the reference annotation, not from the tissue.
- Fine labels such as CMS subtypes cut through continuous biology, so some errors are
  disagreements about boundaries.

## References

Lee et al., Nat Genet 2020 (GSE132465, GSE144735); Pelka et al., Cell 2021 (GSE178341);
scVI ([Lopez et al., 2018](https://doi.org/10.1038/s41592-018-0229-2)), scANVI
([Xu et al., 2021](https://doi.org/10.15252/msb.20209620)), scArches
([Lotfollahi et al., 2022](https://doi.org/10.1038/s41587-021-01001-7)).

## License

MIT
