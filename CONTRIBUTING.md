# Contributing

Contributions are welcome, in particular additional reference-mapping methods for the benchmark.
Work happens through pull requests from a fork; `main` is protected and only accepts changes
that pass CI.

## Set up

```bash
git clone https://github.com/<your-user>/crc-singlecell-reference-mapping.git
cd crc-singlecell-reference-mapping
uv sync            # locked environment, Python 3.12
make data          # both cohorts from GEO, about 190 MB
make test          # unit tests, under a minute
```

`uv run ruff check .` and `uv run ruff format .` before committing; the pre-commit hooks run
the same checks.

## Adding a method

Every method is evaluated the same way: trained on the SMC reference, applied to the KUL3
query, scored on held-out patients with patient-bootstrap intervals, and run through the
open-set, calibration and conformal analyses. A new method joins that pipeline rather than
getting its own evaluation.

1. Implement it as a function that returns a `Transfer` (see `src/scmap/baselines.py` for the
   shape; `src/scmap/models.py` for the scvi-tools models). It must produce, for every query
   cell, a predicted label and a confidence score in [0, 1], and, if it has a latent space,
   the embedding of reference and query cells.
2. Register it in `METHODS` in `src/scmap/pipeline.py` and call it from `run_transfer`.
   Hyperparameters go in `configs/default.yaml`, not in the code.
3. Add its label and colour in `src/scmap/plotting.py` (`METHOD_LABELS`, `SERIES`) and add
   it to the `CLASSIFIERS` lists in `scripts/05_figures.py` and `scripts/08_per_patient.py`.
4. Add a unit test in `tests/` that fits the method on a small synthetic dataset and checks
   the output shapes and that confidences lie in [0, 1].
5. Run `make transfer openset figures` and include the regenerated tables under
   `results/tables/` and figures under `results/figures/` in the pull request, together with
   a short note of the runtime and the machine used.

No change to the data split, the evaluation code or the existing methods should be mixed into
a pull request that adds a method; open a separate pull request for those, with the reason.

## Pull requests

- One topic per pull request, with a description of what changed and why.
- CI must pass (ruff, pytest).
- Contributors are listed in `CITATION.cff`; add yourself in the pull request that adds your
  work.

## Style

Plain, specific wording in code comments, docstrings and documentation. 
