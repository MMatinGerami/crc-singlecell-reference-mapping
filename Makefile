.PHONY: all data prepare transfer openset integration figures calibration pelka patients conformal lineage refsize weighted siteshift refsize-scanvi siteshift-scanvi test lint docker

all: prepare transfer openset integration figures calibration pelka patients conformal lineage refsize weighted siteshift

data:         ; bash scripts/download_data.sh
prepare:      ; uv run python scripts/01_prepare_data.py
transfer:     ; uv run python scripts/02_label_transfer.py
openset:      ; uv run python scripts/03_open_set.py
integration:  ; uv run python scripts/04_integration_benchmark.py
figures:      ; uv run python scripts/05_figures.py
calibration:  ; uv run python scripts/06_calibrated_abstention.py
pelka:        ; uv run python scripts/07_external_pelka.py
patients:     ; uv run python scripts/08_per_patient.py
conformal:    ; uv run python scripts/09_conformal_shift.py
lineage:      ; uv run python scripts/10_per_lineage_novelty.py
refsize:      ; uv run python scripts/11_reference_size.py
weighted:     ; uv run python scripts/12_weighted_conformal.py
siteshift:    ; uv run python scripts/13_site_shift_pelka.py
refsize-scanvi: ; uv run python scripts/14_reference_size_scanvi.py
siteshift-scanvi: ; uv run python scripts/15_site_shift_pelka_scanvi.py
docker:       ; docker build -t scmap . && docker run --rm -v "$$PWD/data:/app/data" -v "$$PWD/results:/app/results" scmap make test
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
