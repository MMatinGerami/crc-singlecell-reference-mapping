.PHONY: all data prepare transfer openset integration figures calibration pelka patients conformal lineage refsize test lint

all: prepare transfer openset integration figures calibration pelka patients conformal lineage refsize

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
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
