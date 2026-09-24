.PHONY: all data prepare transfer openset integration figures calibration pelka test lint

all: prepare transfer openset integration figures calibration pelka

data:         ; bash scripts/download_data.sh
prepare:      ; uv run python scripts/01_prepare_data.py
transfer:     ; uv run python scripts/02_label_transfer.py
openset:      ; uv run python scripts/03_open_set.py
integration:  ; uv run python scripts/04_integration_benchmark.py
figures:      ; uv run python scripts/05_figures.py
calibration:  ; uv run python scripts/06_calibrated_abstention.py
pelka:        ; uv run python scripts/07_external_pelka.py
test:         ; uv run pytest -q
lint:         ; uv run ruff check . && uv run ruff format --check .
