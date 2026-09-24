#!/usr/bin/env bash
# Download the two colorectal cancer scRNA-seq cohorts of Lee et al. (2020) from GEO.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
fetch() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series
# SMC cohort (Korea, 23 patients): reference
fetch "$GEO/GSE132nnn/GSE132465/suppl/GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz" smc_counts.txt.gz
fetch "$GEO/GSE132nnn/GSE132465/suppl/GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz"     smc_annotation.txt.gz
# KUL3 cohort (Belgium, 6 patients): query
fetch "$GEO/GSE144nnn/GSE144735/suppl/GSE144735_processed_KUL3_CRC_10X_raw_UMI_count_matrix.txt.gz" kul3_counts.txt.gz
fetch "$GEO/GSE144nnn/GSE144735/suppl/GSE144735_processed_KUL3_CRC_10X_annotation.txt.gz"            kul3_annotation.txt.gz
shasum -a 256 *.gz > SHA256SUMS
