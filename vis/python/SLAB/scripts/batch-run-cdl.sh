#!/usr/bin/env bash

set -euo pipefail

PY_SCRIPT="generate_cdl.py"
NPROC=6

DIRS=(
  "../../../../data/TDSC/M10_B0.001_R2_D0.02_PR/"
  "../../../../data/TDSC/M10_B0.01_R2_D0.02_PR/"
  "../../../../data/TDSC/M5_B0.1_R2_D0.02_PR/"
  "../../../../data/TDSC/M10_B0.1_R2_D0.02_PR/"
  "../../../../data/TDSC/M20_B0.1_R2_D0.02_PR/"
)

printf '%s\n' "${DIRS[@]}" | xargs -I {} -P "$NPROC" python "$PY_SCRIPT" --dir "{}"