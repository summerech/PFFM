#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_OUTPUT="${OUTPUT_ROOT:-outputs/paper_ablations}"
for variant in full atom_only no_cfg no_constraints; do
  echo "[$(date -Iseconds)] running $variant"
  OUTPUT_ROOT="$BASE_OUTPUT" PFFM_VARIANT="$variant" \
    bash scripts/run_paper_generation.sh
done
