#!/usr/bin/env bash
set -euo pipefail

TDROOT="${TARGETDIFF_ROOT:-/home/cql/targetdiff-official/targetdiff-main}"
ROOT="short_benchmark/pocket_encoder_esm2_full100_three_seed"

cd "$TDROOT"
mkdir -p "$ROOT/logs"
trap 'date -Iseconds >"$ROOT/FAILED"' ERR
date -Iseconds >"$ROOT/RUNNING"
rm -f "$ROOT/FAILED" "$ROOT/DONE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="/home/cql/anaconda3/envs/targetdiff-official/lib:${LD_LIBRARY_PATH:-}"

ROOT_OUT="$ROOT" \
VARIANTS="esm2_residue_hypergraph" \
SEEDS="2021,2022,2023" \
SAMPLES=100 \
MAX_DOCKING_JOBS=24 \
GENERATION_GPUS="0,1" \
GENERATION_SHARDS_PER_GPU=4 \
bash scripts/run_pfmol_formal_ablation.sh

date -Iseconds >"$ROOT/DONE"
rm -f "$ROOT/RUNNING" "$ROOT/FAILED"
