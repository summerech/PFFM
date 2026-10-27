#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
DOCKING_PYTHON="${DOCKING_PYTHON:-$PYTHON}"
CONFIG="${CONFIG:-configs/training_fragment_bond_flow.yml}"
VOCAB="${VOCAB:-artifacts/metadata/crossdocked_rbrics_dummy_100k.vocab.json}"
PFFM_CHECKPOINT="${PFFM_CHECKPOINT:-checkpoints/pffm_esm2_residue_hypergraph.pt}"
ATOM_ONLY_CHECKPOINT="${ATOM_ONLY_CHECKPOINT:-checkpoints/pffm_atom_only.pt}"
PROPERTY_CHECKPOINT="${PROPERTY_CHECKPOINT:-checkpoints/fragment_property_qed.pt}"
DATA_ROOT="${DATA_ROOT:-data/crossdocked_v1.1_rmsd1.0_pocket10/crossdocked_pocket10}"
QVINA_BIN="${QVINA_BIN:-qvina2.1}"
RECEPTOR_BIN="${RECEPTOR_BIN:-mk_prepare_receptor.py}"
LIGAND_BIN="${LIGAND_BIN:-mk_prepare_ligand.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_main}"
SEEDS="${SEEDS:-2021,2022,2023}"
SAMPLES_PER_POCKET="${SAMPLES_PER_POCKET:-100}"
DEVICE="${DEVICE:-cuda:0}"
VARIANT="${PFFM_VARIANT:-full}"
MAX_DOCKING_JOBS="${MAX_DOCKING_JOBS:-16}"

POCKETS="$(seq 0 99 | paste -sd, -)"
IFS=',' read -r -a SEED_LIST <<<"$SEEDS"

case "$VARIANT" in
  full) CHECKPOINT="$PFFM_CHECKPOINT"; GUIDANCE=2; EXTRA=() ;;
  atom_only) CHECKPOINT="$ATOM_ONLY_CHECKPOINT"; GUIDANCE=2; EXTRA=() ;;
  no_cfg) CHECKPOINT="$PFFM_CHECKPOINT"; GUIDANCE=1; EXTRA=() ;;
  no_constraints) CHECKPOINT="$PFFM_CHECKPOINT"; GUIDANCE=2; EXTRA=(--disable-constrained-decoding) ;;
  *) echo "Unknown PFFM_VARIANT: $VARIANT" >&2; exit 2 ;;
esac

for required in "$CONFIG" "$VOCAB" "$CHECKPOINT" "$PROPERTY_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "Missing required file: $required" >&2; exit 3; }
done

for seed in "${SEED_LIST[@]}"; do
  out="$OUTPUT_ROOT/$VARIANT/seed_$seed"
  mkdir -p "$out"/{logs,assembled,raw_sdf,docked_csv,docked_poses,prepared_ligands,dock_done}
  [[ -f "$out/DONE" ]] && continue

  if [[ ! -f "$out/GENERATION_DONE" ]]; then
    "$PYTHON" scripts/sample_fragment_flow.py "$CONFIG" \
      --checkpoint "$CHECKPOINT" --vocab "$VOCAB" --pockets "$POCKETS" \
      --samples-per-pocket "$SAMPLES_PER_POCKET" --num-steps 100 \
      --desired-heavy-atoms 28 --desired-qed 0.8 --guidance-scale "$GUIDANCE" \
      --token-temperature 0.9 --property-checkpoint "$PROPERTY_CHECKPOINT" \
      --seed "${seed}00" --output "$out/samples.json" --device "$DEVICE" \
      "${EXTRA[@]}" >"$out/logs/generate.log" 2>&1
    "$PYTHON" scripts/assemble_fragment_samples.py "$out/samples.json" \
      --output-dir "$out/assembled" >"$out/logs/assemble.log" 2>&1
    "$PYTHON" scripts/evaluate_fragment_samples.py "$out/assembled/summary.csv" \
      --samples "$out/samples.json" --requested "$((100*SAMPLES_PER_POCKET))" \
      --output "$out/light_metrics.json" >"$out/logs/light_metrics.log" 2>&1
    "$PYTHON" scripts/prepare_fragment_docking.py "$CONFIG" \
      --samples "$out/samples.json" --sdf "$out/assembled/assembled.sdf" \
      --summary "$out/assembled/summary.csv" --output-dir "$out/raw_sdf" \
      --out "$out/manifest.csv" >"$out/logs/prepare_docking.log" 2>&1
    date -Iseconds >"$out/GENERATION_DONE"
  fi

  for pocket in $(seq 0 99); do
    [[ -f "$out/dock_done/$pocket.done" ]] && continue
    while [[ $(jobs -rp | wc -l) -ge "$MAX_DOCKING_JOBS" ]]; do wait -n; done
    (
      "$DOCKING_PYTHON" scripts/crossdocked_pamol_style_docking.py \
        --input "$out/manifest_pocket${pocket}.csv" \
        --out "$out/docked_csv/pocket_${pocket}.csv" --data-root "$DATA_ROOT" \
        --prepared-root "$OUTPUT_ROOT/prepared_receptors_box20" \
        --pose-root "$out/docked_poses/pocket_${pocket}" \
        --prepared-ligand-root "$out/prepared_ligands/pocket_${pocket}" \
        --dock-bin "$QVINA_BIN" --receptor-bin "$RECEPTOR_BIN" \
        --meeko-ligand-bin "$LIGAND_BIN" --box-size 20 --exhaustiveness 8 \
        --num-modes 9 --seed 17 --cpu 1 >"$out/logs/dock_${pocket}.log" 2>&1
      date -Iseconds >"$out/dock_done/$pocket.done"
    ) &
  done
  wait

  files=()
  for pocket in $(seq 0 99); do files+=("$out/docked_csv/pocket_${pocket}.csv"); done
  "$PYTHON" scripts/merge_csv_files.py "${files[@]}" --output "$out/docked.csv"
  "$PYTHON" scripts/summarize_filtered_pockets.py --root "$out" \
    --input "$out/docked.csv" --pockets "$POCKETS" \
    --requested "$((100*SAMPLES_PER_POCKET))" --model-name "PFFM $VARIANT seed $seed" \
    --out "$out/metrics.json" --csv-out "$out/metrics.csv"
  date -Iseconds >"$out/DONE"
done
