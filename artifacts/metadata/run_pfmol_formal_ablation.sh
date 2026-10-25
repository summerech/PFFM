#!/usr/bin/env bash
set -uo pipefail

TDROOT="${TARGETDIFF_ROOT:-/home/cql/targetdiff-official/targetdiff-main}"
PFROOT="${PFMOL_ROOT:-/home/cql/pfmol-flow}"
TDPYTHON="${TARGETDIFF_PYTHON:-/home/cql/anaconda3/envs/targetdiff-official/bin/python}"
PFPYTHON="${PFMOL_PYTHON:-/home/cql/anaconda3/envs/pfmol-flow/bin/python}"
ROOT_OUT="${ROOT_OUT:-short_benchmark/pfmol_formal_ablation_clean}"
VARIANTS_CSV="${VARIANTS:-a_full,b_no_cfg,c_no_property_loss,d_no_constraints}"
SEEDS_CSV="${SEEDS:-2021,2022,2023}"
SAMPLES="${SAMPLES:-100}"
MAX_DOCKING_JOBS="${MAX_DOCKING_JOBS:-32}"
GENERATION_GPUS="${GENERATION_GPUS:-0,1}"
GENERATION_SHARDS_PER_GPU="${GENERATION_SHARDS_PER_GPU:-1}"

cd "$TDROOT" || exit 1
export PYTHONPATH="$TDROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT_OUT/logs"
IFS=',' read -r -a VARIANTS <<<"$VARIANTS_CSV"
IFS=',' read -r -a SEEDS <<<"$SEEDS_CSV"
pocket_list=$(seq 0 99 | paste -sd, -)
even_pockets=$(seq 0 2 98 | paste -sd, -)
odd_pockets=$(seq 1 2 99 | paste -sd, -)

variant_args() {
  local variant="$1"
  case "$variant" in
    a_full)
      printf '%s\n' --guidance-scale 2
      ;;
    b_no_cfg)
      printf '%s\n' --guidance-scale 1
      ;;
    c_no_property_loss)
      printf '%s\n' \
        --checkpoint short_benchmark/fragment_cfg_no_property_2k.pt \
        --guidance-scale 2
      ;;
    d_no_constraints)
      printf '%s\n' \
        --disable-constrained-decoding \
        --guidance-scale 2
      ;;
    atom_only)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_ablation/checkpoints/atom_only.pt \
        --guidance-scale 2
      ;;
    residue_graph)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_ablation/checkpoints/residue_graph.pt \
        --guidance-scale 2
      ;;
    residue_hypergraph)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_ablation/checkpoints/residue_hypergraph.pt \
        --guidance-scale 2
      ;;
    residue_hypergraph_guided)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_ablation_v2/checkpoints/residue_hypergraph_guided.pt \
        --guidance-scale 2
      ;;
    esm2_residue_hypergraph)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_esm2/checkpoints/esm2_residue_hypergraph.pt \
        --guidance-scale 2
      ;;
    esm2_no_cfg)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_esm2/checkpoints/esm2_residue_hypergraph.pt \
        --guidance-scale 1
      ;;
    esm2_no_constraints)
      printf '%s\n' \
        --checkpoint short_benchmark/pocket_encoder_esm2/checkpoints/esm2_residue_hypergraph.pt \
        --disable-constrained-decoding \
        --guidance-scale 2
      ;;
    *)
      echo "Unknown ablation variant: $variant" >&2
      return 2
      ;;
  esac
}

run_one() {
  local variant="$1" seed="$2" out="$ROOT_OUT/$variant/seed_$seed"
  mkdir -p "$out"/{logs,docked_csv,docked_poses,prepared_ligands,dock_done}
  [[ -f "$out/DONE" ]] && return 0
  rm -f "$out/FAILED"
  date -Iseconds >"$out/RUNNING"

  local common=(
    configs/training_fragment_bond_flow.yml
    --checkpoint short_benchmark/fragment_cfg_property_st_100k_2k.pt
    --vocab data/processed/crossdocked_rbrics_dummy_100k.vocab.json
    --samples-per-pocket "$SAMPLES"
    --num-steps 100
    --desired-heavy-atoms 28
    --desired-qed 0.8
    --guidance-scale 2
    --token-temperature 0.9
    --property-checkpoint short_benchmark/fragment_property_qed_100k_10k.pt
  )
  local extra=()
  mapfile -t extra < <(variant_args "$variant") || return 1

  if [[ ! -f "$out/GENERATION_DONE" ]]; then
    IFS=',' read -r -a generation_gpus <<<"$GENERATION_GPUS"
    local gpu_count="${#generation_gpus[@]}"
    local total_shards=$((gpu_count * GENERATION_SHARDS_PER_GPU))
    if [[ "$total_shards" -eq 1 ]]; then
      CUDA_VISIBLE_DEVICES="${generation_gpus[0]}" "$TDPYTHON" scripts/sample_fragment_flow.py \
        "${common[@]}" "${extra[@]}" --pockets "$pocket_list" \
        --seed "${seed}00" --output "$out/samples.json" --device cuda:0 \
        >"$out/logs/generate_all.log" 2>&1 || {
          date -Iseconds >"$out/FAILED"; rm -f "$out/RUNNING"; return 1;
        }
    else
      local generation_pids=()
      local shard_outputs=()
      local shard gpu shard_pockets shard_output shard_seed
      for ((shard = 0; shard < total_shards; shard++)); do
        gpu="${generation_gpus[$((shard % gpu_count))]}"
        shard_pockets=$(seq "$shard" "$total_shards" 99 | paste -sd, -)
        shard_output="$out/shard_${shard}.json"
        shard_seed=$((seed * 100 + shard))
        shard_outputs+=("$shard_output")
        CUDA_VISIBLE_DEVICES="$gpu" "$TDPYTHON" scripts/sample_fragment_flow.py \
          "${common[@]}" "${extra[@]}" --pockets "$shard_pockets" \
          --seed "$shard_seed" --output "$shard_output" --device cuda:0 \
          >"$out/logs/generate_shard_${shard}.log" 2>&1 &
        generation_pids+=("$!")
      done
      local generation_failed=0
      local generation_pid
      for generation_pid in "${generation_pids[@]}"; do
        wait "$generation_pid" || generation_failed=1
      done
      if [[ "$generation_failed" != 0 ]]; then
        date -Iseconds >"$out/FAILED"; rm -f "$out/RUNNING"; return 1
      fi
      "$TDPYTHON" scripts/merge_json_lists.py "${shard_outputs[@]}" \
        --output "$out/samples.json" || return 1
    fi
    "$TDPYTHON" scripts/assemble_fragment_samples.py "$out/samples.json" \
      --output-dir "$out/assembled" >"$out/logs/assemble.log" 2>&1 || return 1
    "$TDPYTHON" scripts/evaluate_fragment_samples.py "$out/assembled/summary.csv" \
      --samples "$out/samples.json" --requested $((100 * SAMPLES)) \
      --output "$out/light_metrics.json" >"$out/logs/evaluate.log" 2>&1 || return 1
    "$TDPYTHON" scripts/prepare_fragment_docking.py \
      configs/training_fragment_bond_flow.yml --samples "$out/samples.json" \
      --sdf "$out/assembled/assembled.sdf" \
      --summary "$out/assembled/summary.csv" --output-dir "$out/raw_sdf" \
      --out "$out/manifest.csv" >"$out/logs/prepare.log" 2>&1 || return 1
    date -Iseconds >"$out/GENERATION_DONE"
  fi

  if [[ ! -f "$out/DOCKING_DONE" ]]; then
    local failed=0
    for pocket in $(seq 0 99); do
      [[ -f "$out/dock_done/${pocket}.done" ]] && continue
      while [[ $(jobs -rp | wc -l) -ge "$MAX_DOCKING_JOBS" ]]; do
        wait -n || failed=1
      done
      (
        "$PFPYTHON" "$PFROOT/scripts/crossdocked_pamol_style_docking.py" \
          --input "$out/manifest_pocket${pocket}.csv" \
          --out "$out/docked_csv/pocket_${pocket}.csv" \
          --data-root "$PFROOT/data/crossdocked_pocket10/crossdocked_pocket10" \
          --prepared-root "$PFROOT/runs/crossdocked_prepared_receptors_box20" \
          --pose-root "$out/docked_poses/pocket_${pocket}" \
          --prepared-ligand-root "$out/prepared_ligands/pocket_${pocket}" \
          --dock-bin "$PFROOT/tools/qvina/qvina-master/bin/qvina2.1" \
          --receptor-bin /home/cql/anaconda3/envs/pfmol-flow/bin/mk_prepare_receptor.py \
          --meeko-ligand-bin /home/cql/anaconda3/envs/pfmol-flow/bin/mk_prepare_ligand.py \
          --box-size 20 --exhaustiveness 8 --num-modes 9 --seed 17 --cpu 1 \
          >"$out/logs/dock_${pocket}.log" 2>&1 && \
        date -Iseconds >"$out/dock_done/${pocket}.done"
      ) &
    done
    while [[ $(jobs -rp | wc -l) -gt 0 ]]; do wait -n || failed=1; done
    local docked
    docked=$(find "$out/dock_done" -maxdepth 1 -name '*.done' -type f | wc -l)
    if [[ "$failed" != 0 || "$docked" -ne 100 ]]; then
      date -Iseconds >"$out/FAILED"; rm -f "$out/RUNNING"; return 1
    fi
    local inputs=()
    for pocket in $(seq 0 99); do inputs+=("$out/docked_csv/pocket_${pocket}.csv"); done
    "$TDPYTHON" scripts/merge_csv_files.py "${inputs[@]}" --output "$out/docked.csv" || return 1
    date -Iseconds >"$out/DOCKING_DONE"
  fi

  "$TDPYTHON" scripts/summarize_filtered_pockets.py --root "$out" \
    --input "$out/docked.csv" --pockets "$pocket_list" \
    --requested $((100 * SAMPLES)) \
    --model-name "PFMol-Flow ablation: $variant, seed $seed" \
    --out "$out/metrics.json" --csv-out "$out/metrics.csv" \
    >"$out/logs/summarize.log" 2>&1 || return 1
  date -Iseconds >"$out/DONE"
  rm -f "$out/RUNNING" "$out/FAILED"
}

date -Iseconds >"$ROOT_OUT/STARTED"
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "[$(date -Iseconds)] start $variant seed=$seed" | tee -a "$ROOT_OUT/orchestrator.log"
    if ! run_one "$variant" "$seed"; then
      echo "[$(date -Iseconds)] failed $variant seed=$seed" | tee -a "$ROOT_OUT/orchestrator.log"
      date -Iseconds >"$ROOT_OUT/FAILED"
      exit 1
    fi
    echo "[$(date -Iseconds)] done $variant seed=$seed" | tee -a "$ROOT_OUT/orchestrator.log"
  done
done
date -Iseconds >"$ROOT_OUT/DONE"
rm -f "$ROOT_OUT/FAILED"
