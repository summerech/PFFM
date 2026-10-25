#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
INPUT_CSV="${INPUT_CSV:?Set INPUT_CSV to the frozen five-pocket test manifest}"
DATA_ROOT="${DATA_ROOT:-data/crossdocked_v1.1_rmsd1.0_pocket10/crossdocked_pocket10}"
TOPOLOGY_REFINER_CHECKPOINT="${TOPOLOGY_REFINER_CHECKPOINT:-checkpoints/topology_refiner_balanced.pt}"
ENERGY_CHECKPOINT="${ENERGY_CHECKPOINT:-checkpoints/triview_energy_physical.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_refinement_fixed_test}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$OUTPUT_ROOT"

cat >"$OUTPUT_ROOT/FROZEN_CONFIG.txt" <<'EOF'
test_pockets=28,29,47,48,55
topology_steps=12
topology_strength=0.5
topology_physics_step=0.08
topology_physics_iterations=2
topology_protein_clash_scale=0.85
energy_steps=16
translation_step=0.04
rotation_step_deg=1.0
torsion_step_deg=2.0
selection_mode=clash_aware
repulsion_step=0.05
energy_tolerance_kcal=0.25
overlap_weight=3.0
soft_clash_weight=2.0
EOF

"$PYTHON" scripts/refine_topology_preserving.py \
  --input "$INPUT_CSV" --checkpoint "$TOPOLOGY_REFINER_CHECKPOINT" \
  --data-root "$DATA_ROOT" --path-root "$ROOT" --output-dir "$OUTPUT_ROOT/topology_base_sdf" \
  --out "$OUTPUT_ROOT/topology_base.csv" --split-by-pocket --steps 12 --strength 0.5 \
  --physics-step 0.08 --physics-iterations 2 --protein-clash-scale 0.85 \
  --preserve-protein-clearance --clearance-tolerance 0 --device "$DEVICE"

"$PYTHON" scripts/refine_rigid_torsion_energy.py \
  --input "$OUTPUT_ROOT/topology_base.csv" --checkpoint "$ENERGY_CHECKPOINT" \
  --data-root "$DATA_ROOT" --path-root "$ROOT" --output-dir "$OUTPUT_ROOT/balanced_sdf" \
  --out "$OUTPUT_ROOT/balanced.csv" --steps 16 --translation-step 0.04 \
  --rotation-step-deg 1.0 --torsion-step-deg 2.0 --protein-clash-scale 0.85 \
  --selection-mode clash_aware --repulsion-step 0.05 --energy-tolerance-kcal 0.25 \
  --overlap-weight 3.0 --soft-clash-weight 2.0 --split-by-pocket --device "$DEVICE"

echo "Refinement complete. Run the same qvina2.1 and PoseBusters commands on topology_base.csv and balanced.csv, then compare them with scripts/compare_topology_refiner_ab.py."
