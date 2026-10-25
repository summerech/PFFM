# Manuscript-to-code map

| Manuscript component | Primary implementation |
|---|---|
| ESM2 + residue hypergraph + atom field | `models/fragment_flow.py` |
| r-BRICS vocabulary and cache | `scripts/build_fragment_vocab.py`, `scripts/build_rbrics_cache.py`, `datasets/fragment_cache.py` |
| Fragment property predictor | `models/fragment_property.py`, `scripts/train_fragment_property.py` |
| Joint coordinate/type/capacity/edge flow | `models/fragment_flow.py`, `scripts/train_fragment_flow.py` |
| CFG and constrained sampling | `scripts/sample_fragment_flow.py` |
| Attachment/valency-aware assembly | `utils/fragment_assembly.py`, `utils/fragmentation.py` |
| Generation metrics | `scripts/evaluate_fragment_samples.py` |
| Unified qvina2.1 protocol | `scripts/crossdocked_pamol_style_docking.py` |
| Pocket-macro aggregation | `scripts/summarize_filtered_pockets.py` |
| Topology-preserving refiner | `models/topology_preserving_refiner.py` |
| Three-view energy surrogate | `models/shared_pose_energy.py`, `scripts/train_triview_pose_energy.py` |
| Balanced rigid/torsion guidance | `scripts/refine_rigid_torsion_energy.py` |
| PoseBusters/raw-pose audit | `scripts/audit_posebusters_geometry.py` |
| Hypergraph efficiency audit | `scripts/benchmark_pocket_encoder_efficiency.py` |

Only the final modules and manuscript ablations are packaged. Historical V1/V2 experiments are intentionally excluded.
