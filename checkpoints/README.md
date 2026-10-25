# Required checkpoints

The following weights are required to reproduce the paper. They are not bundled because of archive size and upstream redistribution constraints.

| Logical name | Default path used by the runners | Purpose |
|---|---|---|
| `PFFM_CHECKPOINT` | `checkpoints/pffm_esm2_residue_hypergraph.pt` | Final ESM2 + residue-hypergraph fragment-flow generator |
| `PROPERTY_CHECKPOINT` | `checkpoints/fragment_property_qed.pt` | Frozen fragment-level QED property predictor |
| `TOPOLOGY_REFINER_CHECKPOINT` | `checkpoints/topology_refiner_balanced.pt` | Topology-preserving atom-level refiner selected on validation pockets |
| `ENERGY_CHECKPOINT` | `checkpoints/triview_energy_physical.pt` | Raw/local/global three-view energy surrogate used by balanced refinement |

The final paper experiment used `facebook/esm2_t6_8M_UR50D`. Hugging Face may download it on first use; for offline servers, populate the Hugging Face cache and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Before public release, add SHA-256 hashes and a stable download location for all four checkpoints. The code deliberately fails when a required checkpoint is absent instead of silently substituting an older model.
