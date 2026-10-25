# Released checkpoints

These are byte-for-byte copies of the weights used in the reported experiments. SHA-256 values are also listed in `artifacts/SHA256SUMS.txt`.

| Logical name | Default path used by the runners | Purpose |
|---|---|---|
| `PFFM_CHECKPOINT` | `checkpoints/pffm_esm2_residue_hypergraph.pt` | Final ESM2 + residue-hypergraph fragment-flow generator |
| `ATOM_ONLY_CHECKPOINT` | `checkpoints/pffm_atom_only.pt` | Atom-only representation ablation |
| `PROPERTY_CHECKPOINT` | `checkpoints/fragment_property_qed.pt` | Frozen fragment-level QED property predictor |
| `TOPOLOGY_REFINER_CHECKPOINT` | `checkpoints/topology_refiner_balanced.pt` | Topology-preserving atom-level refiner selected on validation pockets |
| `ENERGY_CHECKPOINT` | `checkpoints/triview_energy_physical.pt` | Raw/local/global three-view energy surrogate used by balanced refinement |

The final paper experiment used `facebook/esm2_t6_8M_UR50D`. Hugging Face may download it on first use; for offline servers, populate the Hugging Face cache and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The released files originate from `pocket_encoder_esm2/checkpoints/esm2_residue_hypergraph.pt`, `pocket_encoder_ablation/checkpoints/atom_only.pt`, `fragment_property_qed_100k_10k.pt`, `topology_refiner_v30_validation/v30b.pt`, and `energy_proxy_v32_10k/energy_10k_physical.pt`, respectively. Raw CrossDocked pocket10 structures and qvina2.1 remain separate prerequisites.
