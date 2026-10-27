# PFFM Paper Code

This archive contains the code paths used by the manuscript **PFFM: Protein-Aware Fragment Flow Matching for Structure-Based Drug Design**.

## Included paper components

1. **Protein-aware pocket encoder**: frozen ESM2-T6 residue embeddings, a 5 A residue hypergraph, and atom-level equivariant pocket messages.
2. **r-BRICS fragment representation**: vocabulary construction, LMDB cache construction, fragment property pretraining, and fragment graph assembly.
3. **Joint fragment flow**: continuous fragment-center flow plus discrete fragment identity, attachment-capacity, and edge-state prediction.
4. **Controllable sampling**: heavy-atom/QED classifier-free guidance with the manuscript setting `CFG=2.0`.
5. **Chemical constraints**: capacity-aware connected decoding and attachment/valency-aware assembly.
6. **Unified evaluation**: qvina2.1 redocking, HA@-7, HA-reference, QED, SA, LogP, Lipinski, Diversity, Validity, Uniqueness, and scored coverage.
7. **Physical refinement**: the topology-preserving refiner and the frozen balanced rigid/torsion energy-guidance configuration.
8. **Paper analyses**: ESM2/hypergraph, no-CFG, and no-constraint ablations; CFG Pareto sweep; pocket-encoder efficiency; and the fixed-test refinement comparison.

## Repository layout

```text
configs/                 CrossDocked/PFFM model configuration
datasets/                CrossDocked and fragment-cache loaders
models/                  Final PFFM flow and physical-refinement models
scripts/                 Training, sampling, assembly, docking, and evaluation
utils/                   Protein/ligand transforms, r-BRICS, assembly, metrics
paper_results/           Machine-readable values reported by the paper
checkpoints/             Verified final model and ablation weights
artifacts/               Exact split, vocabulary, raw poses, and docking records
docs/                    Data layout and reproducibility notes
```

## Environment

The reported model was trained and sampled with Python 3.10.20, PyTorch
1.12.1+cu113, and PyTorch Geometric 2.2.0. Docking preparation used a separate
Python 3.10.20 environment with Meeko 0.7.1 and RDKit 2026.3.4. These are the
observed experiment versions, not the newer PyTorch environment installed on
the local Windows workstation.

```bash
conda env create -f environment.yml
conda env create -f environment-docking.yml
conda run -n pffm-paper python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

`environment.yml` pins the CUDA 11.3 PyTorch wheel and its matching PyG wheels.
The docking environment supplies the Meeko preparation commands. For the
convenience runner, set `PYTHON` to the model environment's Python and
`DOCKING_PYTHON` to the docking environment's Python; the latter defaults to
`PYTHON` for an existing combined environment. For example, on Linux:

```bash
PYTHON=/path/to/envs/pffm-paper/bin/python \
DOCKING_PYTHON=/path/to/envs/pffm-docking/bin/python \
QVINA_BIN=/path/to/qvina2.1 bash scripts/run_paper_generation.sh
```

qvina2.1 is an external executable, not a Conda/Python dependency. Set
`QVINA_BIN` to its location. CUDA 11.3-compatible NVIDIA drivers and Linux
x86-64 were used for the reported run; other platforms may need different
binary wheels.

## Required data and checkpoints

See `docs/DATA.md`, `checkpoints/README.md`, and `artifacts/README.md`. The final weights, vocabulary, public split, 30,000 main-run raw poses, and per-molecule docking records are included. The raw CrossDocked dataset, ESM2 upstream weights, prepared receptors, and docked pose files are not duplicated; obtain or regenerate them as described in the artifact notes.

### External sources

- **CrossDocked2020 v1.1 raw structures:** [official University of Pittsburgh archive](https://bits.csb.pitt.edu/files/crossdock2020/v1.1/). Use v1.1 rather than the newer v1.3 for this experiment.
- **Processed pocket10 training data and test structures:** follow the [official TargetDiff data instructions](https://github.com/guanjq/targetdiff/blob/main/README.md#data) to download `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` and `test_set.zip`. The exact PFFM split is already included at `artifacts/metadata/crossdocked_pocket10_pose_split.pt`.
- **Pretrained protein encoder:** [Meta's Hugging Face ESM2-T6 model](https://huggingface.co/facebook/esm2_t6_8M_UR50D), pinned to [revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`](https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/c731040fcd8d73dceaa04b0a8e6329b345b0f5df). The encoder loads from the local Hugging Face cache, so download it before offline sampling.

## Main paper configuration

- Public CrossDocked2020 pocket10 split: 100 test pockets.
- Three seeds: 2021, 2022, and 2023.
- 100 requested molecules per pocket and seed.
- 100 Euler flow steps.
- Target heavy atoms: 28; target QED: 0.8.
- CFG scale: 2.0; property dropout during training: 0.15.
- Token temperature: 0.9.
- Pocket encoder: `esm2_residue_hypergraph`.
- Docking: qvina2.1, reference-ligand-centered 20 A cube, exhaustiveness 8, 9 modes, seed 17.

Run the final generation protocol with:

```bash
bash scripts/run_paper_generation.sh
```

The script is resumable by per-seed marker files and uses environment variables for machine-specific paths. It performs raw generation, fragment assembly, unified docking, and pocket-macro aggregation without Vina-based selection or reranking. For the exact eight-shard seed schedule of the reported run, see `artifacts/metadata/run_pocket_encoder_esm2_full100.sh` and `artifacts/metadata/run_pfmol_formal_ablation.sh`.

Validate the released files against the reported three-seed result with `python scripts/verify_release_artifacts.py`.

## Ablations

```bash
bash scripts/run_paper_ablations.sh
```

This reproduces only the ablations retained in the manuscript:

- full PFFM;
- without ESM2 + residue hypergraph;
- without CFG;
- without connectivity/junction constraints.

## Physical refinement

The manuscript reports a frozen one-shot comparison on five held-out pockets. Configure the checkpoint and input paths, then run:

```bash
bash scripts/run_paper_refinement.sh
```

The balanced parameters are frozen in the script and match the manuscript.

## Provenance

`paper_results/PROVENANCE.md` records sample counts and the unified docking protocol. The JSON/CSV files in `paper_results/` are copied from the final manuscript package. No published baseline values are substituted when official samples are unavailable.

## Upstream notice

The dataset loaders and selected utility structure originated from the public TargetDiff codebase and are distributed with its license in `LICENSE_TARGETDIFF`. PFFM-specific fragment flow, multiscale protein encoding, constrained decoding, and refinement extensions are identified in `docs/CODE_MAP.md`.
