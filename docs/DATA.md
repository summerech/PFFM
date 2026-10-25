# Data layout

The exact public 100-pocket split is supplied as `artifacts/metadata/crossdocked_pocket10_pose_split.pt`; the exact sampling vocabulary is `artifacts/metadata/crossdocked_rbrics_dummy_100k.vocab.json`. The historical `dummy` basename is the actual file used by the final run. Do not substitute the similarly named `crossdocked_rbrics_100k.vocab.json`.

PFFM uses the processed CrossDocked2020 pocket10 data and the public Pocket2Mol/TargetDiff split.

Expected files:

```text
data/
  crossdocked_v1.1_rmsd1.0_pocket10/
    crossdocked_pocket10/
    crossdocked_v1.1_rmsd1.0_processed.lmdb
    crossdocked_v1.1_rmsd1.0_processed_name2id.pt
  crossdocked_pocket10_pose_split.pt
  processed/
    crossdocked_rbrics_100k.lmdb
    crossdocked_rbrics_dummy_100k.vocab.json
```

Copy the supplied split into the configuration's expected `data/crossdocked_pocket10_pose_split.pt` path before sampling. The processed CrossDocked LMDB, pocket structures, prepared receptors, and fragment cache are not included; obtain or build them from the public source data. Set `DATA_ROOT`, `QVINA_BIN`, `RECEPTOR_BIN`, and `LIGAND_BIN` for your installation. The exact main-run eight-shard wrappers are archived in `artifacts/metadata/`. The simplified `run_paper_generation.sh` uses a single-shard random-number schedule, so its samples need not be bit-identical to the released samples. Reference-ligand centers define the 20 A docking boxes.

Build the paper fragment artifacts with:

```bash
python scripts/build_fragment_vocab.py --help
python scripts/build_rbrics_cache.py --help
```

The raw CrossDocked dataset is not included. Verify its license before redistribution. Server-absolute paths in original CSV files are provenance fields, not portable paths; join records by pocket and sample identifiers.
