# Data layout

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
    crossdocked_rbrics_100k.vocab.json
```

The exact public split must be used for the 100-pocket comparison. Do not remove failed pockets. Reference-ligand centers define the docking boxes.

Build the paper fragment artifacts with:

```bash
python scripts/build_fragment_vocab.py --help
python scripts/build_rbrics_cache.py --help
```

The raw CrossDocked dataset and official split are not included in this archive. Verify their licenses before redistribution.
