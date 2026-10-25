# Experiment artifacts

`metadata/` stores the final sampling vocabulary, public pocket10 split, model configuration, and original main-run wrappers. `pffm_100pocket/` stores one archive per main seed and an archive of strict no-CFG/no-constraints ablation scores plus frozen five-pocket refinement records.

| Main seed | Requested | Valid | qvina2.1 scored | Pocket-macro Vina Dock (kcal/mol) |
|---|---:|---:|---:|---:|
| 2021 | 10,000 | 10,000 | 9,988 | -8.341317 |
| 2022 | 10,000 | 10,000 | 9,987 | -8.336360 |
| 2023 | 10,000 | 10,000 | 9,991 | -8.345112 |

The three-seed mean is -8.340930 kcal/mol, matching `paper_results/esm2_full100_three_seed_summary.json`. Each main archive includes `samples.json`, `assembled/assembled.sdf`, `assembled/summary.csv`, `manifest.csv`, `docked.csv`, `metrics.json`, and `metrics.csv`. All three docking tables have exactly 100 pockets x 100 distinct sample identifiers. The ablation archive includes per-molecule docking tables for no-CFG and no-constraints, seeds 2021-2023, and the fixed-test refinement comparison.

**Scope:** The archives preserve raw generated poses and docking scores, but not the roughly 265 MB/seed of docked PDBQT poses or prepared receptor binaries. The raw CrossDocked dataset and upstream ESM2 weights are also not mirrored. Re-docking requires obtaining the public structures and regenerating receptor preparations under the documented qvina2.1 protocol. Server-absolute paths in original CSV files are provenance fields, not portable paths.

Run `python scripts/verify_release_artifacts.py` to check archive completeness, pocket/sample identity, SHA-256 hashes, and agreement with the reported mean.
