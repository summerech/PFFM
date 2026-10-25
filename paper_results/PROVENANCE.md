# Unified qvina2.1 result provenance

All methods use the public CrossDocked2020 100-pocket split, a
reference-ligand-centered 20 Angstrom box, and qvina2.1 exhaustiveness 8.
Metrics are first computed per pocket and then macro-averaged.

- Pocket2Mol: 10,000 requested, 9,858 valid and scored; all accepted samples
  map to official source paths, with zero pocket-identity mismatches.
- DecompDiff: 10,000 requested, 8,390 valid connected 3D molecules and 8,199
  scored poses; 100 official result files were audited with zero pocket-
  identity mismatches. Invalid or disconnected outputs remain in the
  denominator for validity and coverage.
- DiffSBDD: 10,000 requested, 10,000 valid molecules and 9,901 scored poses
  across all 100 pockets. The official samples were exported without
  molecule selection and re-docked with the unified protocol.
- MolFORM: 10,000 requested, 8,325 valid connected reconstructions and 8,279
  scored poses across all 100 pockets. All 100 result files matched the public
  split, with zero pocket-identity mismatches; reconstruction and
  ligand-preparation failures remain in the validity and coverage denominators.
- PFFM: three seeds (2021, 2022, 2023), 10,000 requested molecules per
  seed; the table reports the mean of seed-level pocket-macro metrics.
- TargetDiff, MolCRAFT, and IPDiff: previously audited official samples
  re-docked with the same protocol.
- BindDM: omitted from the unified table because no compatible official
  checkpoint or compatible raw-sample archive was available. Published
  values were not substituted.

The CFG sweep uses 10 fixed training-development pockets, three seeds, 30 raw
samples per pocket and scale, and the same docking protocol. It is a mechanism
analysis and was not used to retrospectively select the main-test setting.
