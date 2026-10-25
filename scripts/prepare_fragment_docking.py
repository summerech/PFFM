import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem

import utils.misc as misc
from datasets import get_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare assembled fragment samples for CrossDocked docking.")
    parser.add_argument("config")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument(
        "--summary",
        help="Assembly summary used to retain only valid samples while preserving attempt indices.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dataset-split", choices=("train", "val", "test"), default="test"
    )
    args = parser.parse_args()

    config = misc.load_config(args.config)
    _, subsets = get_dataset(config.data)
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    all_pocket_indices = sorted({int(sample["pocket_index"]) for sample in samples})
    molecules = [mol for mol in Chem.SDMolSupplier(args.sdf, removeHs=False) if mol is not None]
    if args.summary:
        with Path(args.summary).open(newline="", encoding="utf-8") as handle:
            assembly_rows = list(csv.DictReader(handle))
        if len(assembly_rows) != len(samples):
            raise RuntimeError(
                f"Summary/sample mismatch: {len(assembly_rows)} != {len(samples)}"
            )
        samples = [
            sample
            for sample, row in zip(samples, assembly_rows)
            if int(row["valid"])
        ]
    if len(molecules) != len(samples):
        raise RuntimeError(f"SDF/sample mismatch: {len(molecules)} != {len(samples)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pocket_metadata = {}
    for pocket_index in all_pocket_indices:
        data = subsets[args.dataset_split][pocket_index]
        pocket_metadata[pocket_index] = (
            str(data.protein_filename),
            str(data.ligand_filename),
        )
    rows = []
    for index, (sample, molecule) in enumerate(zip(samples, molecules)):
        pocket_index = int(sample["pocket_index"])
        pocket_pdb, reference_ligand = pocket_metadata[pocket_index]
        ligand_path = output_dir / f"ligand_{index:04d}.sdf"
        writer = Chem.SDWriter(str(ligand_path))
        writer.write(molecule)
        writer.close()
        rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "pocket_index": pocket_index,
                "dataset_split": sample.get("dataset_split", args.dataset_split),
                "pocket_pdb": pocket_pdb,
                "reference_ligand_sdf": reference_ligand,
                "sdf": str(ligand_path),
                "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                "generation_time": sample.get("generation_time", ""),
            }
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index", "pocket_index", "dataset_split", "pocket_pdb", "reference_ligand_sdf",
        "sdf", "smiles", "generation_time",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for pocket_index in sorted(pocket_metadata):
        pocket_rows = [row for row in rows if row["pocket_index"] == pocket_index]
        pocket_output = output.with_name(
            f"{output.stem}_pocket{pocket_index}{output.suffix}"
        )
        with pocket_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pocket_rows)
    print(f"prepared {len(rows)} ligands across {len(pocket_metadata)} pockets")
    for pocket_index, metadata in sorted(pocket_metadata.items()):
        print(f"pocket {pocket_index}: receptor={metadata[0]} reference={metadata[1]}")
    print(f"saved docking manifest to {output}")


if __name__ == "__main__":
    main()
