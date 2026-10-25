import argparse
import csv
import json
import os

import numpy as np

from rdkit import Chem
from rdkit.Chem import QED

from utils.fragment_assembly import assemble_fragment_graph


def load_pockets(config_path, dataset_split):
    if not config_path:
        return None
    import utils.misc as misc
    from datasets import get_dataset

    config = misc.load_config(config_path)
    result = get_dataset(config.data)
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError("Configured dataset does not expose train/test subsets.")
    if dataset_split not in result[1]:
        raise KeyError(f"Configured dataset has no '{dataset_split}' subset.")
    return result[1][dataset_split]


def main():
    parser = argparse.ArgumentParser(description="Assemble coarse fragment-flow samples.")
    parser.add_argument("input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "--dataset-split", choices=("train", "val", "test"), default="test"
    )
    parser.add_argument("--optimize-geometry", action="store_true")
    parser.add_argument("--protein-repulsion", action="store_true")
    parser.add_argument("--geometry-max-iters", type=int, default=500)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    os.makedirs(args.output_dir, exist_ok=True)
    pockets = load_pockets(args.config, args.dataset_split)
    rows = []
    writer = Chem.SDWriter(os.path.join(args.output_dir, "assembled.sdf"))
    for sample_index, sample in enumerate(samples):
        try:
            protein_positions = None
            protein_atomic_numbers = None
            if args.protein_repulsion:
                if pockets is None:
                    raise ValueError("--protein-repulsion requires --config.")
                condition_index = int(
                    sample.get("conditioning_pocket_index", sample["pocket_index"])
                )
                sample_split = sample.get("dataset_split", args.dataset_split)
                if sample_split != args.dataset_split:
                    raise ValueError(
                        f"sample split {sample_split!r} does not match "
                        f"--dataset-split {args.dataset_split!r}"
                    )
                pocket = pockets[condition_index]
                protein_positions = pocket.protein_pos.detach().cpu().numpy()
                if hasattr(pocket, "protein_element"):
                    protein_atomic_numbers = (
                        pocket.protein_element.detach().cpu().numpy().astype(np.int64)
                    )
            molecule, junctions = assemble_fragment_graph(
                sample["fragment_tokens"],
                sample["fragment_positions"],
                sample["edges"],
                random_seed=sample_index,
                optimize_geometry=args.optimize_geometry,
                protein_positions=protein_positions,
                protein_atomic_numbers=protein_atomic_numbers,
                geometry_max_iters=args.geometry_max_iters,
                protein_repulsion=args.protein_repulsion,
            )
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
            molecule.SetProp("pocket_index", str(sample["pocket_index"]))
            molecule.SetProp("smiles", smiles)
            writer.write(molecule)
            rows.append(
                {
                    "pocket_index": sample["pocket_index"],
                    "dataset_split": sample.get("dataset_split", args.dataset_split),
                    "valid": 1,
                    "smiles": smiles,
                    "num_fragments": len(sample["fragment_tokens"]),
                    "num_junctions": len(junctions),
                    "qed": QED.qed(molecule),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "pocket_index": sample["pocket_index"],
                    "dataset_split": sample.get("dataset_split", args.dataset_split),
                    "valid": 0,
                    "smiles": "",
                    "num_fragments": len(sample["fragment_tokens"]),
                    "num_junctions": 0,
                    "qed": "",
                    "error": str(error),
                }
            )
    writer.close()
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
