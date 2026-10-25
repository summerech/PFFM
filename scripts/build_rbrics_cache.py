import argparse
from collections import Counter
import json
import os

import lmdb
import pickle
from torch.utils.data import ConcatDataset
from rdkit import Chem
from tqdm.auto import tqdm

import utils.misc as misc
from datasets import get_dataset
from utils.fragmentation import (
    decompose_rbrics,
    mol_from_targetdiff_data,
    reconstruct_from_fragments,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build an rBRICS fragment cache for a TargetDiff dataset."
    )
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "trainvaltest", "all"), default="train"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--map-size-gb", type=float, default=8.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = misc.load_config(args.config)
    dataset_result = get_dataset(config.data)
    if isinstance(dataset_result, tuple):
        dataset, subsets = dataset_result
        if args.split == "all":
            selected = dataset
        elif args.split == "trainvaltest":
            selected = ConcatDataset([subsets[name] for name in ("train", "val", "test") if len(subsets[name])])
        else:
            selected = subsets[args.split]
    else:
        dataset = selected = dataset_result

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    env = lmdb.open(
        output,
        map_size=int(args.map_size_gb * 1024 ** 3),
        subdir=False,
        readonly=False,
        lock=True,
    )
    fragment_frequency = Counter()
    fragment_counts = []
    junction_counts = []
    failures = []
    total = len(selected) if args.limit is None else min(len(selected), args.limit)
    transaction = env.begin(write=True)
    successful = 0
    for selected_index in tqdm(range(total), desc="rBRICS cache"):
        try:
            data = selected[selected_index]
            mol = mol_from_targetdiff_data(data)
            record = decompose_rbrics(mol)
            rebuilt = reconstruct_from_fragments(mol, record)
            source_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
            rebuilt_smiles = Chem.MolToSmiles(rebuilt, isomericSmiles=False)
            if source_smiles != rebuilt_smiles:
                raise ValueError(
                    f"Reconstruction mismatch: {source_smiles} != {rebuilt_smiles}"
                )
            record["dataset_index"] = int(data.id)
            record["selected_index"] = selected_index
            record["ligand_filename"] = data.ligand_filename
            record["smiles"] = source_smiles
            transaction.put(
                f"id:{record['dataset_index']}".encode("ascii"), pickle.dumps(record)
            )
            successful += 1
            fragment_counts.append(int(record["num_fragments"]))
            junction_counts.append(record["junction_atom_index"].size(1))
            fragment_frequency.update(record["fragment_keys"])
            if successful % 1000 == 0:
                transaction.commit()
                transaction = env.begin(write=True)
        except Exception as error:
            failures.append({"selected_index": selected_index, "error": str(error)})
            if args.strict:
                transaction.abort()
                raise
    transaction.put(b"__len__", pickle.dumps(successful))
    transaction.commit()
    env.sync()
    env.close()

    summary = {
        "num_molecules": successful,
        "mean_fragments": sum(fragment_counts) / max(successful, 1),
        "max_fragments": max(fragment_counts, default=0),
        "single_fragment_rate": sum(count == 1 for count in fragment_counts) / max(successful, 1),
        "mean_junctions": sum(junction_counts) / max(successful, 1),
        "vocabulary_size": len(fragment_frequency),
        "requested": total,
        "successful": successful,
        "failed": len(failures),
        "success_rate": successful / max(total, 1),
        "failures": failures[:100],
    }
    vocab_path = output + ".vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(
            dict(sorted(fragment_frequency.items(), key=lambda item: (-item[1], item[0]))),
            handle,
            indent=2,
        )
    summary_path = output + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved cache to {output}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved vocabulary to {vocab_path}")


if __name__ == "__main__":
    main()
