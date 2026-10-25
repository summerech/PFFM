import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation.sascorer import compute_sa_score


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lipinski_count(molecule):
    return sum(
        int(rule)
        for rule in (
            Descriptors.ExactMolWt(molecule) < 500,
            Lipinski.NumHDonors(molecule) <= 5,
            Lipinski.NumHAcceptors(molecule) <= 10,
            -2 <= Crippen.MolLogP(molecule) <= 5,
            Lipinski.NumRotatableBonds(molecule) <= 10,
        )
    )


def diversity(molecules):
    if len(molecules) < 2:
        return 0.0
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048) for molecule in molecules]
    similarity_sum = 0.0
    similarity_count = 0
    for index, fingerprint in enumerate(fingerprints[:-1]):
        values = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :])
        similarity_sum += sum(values)
        similarity_count += len(values)
    return float(1.0 - similarity_sum / similarity_count)


def mean(values):
    return float(np.mean(values)) if values else None


def bootstrap_ci95(values, repeats=10000, seed=17):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repeats, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description="Summarize an explicitly selected pocket subset without molecule reranking.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--input", help="Optional final combined docking CSV. Preferred after retry/repair merging.")
    parser.add_argument("--pockets", required=True)
    parser.add_argument("--model-name", default="PFMol-Flow filtered-pocket analysis")
    parser.add_argument("--requested", type=int, help="Generation attempts, including invalid/unreconstructed samples.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    pockets = [int(value) for value in args.pockets.split(",") if value.strip()]
    if args.input:
        rows = [row for row in read(args.input) if int(row["pocket_index"]) in set(pockets)]
    else:
        rows = []
        for pocket in pockets:
            rows.extend(read(Path(args.root) / "docked_csv" / f"pocket_{pocket}.csv"))

    requested = args.requested or len(rows)
    valid_molecules = []
    canonical = []
    valid_rows = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row.get("smiles", ""))
        if molecule is None:
            continue
        valid_molecules.append(molecule)
        canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        valid_rows.append(row)

    scored = [row for row in valid_rows if row.get("vina_score", "").strip()]
    scores = [float(row["vina_score"]) for row in scored]
    fixed = [int(row["high_affinity_fixed_7"]) for row in scored]
    relative = [int(row["high_affinity_reference"]) for row in scored]
    generation_times = [float(row["generation_time"]) for row in valid_rows if row.get("generation_time", "").strip()]

    grouped = defaultdict(list)
    for row in scored:
        grouped[int(row["pocket_index"])].append(row)
    per_pocket = []
    for pocket in pockets:
        pocket_rows = grouped[pocket]
        pocket_scores = [float(row["vina_score"]) for row in pocket_rows]
        per_pocket.append(
            {
                "pocket": pocket,
                "count": len(pocket_rows),
                "vina": mean(pocket_scores),
                "ha_fixed_7": mean([int(row["high_affinity_fixed_7"]) for row in pocket_rows]),
                "ha_reference": mean([int(row["high_affinity_reference"]) for row in pocket_rows]),
                "reference_vina": float(pocket_rows[0]["reference_vina_score"]) if pocket_rows else None,
            }
        )

    result = {
        "selection": {
            "pockets": pockets,
            "excluded_pockets": sorted(set(range(100)) - set(pockets)),
            "selection_type": "full_100_pocket" if len(pockets) == 100 else "post_hoc_pocket_subset",
            "molecule_reranking": False,
        },
        "requested": requested,
        "valid": len(valid_molecules),
        "scored": len(scored),
        "validity": len(valid_molecules) / requested if requested else None,
        "docking_success": len(scored) / len(valid_rows) if valid_rows else None,
        "overall_scored_rate": len(scored) / requested if requested else None,
        "uniqueness": len(set(canonical)) / len(canonical) if canonical else None,
        "vina_micro": mean(scores),
        "vina_macro": mean([row["vina"] for row in per_pocket]),
        "ha_fixed_7_micro": mean(fixed),
        "ha_fixed_7_macro": mean([row["ha_fixed_7"] for row in per_pocket]),
        "ha_reference_micro": mean(relative),
        "ha_reference_macro": mean([row["ha_reference"] for row in per_pocket]),
        "pocket_bootstrap_ci95": {
            "vina": bootstrap_ci95([row["vina"] for row in per_pocket], seed=17),
            "ha_fixed_7": bootstrap_ci95([row["ha_fixed_7"] for row in per_pocket], seed=18),
            "ha_reference": bootstrap_ci95([row["ha_reference"] for row in per_pocket], seed=19),
        },
        "qed": mean([QED.qed(molecule) for molecule in valid_molecules]),
        "sa": mean([compute_sa_score(molecule) for molecule in valid_molecules]),
        "logp": mean([Crippen.MolLogP(molecule) for molecule in valid_molecules]),
        "lipinski": mean([lipinski_count(molecule) for molecule in valid_molecules]),
        "diversity": diversity(valid_molecules),
        "generation_time_total_seconds": float(sum(generation_times)),
        "generation_time_mean_seconds": mean(generation_times),
        "per_pocket": per_pocket,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary_row = {
        "Models": args.model_name,
        "Pockets": ",".join(map(str, pockets)),
        "N": len(scored),
        "Vina Score": result["vina_micro"],
        "Vina Macro": result["vina_macro"],
        "HA@-7": result["ha_fixed_7_micro"],
        "HA-reference": result["ha_reference_micro"],
        "QED": result["qed"],
        "SA": result["sa"],
        "LogP": result["logp"],
        "Lipinski": result["lipinski"],
        "Diversity": result["diversity"],
        "Validity": result["validity"],
        "Uniqueness": result["uniqueness"],
        "Time total (s)": result["generation_time_total_seconds"],
    }
    with Path(args.csv_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row))
        writer.writeheader()
        writer.writerow(summary_row)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
