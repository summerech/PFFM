import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED

from utils.evaluation.sascorer import compute_sa_score


def lipinski_count(mol):
    rules = (
        Descriptors.ExactMolWt(mol) < 500,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        -2 <= Crippen.MolLogP(mol) <= 5,
        Lipinski.NumRotatableBonds(mol) <= 10,
    )
    return sum(int(rule) for rule in rules)


def internal_diversity(mols):
    if len(mols) < 2:
        return 0.0
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    similarities = []
    for i, fp in enumerate(fingerprints):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fp, fingerprints[i + 1 :]))
    return float(1.0 - np.mean(similarities))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--requested", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requested = args.requested or len(rows)

    mols = []
    smiles = []
    for row in rows:
        if int(row.get("valid", 0)) != 1:
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        mols.append(mol)
        smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))

    if not mols:
        raise RuntimeError("No valid molecules were found")

    metrics = {
        "requested": requested,
        "valid": len(mols),
        "unique": len(set(smiles)),
        "validity": len(mols) / requested,
        "uniqueness": len(set(smiles)) / len(mols),
        "qed": float(np.mean([QED.qed(mol) for mol in mols])),
        "sa": float(np.mean([compute_sa_score(mol) for mol in mols])),
        "logp": float(np.mean([Crippen.MolLogP(mol) for mol in mols])),
        "lipinski": float(np.mean([lipinski_count(mol) for mol in mols])),
        "diversity": internal_diversity(mols),
        "mean_mw": float(np.mean([Descriptors.ExactMolWt(mol) for mol in mols])),
        "mean_heavy_atoms": float(np.mean([mol.GetNumHeavyAtoms() for mol in mols])),
        "std_heavy_atoms": float(np.std([mol.GetNumHeavyAtoms() for mol in mols])),
        "mean_fragments": float(np.mean([int(row["num_fragments"]) for row in rows])),
    }

    if args.samples:
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
        times = [float(sample["generation_time"]) for sample in samples]
        metrics["generation_time_total_seconds"] = float(sum(times))
        metrics["generation_time_mean_seconds"] = float(np.mean(times))
        metrics["predicted_qed_mean"] = float(np.mean([sample["qed_prediction"] for sample in samples]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
