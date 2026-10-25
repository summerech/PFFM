import argparse
import csv
import json
from pathlib import Path

import torch
from rdkit import Chem


def element_number(symbol):
    symbol = symbol.strip().capitalize()
    aliases = {"A": "C", "C": "C", "N": "N", "O": "O", "S": "S", "P": "P",
               "F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "Zn": "Zn", "Mg": "Mg"}
    symbol = aliases.get(symbol, symbol)
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(symbol)
    except RuntimeError:
        return 0


def parse_pdb(path):
    atoms, positions = [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line[:6].strip() != "ATOM":
            continue
        symbol = line[76:78].strip() or line[12:16].strip()[0]
        z = element_number(symbol)
        if z <= 1:
            continue
        atoms.append(z)
        positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not atoms:
        raise ValueError(f"no protein atoms in {path}")
    return torch.tensor(atoms), torch.tensor(positions, dtype=torch.float32)


def parse_pdbqt(path):
    atoms, positions = [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        token = line.split()[-1]
        symbol = "Cl" if token.startswith("Cl") else "Br" if token.startswith("Br") else token[:1]
        z = element_number(symbol)
        if z <= 1:
            continue
        atoms.append(z)
        positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not atoms:
        raise ValueError(f"no ligand atoms in {path}")
    return torch.tensor(atoms), torch.tensor(positions, dtype=torch.float32)


def resolve(path, roots):
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in roots:
        candidate = Path(root) / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser(description="Build atom-level pocket/docked-pose affinity data.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop", type=float, default=8.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    args = parser.parse_args()

    records, errors, seen = [], {}, set()
    pocket_cache = {}
    for input_name in args.inputs:
        with Path(input_name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    score = float(row["vina_score"])
                    pocket_name = row["pocket_pdb"]
                    pose_name = row["docked_pose_pdbqt"]
                    key = (pocket_name, row.get("smiles", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    if pocket_name not in pocket_cache:
                        pocket_path = resolve(pocket_name, [args.data_root])
                        pocket_cache[pocket_name] = parse_pdb(pocket_path)
                    protein_z, protein_pos = pocket_cache[pocket_name]
                    pose_path = resolve(pose_name, [args.pose_root, Path(input_name).parent])
                    ligand_z, ligand_pos = parse_pdbqt(pose_path)
                    distance = torch.cdist(protein_pos, ligand_pos).min(dim=1).values
                    index = torch.where(distance <= args.crop)[0]
                    if index.numel() > args.max_pocket_atoms:
                        index = index[distance[index].argsort()[:args.max_pocket_atoms]]
                    if index.numel() < 8:
                        raise ValueError("fewer than 8 pocket atoms after crop")
                    center = ligand_pos.mean(0)
                    records.append({
                        "protein_z": protein_z[index],
                        "protein_pos": protein_pos[index] - center,
                        "ligand_z": ligand_z,
                        "ligand_pos": ligand_pos - center,
                        "vina_score": score,
                        "high_affinity": float(score <= -7.0),
                        "pocket_pdb": pocket_name,
                        "smiles": row.get("smiles", ""),
                        "pose": str(pose_path),
                    })
                except Exception as exc:
                    name = type(exc).__name__ + ": " + str(exc)
                    errors[name] = errors.get(name, 0) + 1
    pockets = sorted({record["pocket_pdb"] for record in records})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "pockets": pockets}, output)
    print(json.dumps({"records": len(records), "pockets": len(pockets), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
