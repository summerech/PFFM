import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_atomic_affinity_dataset import parse_pdb, parse_pdbqt, resolve
from utils.junction_torsion import build_junction_torsions


def bond_matrix(molecule):
    matrix = torch.zeros(molecule.GetNumAtoms(), molecule.GetNumAtoms(), dtype=torch.long)
    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    for bond in molecule.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        matrix[src, dst] = matrix[dst, src] = mapping.get(bond.GetBondType(), 1)
    return matrix


def load_raw(path):
    molecule = next((mol for mol in Chem.SDMolSupplier(str(path), removeHs=True) if mol), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("invalid raw SDF")
    return (
        molecule,
        torch.tensor([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long),
        torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32),
    )


def align_flexible_pose(raw_z, raw_pos, prepared_z, prepared_pos, docked_z, docked_pos):
    if not torch.equal(prepared_z, docked_z):
        raise ValueError("prepared/docked PDBQT atom order mismatch")
    if raw_z.numel() != prepared_z.numel():
        raise ValueError("raw/prepared heavy atom count mismatch")
    unused = set(range(prepared_z.numel()))
    raw_to_prepared = []
    for raw_index, atomic_number in enumerate(raw_z.tolist()):
        candidates = [index for index in unused if int(prepared_z[index]) == atomic_number]
        if not candidates:
            raise ValueError("raw/prepared element mismatch")
        distances = torch.stack(
            [(prepared_pos[index] - raw_pos[raw_index]).norm() for index in candidates]
        )
        best = candidates[int(distances.argmin())]
        if float(distances.min()) > 0.15:
            raise ValueError("raw/prepared coordinate mapping exceeded 0.15 A")
        raw_to_prepared.append(best)
        unused.remove(best)
    return docked_z[raw_to_prepared], docked_pos[raw_to_prepared]


def main():
    parser = argparse.ArgumentParser(description="Build raw-to-docked coordinate-flow records.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop", type=float, default=8.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    args = parser.parse_args()
    records, errors, pocket_cache = [], {}, {}
    for input_name in args.input:
        with Path(input_name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    pocket_name = row["pocket_pdb"]
                    if pocket_name not in pocket_cache:
                        pocket_cache[pocket_name] = parse_pdb(
                            resolve(pocket_name, [args.data_root])
                        )
                    protein_z, protein_pos = pocket_cache[pocket_name]
                    raw_path = resolve(
                        row["sdf"], [args.path_root, Path(input_name).parent]
                    )
                    docked_path = resolve(
                        row["docked_pose_pdbqt"],
                        [args.path_root, Path(input_name).parent],
                    )
                    molecule, raw_z, raw_pos = load_raw(raw_path)
                    docked_z, docked_pos = parse_pdbqt(docked_path)
                    prepared_name = row.get("prepared_ligand_pdbqt", "").strip()
                    if prepared_name:
                        prepared_path = resolve(
                            prepared_name, [args.path_root, Path(input_name).parent]
                        )
                        prepared_z, prepared_pos = parse_pdbqt(prepared_path)
                        docked_z, docked_pos = align_flexible_pose(
                            raw_z,
                            raw_pos,
                            prepared_z,
                            prepared_pos,
                            docked_z,
                            docked_pos,
                        )
                    if not torch.equal(raw_z, docked_z):
                        raise ValueError("raw/docked atom order mismatch")
                    distance = torch.minimum(
                        torch.cdist(protein_pos, raw_pos).min(1).values,
                        torch.cdist(protein_pos, docked_pos).min(1).values,
                    )
                    pocket_index = torch.where(distance <= args.crop)[0]
                    if pocket_index.numel() > args.max_pocket_atoms:
                        pocket_index = pocket_index[
                            distance[pocket_index].argsort()[: args.max_pocket_atoms]
                        ]
                    if pocket_index.numel() < 8:
                        raise ValueError("fewer than 8 cropped pocket atoms")
                    center = raw_pos.mean(0)
                    junctions = build_junction_torsions(
                        molecule, raw_pos - center, docked_pos - center
                    )
                    records.append(
                        {
                            "protein_z": protein_z[pocket_index],
                            "protein_pos": protein_pos[pocket_index] - center,
                            "ligand_z": raw_z,
                            "raw_pos": raw_pos - center,
                            "target_pos": docked_pos - center,
                            "bond_type": bond_matrix(molecule),
                            "pocket_index": int(row["pocket_index"]),
                            "dataset_split": row.get("dataset_split", ""),
                            "pocket_pdb": pocket_name,
                            "vina_score": float(row["vina_score"]),
                            "smiles": row.get("smiles", ""),
                            "raw_sdf": row["sdf"],
                            "docked_pose_pdbqt": row["docked_pose_pdbqt"],
                            **junctions,
                        }
                    )
                except Exception as exc:
                    key = f"{type(exc).__name__}: {exc}"
                    errors[key] = errors.get(key, 0) + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "pockets": sorted(pocket_cache)}, output)
    print(json.dumps({"records": len(records), "pockets": len(pocket_cache), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
