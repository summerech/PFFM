import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.topology_preserving_refiner import TopologyPreservingAtomicRefiner, physical_terms
from scripts.build_atomic_affinity_dataset import parse_pdb, resolve
from scripts.build_atomic_refiner_dataset import bond_matrix, load_raw
from utils.geometry_reference import build_geometry_reference
from utils.junction_torsion import build_junction_torsions


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Apply topology-preserving atom-level refinement.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop", type=float, default=8.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--physics-step", type=float, default=0.08)
    parser.add_argument("--physics-iterations", type=int, default=2)
    parser.add_argument("--protein-clash-scale", type=float, default=0.75)
    parser.add_argument("--preserve-protein-clearance", action="store_true")
    parser.add_argument("--clearance-tolerance", type=float, default=0.0)
    parser.add_argument("--split-by-pocket", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = TopologyPreservingAtomicRefiner(
        hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"]
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pocket_cache, output_rows, failures = {}, [], 0
    for index, row in enumerate(rows):
        try:
            pocket_name = row["pocket_pdb"]
            if pocket_name not in pocket_cache:
                pocket_cache[pocket_name] = parse_pdb(resolve(pocket_name, [args.data_root]))
            protein_z, protein_pos = pocket_cache[pocket_name]
            raw_path = resolve(row["sdf"], [args.path_root, Path(args.input).parent])
            molecule, ligand_z, raw_pos = load_raw(raw_path)
            distance = torch.cdist(protein_pos, raw_pos).min(1).values
            pocket_atoms = torch.where(distance <= args.crop)[0]
            if pocket_atoms.numel() > args.max_pocket_atoms:
                pocket_atoms = pocket_atoms[distance[pocket_atoms].argsort()[:args.max_pocket_atoms]]
            if pocket_atoms.numel() < 8:
                raise ValueError("fewer than 8 cropped pocket atoms")
            center = raw_pos.mean(0)
            geometry = build_junction_torsions(molecule, raw_pos - center)
            geometry_reference, reference_force_field, reference_status = build_geometry_reference(
                molecule, raw_pos - center
            )
            record = {
                "protein_z": protein_z[pocket_atoms].to(args.device),
                "protein_pos": (protein_pos[pocket_atoms] - center).to(args.device),
                "ligand_z": ligand_z.to(args.device),
                "raw_pos": (raw_pos - center).to(args.device),
                "geometry_reference_pos": geometry_reference.to(args.device),
                "bond_type": bond_matrix(molecule).to(args.device),
                **{key: value.to(args.device) if torch.is_tensor(value) else value for key, value in geometry.items()},
            }
            before = physical_terms(record["raw_pos"], record)
            refined, status = model.refine(
                record, steps=args.steps, strength=args.strength,
                physics_step=args.physics_step, physics_iterations=args.physics_iterations,
                weights=checkpoint.get("physics_weights"),
                protein_clash_scale=args.protein_clash_scale,
                preserve_protein_clearance=args.preserve_protein_clearance,
                clearance_tolerance=args.clearance_tolerance,
            )
            after = physical_terms(refined, record)
            refined = refined.cpu() + center
            conformer = molecule.GetConformer()
            for atom_index, coordinates in enumerate(refined.tolist()):
                conformer.SetAtomPosition(atom_index, coordinates)
            output_path = (output_dir / f"refined_{index:05d}.sdf").resolve()
            writer = Chem.SDWriter(str(output_path))
            writer.write(molecule)
            writer.close()
            result = dict(row)
            result.update({
                "original_sdf": row["sdf"],
                "sdf": str(output_path),
                "refiner_accepted_steps": status["accepted_steps"],
                "refiner_attempted_steps": status["attempted_steps"],
                "raw_protein_clash_rate": float(before["protein_clash_rate"]),
                "refined_protein_clash_rate": float(after["protein_clash_rate"]),
                "raw_physics_bond": float(before["bond"]),
                "refined_physics_bond": float(after["bond"]),
                "refiner_error": "",
                "geometry_reference_force_field": reference_force_field,
                "geometry_reference_status": reference_status,
            })
            output_rows.append(result)
        except Exception as error:
            failures += 1
            result = dict(row)
            result["refiner_error"] = f"{type(error).__name__}: {error}"[:500]
            output_rows.append(result)

    output = Path(args.out)
    write_rows(output, output_rows)
    if args.split_by_pocket:
        pockets = {}
        for row in output_rows:
            pockets.setdefault(int(row["pocket_index"]), []).append(row)
        for pocket, pocket_rows in pockets.items():
            write_rows(output.with_name(f"{output.stem}_pocket{pocket}{output.suffix}"), pocket_rows)
    print(json.dumps({"processed": len(output_rows), "failures": failures, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
