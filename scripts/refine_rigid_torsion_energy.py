import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shared_pose_energy import SharedPoseEnergy
from models.topology_preserving_refiner import physical_terms
from scripts.build_atomic_affinity_dataset import parse_pdb, resolve
from scripts.build_atomic_refiner_dataset import bond_matrix, load_raw
from utils.geometry_reference import build_geometry_reference
from utils.junction_torsion import build_junction_torsions, rotate_subtree


def rotate_rigid(position, rotation):
    angle = rotation.norm()
    if float(angle) < 1e-10:
        return position
    axis = rotation / angle
    center = position.mean(0, keepdim=True)
    vector = position - center
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return center + (
        vector * cosine
        + torch.cross(axis.expand_as(vector), vector, dim=-1) * sine
        + axis * (vector * axis).sum(-1, keepdim=True) * (1.0 - cosine)
    )


def energy_value(model, record, position):
    view = {
        "protein_z": record["protein_z"],
        "protein_pos": record["protein_pos"],
        "ligand_z": record["ligand_z"],
        "ligand_pos": position,
    }
    transformed = model.forward_records([view])[0]
    return transformed, model.inverse_energy(transformed)


def geometry_safe(candidate, baseline_terms, record, reference, protein_clash_scale):
    terms = physical_terms(
        candidate,
        record,
        reference,
        protein_clash_scale=protein_clash_scale,
    )
    safe = bool(
        terms["bond"] <= baseline_terms["bond"] + 1e-6
        and terms["angle"] <= baseline_terms["angle"] + 1e-6
        and terms["intramolecular_clash"]
        <= baseline_terms["intramolecular_clash"] + 1e-7
        and terms["protein_clash_count"]
        <= baseline_terms["protein_clash_count"] + 1e-7
        and terms["protein_overlap_max"]
        <= baseline_terms["protein_overlap_max"] + 1e-7
    )
    return safe, terms


def protein_clash_score(terms, ligand_atoms, overlap_weight=2.0, soft_weight=2.0):
    count = terms["protein_clash_count"] / max(int(ligand_atoms), 1)
    return (
        count
        + float(overlap_weight) * terms["protein_overlap_max"]
        + float(soft_weight) * terms["protein_clash"]
    )


def guided_step(
    model,
    record,
    position,
    baseline_terms,
    reference,
    translation_step,
    rotation_step,
    torsion_step,
    use_torsion,
    protein_clash_scale,
    selection_mode="energy",
    energy_tolerance_kcal=0.0,
    repulsion_step=0.0,
    overlap_weight=2.0,
    soft_clash_weight=2.0,
):
    with torch.enable_grad():
        variable = position.detach().requires_grad_(True)
        transformed, _ = energy_value(model, record, variable)
        gradient = torch.autograd.grad(transformed, variable)[0]
    current_energy = float(transformed.detach())
    current_energy_kcal = float(model.inverse_energy(transformed.detach()))
    with torch.no_grad():
        current_terms = physical_terms(
            position,
            record,
            reference,
            protein_clash_scale=protein_clash_scale,
        )
        current_clash_score = float(
            protein_clash_score(
                current_terms,
                position.size(0),
                overlap_weight,
                soft_clash_weight,
            )
        )
    candidates = []

    translation_gradient = gradient.mean(0)
    translation = -translation_step * F.normalize(translation_gradient, dim=0)
    candidates.append(("translation", position + translation))

    centered = position - position.mean(0, keepdim=True)
    torque = torch.cross(centered, gradient, dim=-1).sum(0)
    rotation = -rotation_step * F.normalize(torque, dim=0)
    candidates.append(("rotation", rotate_rigid(position, rotation)))

    rigid = rotate_rigid(position + translation, rotation)
    candidates.append(("rigid", rigid))

    if selection_mode == "clash_aware" and repulsion_step > 0:
        with torch.enable_grad():
            repulsion_variable = position.detach().requires_grad_(True)
            repulsion_terms = physical_terms(
                repulsion_variable,
                record,
                reference,
                protein_clash_scale=protein_clash_scale,
            )
            repulsion_objective = protein_clash_score(
                repulsion_terms,
                position.size(0),
                overlap_weight,
                soft_clash_weight,
            )
            repulsion_gradient = torch.autograd.grad(
                repulsion_objective, repulsion_variable
            )[0].mean(0)
        if float(repulsion_gradient.norm()) > 1e-10:
            direction = -F.normalize(repulsion_gradient, dim=0)
            for factor in (1.0, 0.5, 0.25):
                candidates.append(
                    (f"repulsion_{factor:g}", position + factor * repulsion_step * direction)
                )

    if use_torsion and record["junction_index"].numel():
        torsion_candidate = rigid
        torsion_gradients = []
        for indices, mask in zip(record["junction_index"], record["rotate_mask"]):
            _, src, dst, _ = indices.tolist()
            axis = F.normalize(position[dst] - position[src], dim=0)
            lever = position[mask] - position[src]
            derivative = torch.cross(axis.expand_as(lever), lever, dim=-1)
            torsion_gradients.append((gradient[mask] * derivative).sum())
        scale = torch.stack(torsion_gradients).abs().median().clamp(min=1e-8)
        for indices, mask, torsion_gradient in zip(
            record["junction_index"], record["rotate_mask"], torsion_gradients
        ):
            _, src, dst, _ = indices.tolist()
            delta = -torsion_step * torch.tanh(torsion_gradient / scale)
            torsion_candidate = rotate_subtree(
                torsion_candidate, src, dst, mask, delta.detach()
            )
        candidates.append(("rigid_torsion", torsion_candidate))

    accepted = []
    with torch.no_grad():
        for mode, candidate in candidates:
            safe, terms = geometry_safe(
                candidate,
                baseline_terms,
                record,
                reference,
                protein_clash_scale,
            )
            candidate_energy, candidate_kcal = energy_value(model, record, candidate)
            candidate_clash_score = float(
                protein_clash_score(
                    terms,
                    position.size(0),
                    overlap_weight,
                    soft_clash_weight,
                )
            )
            if selection_mode == "clash_aware":
                physical_improved = candidate_clash_score < current_clash_score - 1e-8
                physical_non_regression = candidate_clash_score <= current_clash_score + 1e-8
                energy_improved = float(candidate_kcal) < current_energy_kcal - 1e-6
                energy_allowed = float(candidate_kcal) <= current_energy_kcal + energy_tolerance_kcal
                accept = safe and (
                    (physical_improved and energy_allowed)
                    or (energy_improved and physical_non_regression)
                )
                rank = (
                    candidate_clash_score,
                    float(candidate_kcal),
                )
            else:
                accept = safe and float(candidate_energy) < current_energy - 1e-8
                rank = (float(candidate_energy), 0.0)
            if accept:
                accepted.append((rank, mode, candidate.detach(), terms))
    if not accepted:
        return position, None
    _, mode, candidate, terms = min(accepted, key=lambda item: item[0])
    return candidate, {"mode": mode, "terms": terms}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Energy guidance restricted to rigid pose and junction torsions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--translation-step", type=float, default=0.04)
    parser.add_argument("--rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--torsion-step-deg", type=float, default=2.0)
    parser.add_argument("--disable-torsion", action="store_true")
    parser.add_argument("--protein-clash-scale", type=float, default=0.85)
    parser.add_argument("--selection-mode", choices=("energy", "clash_aware"), default="energy")
    parser.add_argument("--energy-tolerance-kcal", type=float, default=0.25)
    parser.add_argument("--repulsion-step", type=float, default=0.05)
    parser.add_argument("--overlap-weight", type=float, default=2.0)
    parser.add_argument("--soft-clash-weight", type=float, default=2.0)
    parser.add_argument("--crop", type=float, default=8.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=256)
    parser.add_argument("--split-by-pocket", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = SharedPoseEnergy(
        hidden_dim=checkpoint["hidden_dim"],
        physical_features=checkpoint.get("physical_features", False),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache, output_rows, failures = {}, [], 0
    mode_counts = {}
    for index, row in enumerate(rows):
        try:
            pocket_name = row["pocket_pdb"]
            if pocket_name not in cache:
                cache[pocket_name] = parse_pdb(resolve(pocket_name, [args.data_root]))
            protein_z, protein_pos = cache[pocket_name]
            molecule, ligand_z, raw_absolute = load_raw(
                resolve(row["sdf"], [args.path_root, Path(args.input).parent])
            )
            distance = torch.cdist(protein_pos, raw_absolute).min(1).values
            pocket_atoms = torch.where(distance <= args.crop)[0]
            if pocket_atoms.numel() > args.max_pocket_atoms:
                pocket_atoms = pocket_atoms[
                    distance[pocket_atoms].argsort()[: args.max_pocket_atoms]
                ]
            if pocket_atoms.numel() < 8:
                raise ValueError("fewer than 8 cropped pocket atoms")
            center = raw_absolute.mean(0)
            raw = raw_absolute - center
            geometry = build_junction_torsions(molecule, raw)
            reference, _, _ = build_geometry_reference(molecule, raw)
            record = {
                "protein_z": protein_z[pocket_atoms].to(args.device),
                "protein_pos": (protein_pos[pocket_atoms] - center).to(args.device),
                "ligand_z": ligand_z.to(args.device),
                "raw_pos": raw.to(args.device),
                "bond_type": bond_matrix(molecule).to(args.device),
                **{
                    key: value.to(args.device) if torch.is_tensor(value) else value
                    for key, value in geometry.items()
                },
            }
            reference = reference.to(args.device)
            baseline_terms = physical_terms(
                record["raw_pos"],
                record,
                reference,
                protein_clash_scale=args.protein_clash_scale,
            )
            with torch.no_grad():
                _, initial_kcal = energy_value(model, record, record["raw_pos"])
            position = record["raw_pos"].clone()
            accepted = 0
            for _ in range(args.steps):
                position, status = guided_step(
                    model,
                    record,
                    position,
                    baseline_terms,
                    reference,
                    args.translation_step,
                    math.radians(args.rotation_step_deg),
                    math.radians(args.torsion_step_deg),
                    not args.disable_torsion,
                    args.protein_clash_scale,
                    args.selection_mode,
                    args.energy_tolerance_kcal,
                    args.repulsion_step,
                    args.overlap_weight,
                    args.soft_clash_weight,
                )
                if status is not None:
                    accepted += 1
                    mode_counts[status["mode"]] = mode_counts.get(status["mode"], 0) + 1
            with torch.no_grad():
                _, final_kcal = energy_value(model, record, position)
            absolute = position.cpu() + center
            conformer = molecule.GetConformer()
            for atom_index, coordinates in enumerate(absolute.tolist()):
                conformer.SetAtomPosition(atom_index, coordinates)
            output_path = (output_dir / f"guided_{index:05d}.sdf").resolve()
            writer = Chem.SDWriter(str(output_path)); writer.write(molecule); writer.close()
            result = dict(row)
            result.update({
                "original_sdf": row["sdf"],
                "sdf": str(output_path),
                "energy_initial_prediction": float(initial_kcal),
                "energy_final_prediction": float(final_kcal),
                "energy_accepted_steps": accepted,
                "energy_attempted_steps": args.steps,
                "energy_guidance_error": "",
            })
        except Exception as error:
            failures += 1
            result = dict(row)
            result["energy_guidance_error"] = f"{type(error).__name__}: {error}"[:500]
        output_rows.append(result)
    output = Path(args.out)
    write_rows(output, output_rows)
    if args.split_by_pocket:
        for pocket in sorted({int(row["pocket_index"]) for row in output_rows}):
            write_rows(
                output.with_name(f"{output.stem}_pocket{pocket}{output.suffix}"),
                [row for row in output_rows if int(row["pocket_index"]) == pocket],
            )
    print(json.dumps({
        "molecules": len(output_rows),
        "failures": failures,
        "accepted_modes": mode_counts,
        "mean_accepted_steps": sum(int(row.get("energy_accepted_steps", 0)) for row in output_rows) / max(len(output_rows), 1),
    }, indent=2))


if __name__ == "__main__":
    main()
