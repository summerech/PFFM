import argparse
import csv
import json
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem


AFFINITY = re.compile(r"Affinity:\s*([-+0-9.eE]+)")


def receptor_name(pocket_name):
    return Path(pocket_name).with_suffix(".pdbqt").as_posix().replace("/", "__").replace("\\", "__")


def receptor_center(path):
    coordinates = []
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    pass
    if not coordinates:
        raise ValueError("receptor contains no atom coordinates")
    return tuple(sum(axis) / len(coordinates) for axis in zip(*coordinates))


def reference_center(path):
    molecule = next(
        (item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item),
        None,
    )
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("reference ligand has no coordinates")
    coordinates = molecule.GetConformer().GetPositions()
    return tuple(float(coordinates[:, axis].mean()) for axis in range(3))


def prepare_molecule(sdf, output_sdf):
    molecule = next((item for item in Chem.SDMolSupplier(str(sdf), removeHs=False) if item), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("invalid raw SDF")
    molecule = Chem.AddHs(molecule, addCoords=True)
    for atom in molecule.GetAtoms():
        atom.SetNoImplicit(True)
    writer = Chem.SDWriter(str(output_sdf)); writer.write(molecule); writer.close()


def process_one(task):
    index, row, root, data_root, prepared_root, output_root, qvina, meeko, box_size = task
    try:
        raw_sdf = Path(row["sdf"])
        if not raw_sdf.is_absolute():
            raw_sdf = Path(root) / raw_sdf
        receptor = Path(prepared_root) / receptor_name(row["pocket_pdb"])
        if not receptor.exists():
            raise FileNotFoundError(f"missing receptor: {receptor}")
        pocket = int(row["pocket_index"])
        pocket_dir = Path(output_root) / f"pocket_{pocket}"
        pocket_dir.mkdir(parents=True, exist_ok=True)
        prepared = pocket_dir / f"prepared_{index:06d}.pdbqt"
        local_pose = pocket_dir / f"local_{index:06d}.pdbqt"
        with tempfile.TemporaryDirectory(prefix="pfmol_local_min_") as directory:
            hydrogenated = Path(directory) / "raw_h.sdf"
            prepare_molecule(raw_sdf, hydrogenated)
            result = subprocess.run(
                [meeko, "-i", str(hydrogenated), "-o", str(prepared), "--rigid_macrocycles"],
                text=True, capture_output=True, check=False,
            )
            if result.returncode or not prepared.exists():
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        reference_name = row.get("reference_ligand_sdf", "").strip()
        reference = Path(reference_name) if reference_name else None
        if reference is not None and not reference.is_absolute():
            reference = Path(data_root) / reference
        center = (
            reference_center(reference)
            if reference is not None and reference.exists()
            else receptor_center(receptor)
        )
        command = [
            qvina, "--receptor", str(receptor), "--ligand", str(prepared),
            "--local_only", "--out", str(local_pose), "--cpu", "1", "--seed", "17",
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        match = AFFINITY.search(result.stdout + "\n" + result.stderr)
        if result.returncode or not match or not local_pose.exists():
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        output = dict(row)
        output["prepared_raw_pdbqt"] = str(prepared)
        output["local_pose_pdbqt"] = str(local_pose)
        output["local_vina_score"] = f"{float(match.group(1)):.6f}"
        output["local_error"] = ""
        return index, output
    except Exception as exc:
        output = dict(row)
        output.update({"prepared_raw_pdbqt": "", "local_pose_pdbqt": "", "local_vina_score": "", "local_error": f"{type(exc).__name__}: {exc}"[:500]})
        return index, output


def main():
    parser = argparse.ArgumentParser(description="Generate qvina local-minimum poses for three-view training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--data-root", default=".", help="Root for relative reference-ligand paths."
    )
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--qvina", required=True)
    parser.add_argument("--meeko", required=True)
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    Path(args.pose_root).mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            i,
            row,
            str(Path(args.root).resolve()),
            str(Path(args.data_root).resolve()),
            str(Path(args.prepared_root).resolve()),
            str(Path(args.pose_root).resolve()),
            args.qvina,
            args.meeko,
            args.box_size,
        )
        for i, row in enumerate(rows)
    ]
    output = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            index, row = future.result(); output[index] = row
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(f"{completed}/{len(tasks)}", flush=True)
    fields = sorted({key for row in output for key in row})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    scores = [float(row["local_vina_score"]) for row in output if row["local_vina_score"]]
    errors = [row["local_error"] for row in output if row["local_error"]]
    print(json.dumps({"rows": len(output), "success": len(scores), "errors": len(errors), "local_min": min(scores) if scores else None, "local_mean": sum(scores) / len(scores) if scores else None, "local_max": max(scores) if scores else None, "error_examples": errors[:5]}, indent=2))


if __name__ == "__main__":
    main()
