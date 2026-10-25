import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.vina_score import embed_molecule, prepare_ligand_pdbqt, score_with_cli_vina, write_ligand_pdbqt


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_pdbqt_atom_xyz(line):
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None


def receptor_center_from_pdbqt(path):
    remarks = {}
    coords = []
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            upper = line.upper()
            if upper.startswith("REMARK"):
                tokens = upper.replace("=", " ").split()
                for key in ("CENTER_X", "CENTER_Y", "CENTER_Z"):
                    if key in tokens:
                        idx = tokens.index(key)
                        if idx + 1 < len(tokens):
                            try:
                                remarks[key] = float(tokens[idx + 1])
                            except ValueError:
                                pass
            if line.startswith(("ATOM", "HETATM")):
                xyz = parse_pdbqt_atom_xyz(line)
                if xyz is not None:
                    coords.append(xyz)
    if {"CENTER_X", "CENTER_Y", "CENTER_Z"} <= set(remarks):
        return [remarks["CENTER_X"], remarks["CENTER_Y"], remarks["CENTER_Z"]]
    if not coords:
        raise RuntimeError(f"no ATOM coordinates found in receptor: {path}")
    return [sum(values) / len(coords) for values in zip(*coords)]


def load_reference_mol(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next((mol for mol in supplier if mol is not None), None)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
    else:
        raise ValueError("reference ligand must be .pdb, .sdf, or .mol2")
    if mol is None:
        raise RuntimeError(f"could not read reference ligand: {path}")
    mol = Chem.AddHs(mol, addCoords=True)
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    return mol


def find_docking_binary(path):
    if path:
        return path
    for name in ("qvina2.1", "qvina-w", "vina"):
        found = shutil.which(name)
        if found:
            return found
    if os.name != "nt":
        repo_root = Path(__file__).resolve().parents[1]
        for candidate in (
            repo_root / "tools" / "qvina" / "qvina-master" / "bin" / "qvina2.1",
            repo_root / "tools" / "qvina" / "qvina-master" / "bin" / "qvina-w",
        ):
            if candidate.exists():
                return str(candidate)
    local_vina = Path(__file__).resolve().parents[1] / "tools" / "vina" / "vina.exe"
    if local_vina.exists():
        return str(local_vina)
    raise RuntimeError("no qvina/vina executable found")


def score_mol(
    vina_bin,
    receptor,
    mol,
    center,
    size,
    exhaustiveness,
    seed,
    num_modes,
    tmp_dir,
    stem,
    meeko_bin=None,
    persist_pose=None,
    persist_ligand=None,
    allow_rigid_fallback=False,
    cpu=None,
):
    ligand_path = tmp_dir / f"{stem}.pdbqt"
    docked_path = tmp_dir / f"{stem}_docked.pdbqt"
    log_path = tmp_dir / f"{stem}.log"
    try:
        prepare_ligand_pdbqt(mol, ligand_path, meeko_bin, require_meeko=True)
    except Exception as exc:
        if not allow_rigid_fallback:
            raise RuntimeError(f"Meeko ligand preparation failed for {stem}: {exc}") from exc
        print(f"Meeko ligand preparation failed for {stem}; using rigid fallback writer: {exc}")
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        write_ligand_pdbqt(mol, ligand_path)
    if persist_ligand is not None:
        persist_ligand = Path(persist_ligand)
        persist_ligand.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ligand_path, persist_ligand)
    score = score_with_cli_vina(
        vina_bin,
        receptor,
        ligand_path,
        center,
        size,
        exhaustiveness,
        seed,
        num_modes,
        docked_path,
        log_path,
        cpu=cpu,
    )
    if persist_pose is not None:
        persist_pose = Path(persist_pose)
        persist_pose.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(docked_path, persist_pose)
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="PAMol-style docking: receptor-derived center, large box, reference-relative HA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--receptor-pdbqt", required=True)
    parser.add_argument("--reference-ligand", required=True)
    parser.add_argument("--box-size", type=float, default=126.0)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dock-bin", default=None, help="qvina2.1/qvina-w/vina executable. Defaults to PATH, then tools/vina/vina.exe.")
    parser.add_argument("--meeko-ligand-bin", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    receptor = Path(args.receptor_pdbqt)
    vina_bin = find_docking_binary(args.dock_bin)
    center = receptor_center_from_pdbqt(receptor)
    size = [args.box_size, args.box_size, args.box_size]
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    with tempfile.TemporaryDirectory(prefix="pfmol_pamol_vina_") as tmp:
        tmp_dir = Path(tmp)
        ref_mol = load_reference_mol(args.reference_ligand)
        try:
            ref_score = score_mol(
                vina_bin,
                receptor,
                ref_mol,
                center,
                size,
                args.exhaustiveness,
                args.seed,
                args.num_modes,
                tmp_dir,
                "reference",
                args.meeko_ligand_bin,
            )
        except Exception as exc:
            ref_score = math.nan
            print(f"reference docking failed: {exc}")

        for idx, row in enumerate(rows, start=1):
            smiles = row.get("smiles", "")
            mol = embed_molecule(smiles, seed=args.seed + idx)
            if mol is None:
                row["vina_score"] = ""
                row["high_affinity"] = ""
                row["reference_vina_score"] = f"{ref_score:.4f}" if not math.isnan(ref_score) else ""
                row["vina_error"] = "failed_to_embed_ligand"
                continue
            try:
                score = score_mol(
                    vina_bin,
                    receptor,
                    mol,
                    center,
                    size,
                    args.exhaustiveness,
                    args.seed,
                    args.num_modes,
                    tmp_dir,
                    f"ligand_{idx}",
                    args.meeko_ligand_bin,
                )
                row["vina_score"] = f"{score:.4f}"
                row["reference_vina_score"] = f"{ref_score:.4f}" if not math.isnan(ref_score) else ""
                row["high_affinity"] = "1" if not math.isnan(ref_score) and score < ref_score else "0"
                row["vina_error"] = ""
            except Exception as exc:
                row["vina_score"] = ""
                row["reference_vina_score"] = f"{ref_score:.4f}" if not math.isnan(ref_score) else ""
                row["high_affinity"] = ""
                row["vina_error"] = str(exc)[:500]
            print(
                f"{idx}/{len(rows)}\tVina={row['vina_score'] or row['vina_error']}\t"
                f"Ref={row['reference_vina_score']}\tHA={row['high_affinity']}"
            )

    write_rows(args.out, rows)
    print(f"docking binary: {vina_bin}")
    print(f"center: {center}; size: {size}; reference_vina={ref_score:.4f}" if not math.isnan(ref_score) else f"center: {center}; size: {size}")
    print(f"saved PAMol-style Vina CSV to {args.out}")


if __name__ == "__main__":
    main()
