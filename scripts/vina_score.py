import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


AD4_TYPES = {
    "C": "C",
    "N": "N",
    "O": "OA",
    "S": "S",
    "P": "P",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
    "B": "B",
    "H": "HD",
}


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


def embed_molecule(smiles, seed=17):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
        except Exception:
            pass
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    return mol


def atom_type(atom):
    symbol = atom.GetSymbol()
    if symbol == "C" and atom.GetIsAromatic():
        return "A"
    return AD4_TYPES.get(symbol, symbol)


def atom_charge(atom):
    if atom.HasProp("_GasteigerCharge"):
        value = atom.GetProp("_GasteigerCharge")
        try:
            charge = float(value)
            if charge == charge:
                return charge
        except ValueError:
            pass
    return 0.0


def write_ligand_pdbqt(mol, path):
    conf = mol.GetConformer()
    lines = ["ROOT"]
    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        charge = atom_charge(atom)
        atype = atom_type(atom)
        name = f"{atom.GetSymbol()}{idx}"
        lines.append(
            f"ATOM  {idx:5d} {name:<4s} LIG     1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:6.3f} {atype:<2s}"
        )
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_meeko_ligand_bin():
    for name in ("mk_prepare_ligand", "mk_prepare_ligand.py"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        Path(sys.executable).parent / "mk_prepare_ligand.py",
        Path(sys.executable).parent / "Scripts" / "mk_prepare_ligand.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def prepare_ligand_pdbqt(mol, path, meeko_bin=None, require_meeko=False):
    meeko_bin = meeko_bin or find_meeko_ligand_bin()
    if meeko_bin:
        sdf_path = Path(path).with_suffix(".sdf")
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        result = subprocess.run(
            [meeko_bin, "-i", str(sdf_path), "-o", str(path), "--rigid_macrocycles"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        output = Path(path)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("Meeko did not create a ligand PDBQT file")
    else:
        if require_meeko:
            raise RuntimeError("mk_prepare_ligand was not found; flexible ligand preparation is required")
        write_ligand_pdbqt(mol, path)


def score_with_python_vina(receptor, ligand, center, size, exhaustiveness, seed, num_modes):
    from vina import Vina

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(str(receptor))
    v.set_ligand_from_file(str(ligand))
    v.compute_vina_maps(center=center, box_size=size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
    energies = v.energies(n_poses=1)
    return float(energies[0][0])


def score_with_cli_vina(
    vina_bin, receptor, ligand, center, size, exhaustiveness, seed, num_modes,
    out_path, log_path, cpu=None,
):
    command = [
        vina_bin,
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--out",
        str(out_path),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--seed",
        str(seed),
        "--num_modes",
        str(num_modes),
    ]
    if cpu is not None:
        command.extend(["--cpu", str(cpu)])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    Path(log_path).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            return float(parts[1])
    detail = (result.stdout + "\n" + result.stderr).strip()
    if len(detail) > 2000:
        detail = detail[-2000:]
    raise RuntimeError(
        "Could not parse Vina affinity from CLI output."
        + (f" Backend output: {detail}" if detail else " Backend output was empty.")
    )


def find_vina_backend(vina_bin):
    if vina_bin:
        return "cli", vina_bin
    try:
        import vina  # noqa: F401

        return "python", None
    except ImportError:
        found = shutil.which("vina")
        if found:
            return "cli", found
    raise RuntimeError(
        "AutoDock Vina is not installed. Install a Vina CLI binary and put vina.exe on PATH, "
        "or install a Python vina package in the pfmol-flow environment."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Add AutoDock Vina scores to a generated molecule CSV.")
    parser.add_argument("--input", required=True, help="CSV containing a smiles column.")
    parser.add_argument("--out", required=True, help="Output CSV with vina_score and high_affinity columns.")
    parser.add_argument("--receptor-pdbqt", required=True, help="Prepared receptor PDBQT file.")
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--center-z", type=float, required=True)
    parser.add_argument("--size-x", type=float, required=True)
    parser.add_argument("--size-y", type=float, required=True)
    parser.add_argument("--size-z", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=-7.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vina-bin", default=None, help="Optional path to vina.exe.")
    parser.add_argument("--meeko-ligand-bin", default=None, help="Optional path to mk_prepare_ligand.exe.")
    return parser.parse_args()


def main():
    args = parse_args()
    receptor = Path(args.receptor_pdbqt)
    if not receptor.exists():
        raise FileNotFoundError(f"receptor PDBQT not found: {receptor}")
    backend, vina_bin = find_vina_backend(args.vina_bin)
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    center = [args.center_x, args.center_y, args.center_z]
    size = [args.size_x, args.size_y, args.size_z]

    with tempfile.TemporaryDirectory(prefix="pfmol_vina_") as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(rows, start=1):
            smiles = row.get("smiles", "")
            mol = embed_molecule(smiles, seed=args.seed + idx)
            if mol is None:
                row["vina_score"] = ""
                row["high_affinity"] = ""
                row["vina_error"] = "failed_to_embed_ligand"
                continue
            ligand_path = tmp_dir / f"ligand_{idx}.pdbqt"
            docked_path = tmp_dir / f"ligand_{idx}_docked.pdbqt"
            log_path = tmp_dir / f"vina_{idx}.log"
            try:
                prepare_ligand_pdbqt(mol, ligand_path, args.meeko_ligand_bin)
                if backend == "python":
                    score = score_with_python_vina(
                        receptor, ligand_path, center, size, args.exhaustiveness, args.seed, args.num_modes
                    )
                else:
                    score = score_with_cli_vina(
                        vina_bin,
                        receptor,
                        ligand_path,
                        center,
                        size,
                        args.exhaustiveness,
                        args.seed,
                        args.num_modes,
                        docked_path,
                        log_path,
                    )
                row["vina_score"] = f"{score:.4f}"
                row["high_affinity"] = "1" if score <= args.threshold else "0"
                row["vina_error"] = ""
            except Exception as exc:
                row["vina_score"] = ""
                row["high_affinity"] = ""
                row["vina_error"] = str(exc)[:500]
            print(f"{idx}/{len(rows)}\tSMILES={smiles}\tVina={row['vina_score'] or row['vina_error']}")

    write_rows(args.out, rows)
    print(f"saved Vina-scored CSV to {args.out}")


if __name__ == "__main__":
    main()
