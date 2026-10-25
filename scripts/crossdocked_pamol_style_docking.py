import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.pamol_style_vina import (
    find_docking_binary,
    load_reference_mol,
    score_mol,
)
from scripts.vina_score import embed_molecule


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


def find_receptor_bin(path=None):
    if path:
        return path
    found = shutil.which("mk_prepare_receptor")
    if found:
        return found
    candidate = Path(sys.executable).parent / "Scripts" / "mk_prepare_receptor.exe"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("mk_prepare_receptor was not found in the current environment")


def resolve_data_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def reference_center_from_sdf(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next((mol for mol in supplier if mol is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"could not read ligand coordinates from {path}")
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    return [
        sum(pos.x for pos in coords) / len(coords),
        sum(pos.y for pos in coords) / len(coords),
        sum(pos.z for pos in coords) / len(coords),
    ]


def disulfide_template_args(pocket_pdb, cutoff=2.5):
    sulfurs = []
    with Path(pocket_pdb).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "SG":
                continue
            if line[17:20].strip() != "CYS":
                continue
            try:
                sulfurs.append(
                    (
                        line[21].strip() or " ",
                        int(line[22:26]),
                        (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                    )
                )
            except ValueError:
                continue
    disulfide = set()
    for index, (chain_a, residue_a, xyz_a) in enumerate(sulfurs):
        for chain_b, residue_b, xyz_b in sulfurs[index + 1 :]:
            distance2 = sum((a - b) ** 2 for a, b in zip(xyz_a, xyz_b))
            if distance2 <= cutoff**2:
                disulfide.add((chain_a, residue_a))
                disulfide.add((chain_b, residue_b))
    by_chain = {}
    for chain, residue in sorted(disulfide):
        by_chain.setdefault(chain, []).append(residue)
    args = []
    for chain, residues in by_chain.items():
        prefix = "" if chain == " " else chain
        args.extend(["-n", f"{prefix}:{','.join(map(str, residues))}=CYX"])
    return args


def normalize_segment_chain_ids(pocket_pdb, output_pdb):
    """Recover chain IDs when a PDB stores distinct chains only as segment IDs."""
    lines = Path(pocket_pdb).read_text(encoding="utf-8", errors="ignore").splitlines()
    residue_segments = {}
    segments = set()
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 76:
            continue
        segment = line[72:76].strip()
        if not segment:
            continue
        key = (line[21:22], line[22:27])
        residue_segments.setdefault(key, set()).add(segment)
        segments.add(segment)
    if not any(len(values) > 1 for values in residue_segments.values()):
        return Path(pocket_pdb)

    chain_lookup = {segment: segment[0] for segment in segments}
    if len(set(chain_lookup.values())) != len(chain_lookup):
        raise RuntimeError(f"segment IDs cannot be mapped uniquely to chains: {segments}")
    normalized = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 76:
            segment = line[72:76].strip()
            if segment:
                line = line[:21] + chain_lookup[segment] + line[22:]
        normalized.append(line)
    output = Path(output_pdb)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return output


def load_generated_mol(row, seed):
    sdf_path = row.get("sdf", "").strip()
    if sdf_path:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            raise RuntimeError(f"could not read generated 3D SDF: {sdf_path}")
        mol = Chem.AddHs(mol, addCoords=True)
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
        return mol
    return embed_molecule(row.get("smiles", ""), seed=seed)


def prepare_receptor_pdbqt(pocket_pdb, reference_ligand, receptor_pdbqt, receptor_bin, box_size, force=False):
    receptor_pdbqt = Path(receptor_pdbqt)
    if receptor_pdbqt.exists() and not force:
        return receptor_pdbqt
    receptor_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    center = reference_center_from_sdf(reference_ligand)
    base = receptor_pdbqt.with_suffix("")
    prepared_pdb = normalize_segment_chain_ids(
        pocket_pdb, receptor_pdbqt.with_suffix(".normalized.pdb")
    )
    command = [
        receptor_bin,
        "--read_pdb",
        str(prepared_pdb),
        "-o",
        str(base),
        "-p",
        str(receptor_pdbqt),
        "--box_center",
        str(center[0]),
        str(center[1]),
        str(center[2]),
        "--box_size",
        str(box_size),
        str(box_size),
        str(box_size),
        "--default_altloc",
        "A",
        "-a",
    ]
    command.extend(disulfide_template_args(prepared_pdb))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or not receptor_pdbqt.exists():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return receptor_pdbqt


def row_key(row):
    return row.get("pocket_pdb") or row.get("receptor_pdb") or row.get("target") or "unknown"


def pdbqt_torsdof(path):
    values = []
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("TORSDOF"):
                try:
                    values.append(int(line.split()[1]))
                except (IndexError, ValueError):
                    pass
    return values[0] if values else None


def parse_args():
    parser = argparse.ArgumentParser(description="CrossDocked pocket10 docking with PAMol-style large-box qvina/vina.")
    parser.add_argument("--input", required=True, help="Generated CSV from sample_crossdocked_test.py.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="data/crossdocked_pocket10/crossdocked_pocket10")
    parser.add_argument("--prepared-root", default="runs/crossdocked_prepared_receptors")
    parser.add_argument(
        "--pose-root",
        default=None,
        help="Persist each best docked PDBQT pose below this directory.",
    )
    parser.add_argument(
        "--prepared-ligand-root",
        default=None,
        help="Persist each Meeko-prepared input PDBQT so docked atoms can be mapped back to the source molecule.",
    )
    parser.add_argument("--box-size", type=float, default=126.0)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=9)
    parser.add_argument("--cpu", type=int, default=None, help="CPU threads allocated to each docking process.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dock-bin", default=None, help="qvina2.1/qvina-w/vina executable. Defaults to PATH, then tools/vina/vina.exe.")
    parser.add_argument("--receptor-bin", default=None)
    parser.add_argument("--meeko-ligand-bin", default=None)
    parser.add_argument("--force-receptor", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    data_root = Path(args.data_root)
    prepared_root = Path(args.prepared_root)
    pose_root = Path(args.pose_root) if args.pose_root else None
    prepared_ligand_root = Path(args.prepared_ligand_root) if args.prepared_ligand_root else None
    vina_bin = find_docking_binary(args.dock_bin)
    receptor_bin = find_receptor_bin(args.receptor_bin)
    size = [args.box_size, args.box_size, args.box_size]

    pocket_cache = {}
    reference_scores = {}
    with tempfile.TemporaryDirectory(prefix="pfmol_crossdocked_vina_") as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(rows, start=1):
            key = row_key(row)
            try:
                pocket_pdb = resolve_data_path(data_root, row["pocket_pdb"])
                reference_ligand = resolve_data_path(data_root, row["reference_ligand_sdf"])
                if key not in pocket_cache:
                    receptor_name = Path(key).with_suffix(".pdbqt").as_posix().replace("/", "__").replace("\\", "__")
                    receptor_pdbqt = prepared_root / receptor_name
                    prepare_receptor_pdbqt(
                        pocket_pdb,
                        reference_ligand,
                        receptor_pdbqt,
                        receptor_bin,
                        args.box_size,
                        force=args.force_receptor,
                    )
                    # Keep scoring centered on the crystallographic ligand. Meeko's
                    # receptor PDBQT does not reliably preserve box-center remarks;
                    # falling back to the receptor atom centroid shifts some pockets
                    # by several angstroms.
                    center = reference_center_from_sdf(reference_ligand)
                    pocket_cache[key] = (receptor_pdbqt, center, reference_ligand)
                receptor_pdbqt, center, reference_ligand = pocket_cache[key]
                if key not in reference_scores:
                    ref_mol = load_reference_mol(reference_ligand)
                    reference_scores[key] = score_mol(
                        vina_bin,
                        receptor_pdbqt,
                        ref_mol,
                        center,
                        size,
                        args.exhaustiveness,
                        args.seed,
                        args.num_modes,
                        tmp_dir,
                        f"reference_{len(reference_scores) + 1}",
                        args.meeko_ligand_bin,
                        cpu=args.cpu,
                    )
                ref_score = reference_scores[key]
                mol = load_generated_mol(row, seed=args.seed + idx)
                if mol is None:
                    raise RuntimeError("failed_to_embed_ligand")
                pose_path = pose_root / f"pose_{idx:06d}.pdbqt" if pose_root else None
                prepared_ligand_path = (
                    prepared_ligand_root / f"ligand_{idx:06d}.pdbqt"
                    if prepared_ligand_root else None
                )
                score = score_mol(
                    vina_bin,
                    receptor_pdbqt,
                    mol,
                    center,
                    size,
                    args.exhaustiveness,
                    args.seed,
                    args.num_modes,
                    tmp_dir,
                    f"ligand_{idx}",
                    args.meeko_ligand_bin,
                    persist_pose=pose_path,
                    persist_ligand=prepared_ligand_path,
                    cpu=args.cpu,
                )
                row["vina_score"] = f"{score:.4f}"
                row["reference_vina_score"] = f"{ref_score:.4f}" if not math.isnan(ref_score) else ""
                row["high_affinity_reference"] = (
                    "1" if not math.isnan(ref_score) and score < ref_score else "0"
                )
                row["high_affinity_fixed_7"] = "1" if score <= -7.0 else "0"
                row["high_affinity"] = row["high_affinity_fixed_7"]
                row["vina_error"] = ""
                row["docking_center_source"] = "reference_ligand_centroid"
                row["docked_pose_pdbqt"] = (
                    str(pose_path) if pose_path else ""
                )
                row["ligand_preparation"] = "meeko_flexible"
                row["torsdof"] = str(pdbqt_torsdof(pose_path)) if pose_path else ""
                row["prepared_ligand_pdbqt"] = (
                    str(prepared_ligand_path) if prepared_ligand_path else ""
                )
            except Exception as exc:
                row["vina_score"] = ""
                row["reference_vina_score"] = ""
                row["high_affinity"] = ""
                row["high_affinity_reference"] = ""
                row["high_affinity_fixed_7"] = ""
                row["vina_error"] = str(exc)[:500]
                row["docking_center_source"] = "reference_ligand_centroid"
                row["docked_pose_pdbqt"] = ""
                row["ligand_preparation"] = "meeko_flexible_failed"
                row["torsdof"] = ""
                row["prepared_ligand_pdbqt"] = ""
            row["docking_backend"] = str(vina_bin)
            if idx == 1 or idx % 10 == 0 or idx == len(rows):
                print(
                    f"{idx}/{len(rows)} pocket={key} vina={row.get('vina_score') or row.get('vina_error')} "
                    f"ref={row.get('reference_vina_score')} ha={row.get('high_affinity')}"
                )

    write_rows(args.out, rows)
    print(f"docking binary: {vina_bin}")
    print(f"prepared receptors: {prepared_root}")
    print(f"saved CrossDocked docking CSV to {args.out}")


if __name__ == "__main__":
    main()
