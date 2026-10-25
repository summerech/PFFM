#!/usr/bin/env python3
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def resolve_path(root, value):
    path = Path(str(value))
    return path if path.is_absolute() else Path(root) / path


def load_mol(path):
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = supplier[0] if supplier and len(supplier) else None
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def geometry_records(paths, source):
    bonds, angles = [], []
    loaded = 0
    for path in paths:
        mol = load_mol(path)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        loaded += 1
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
                continue
            value = rdMolTransforms.GetBondLength(conf, a.GetIdx(), b.GetIdx())
            if math.isfinite(value):
                bonds.append((source, str(bond.GetBondType()), value))
        for center in mol.GetAtoms():
            if center.GetAtomicNum() == 1:
                continue
            neighbors = [a.GetIdx() for a in center.GetNeighbors() if a.GetAtomicNum() != 1]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    value = rdMolTransforms.GetAngleDeg(conf, neighbors[i], center.GetIdx(), neighbors[j])
                    if math.isfinite(value):
                        angles.append((source, str(center.GetHybridization()), value))
    return loaded, bonds, angles


def describe(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def summarize_geometry(records, group_index):
    output = {"overall": describe([row[2] for row in records]), "groups": {}}
    for group in sorted({row[group_index] for row in records}):
        output["groups"][group] = describe([row[2] for row in records if row[group_index] == group])
    return output


def posebusters_worker(payload):
    rows, config_file, skip_energy = payload
    from posebusters import PoseBusters

    with open(config_file, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if skip_energy:
        config["modules"] = [m for m in config["modules"] if m.get("function") != "energy_ratio"]
    table = pd.DataFrame(rows)[["mol_pred", "mol_cond"]]
    result = PoseBusters(config=config).bust_table(table, full_report=True).reset_index()
    return result


def boolean_summary(frame):
    preferred = [
        "sanitization", "inchi_convertible", "all_atoms_connected", "bond_lengths",
        "bond_angles", "internal_steric_clash", "aromatic_ring_flatness",
        "double_bond_flatness", "internal_energy", "protein-ligand_maximum_distance",
        "minimum_distance_to_protein", "volume_overlap_with_protein",
    ]
    rates = {}
    selected = []
    for column in preferred:
        if column in frame.columns:
            values = frame[column].dropna()
            rates[column] = float(values.astype(bool).mean()) if len(values) else None
            selected.append(column)
    complete = frame[selected].fillna(False).astype(bool).all(axis=1) if selected else pd.Series(False, index=frame.index)
    empty = pd.Series(np.nan, index=frame.index)
    clashes = pd.to_numeric(frame.get("num_pairwise_clashes_protein", empty), errors="coerce")
    short_bonds = pd.to_numeric(frame.get("number_short_outlier_bonds", empty), errors="coerce").fillna(0)
    long_bonds = pd.to_numeric(frame.get("number_long_outlier_bonds", empty), errors="coerce").fillna(0)
    bad_angles = pd.to_numeric(frame.get("number_outlier_angles", empty), errors="coerce").fillna(0)
    return {
        "n": int(len(frame)),
        "test_pass_rates": rates,
        "all_selected_tests_pass_rate": float(complete.mean()),
        "protein_ligand_collision_rate": float((clashes.fillna(0) > 0).mean()),
        "mean_pairwise_protein_clashes": float(clashes.fillna(0).mean()),
        "median_pairwise_protein_clashes": float(clashes.fillna(0).median()),
        "bond_outlier_molecule_rate": float(((short_bonds + long_bonds) > 0).mean()),
        "angle_outlier_molecule_rate": float((bad_angles > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docking-csv", required=True)
    parser.add_argument("--targetdiff-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-csv")
    parser.add_argument("--train-sample", type=int, default=10000)
    parser.add_argument("--posebusters-config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.docking_csv).sort_values(["pocket_index", "sample_index"])
    if args.stride > 1:
        within_pocket = source.groupby("pocket_index").cumcount()
        source = source.loc[within_pocket.mod(args.stride).eq(0)].reset_index(drop=True)
    rows = []
    for _, row in source.iterrows():
        rows.append({
            "mol_pred": str(resolve_path(args.targetdiff_root, row["sdf"])),
            "mol_cond": str(resolve_path(args.data_root, row["pocket_pdb"])),
            "pocket_index": int(row["pocket_index"]),
            "sample_index": int(row["sample_index"]),
        })

    chunks = [rows[i:i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    payloads = [(chunk, args.posebusters_config, args.skip_energy) for chunk in chunks]
    context = mp.get_context("fork")
    with context.Pool(processes=args.workers) as pool:
        pieces = list(pool.imap_unordered(posebusters_worker, payloads))
    busted = pd.concat(pieces, ignore_index=True)
    meta = pd.DataFrame(rows).rename(columns={"mol_pred": "file"})
    busted = busted.merge(meta, on="file", how="left")
    busted.to_csv(out / "posebusters_details.csv", index=False)

    generated_paths = [row["mol_pred"] for row in rows]
    gen_loaded, gen_bonds, gen_angles = geometry_records(generated_paths, "generated_raw")
    train_loaded, train_bonds, train_angles = 0, [], []
    if args.train_csv:
        train = pd.read_csv(args.train_csv)
        if len(train) > args.train_sample:
            train = train.sample(args.train_sample, random_state=17)
        train_paths = [resolve_path(args.data_root, value) for value in train["ligand_sdf"]]
        train_loaded, train_bonds, train_angles = geometry_records(train_paths, "training_ligand")

    bond_frame = pd.DataFrame(gen_bonds + train_bonds, columns=["source", "bond_type", "length_angstrom"])
    angle_frame = pd.DataFrame(gen_angles + train_angles, columns=["source", "central_hybridization", "angle_degree"])
    bond_frame.to_csv(out / "bond_lengths.csv", index=False)
    angle_frame.to_csv(out / "bond_angles.csv", index=False)
    summary = {
        "protocol": {
            "pose_set": "raw generated poses",
            "posebusters_config": str(args.posebusters_config),
            "energy_test_included": not args.skip_energy,
            "sampling_stride_within_each_pocket": args.stride,
        },
        "posebusters": boolean_summary(busted),
        "geometry": {
            "generated_molecules_loaded": gen_loaded,
            "training_molecules_loaded": train_loaded,
            "bond_lengths": {
                source_name: summarize_geometry([r for r in gen_bonds + train_bonds if r[0] == source_name], 1)
                for source_name in ["generated_raw", "training_ligand"]
            },
            "bond_angles": {
                source_name: summarize_geometry([r for r in gen_angles + train_angles if r[0] == source_name], 1)
                for source_name in ["generated_raw", "training_ligand"]
            },
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
