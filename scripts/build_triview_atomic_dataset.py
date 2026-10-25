import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_atomic_affinity_dataset import parse_pdbqt, resolve
from scripts.build_atomic_refiner_dataset import align_flexible_pose, load_raw


def make_view(record, position):
    return {
        "protein_z": record["protein_z"],
        "protein_pos": record["protein_pos"],
        "ligand_z": record["ligand_z"],
        "ligand_pos": position,
    }


def main():
    parser = argparse.ArgumentParser(description="Build raw/local/global atom-level three-view records.")
    parser.add_argument("--coordinate-data", required=True)
    parser.add_argument("--local-csv", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-strict-order", action="store_true")
    parser.add_argument("--max-raw-local-rmsd", type=float, default=None)
    args = parser.parse_args()

    payload = torch.load(args.coordinate_data, map_location="cpu")
    with Path(args.local_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        row["docked_pose_pdbqt"]: row
        for row in rows
        if row.get("local_vina_score", "").strip() and row.get("raw_score_only", "").strip()
    }
    records, errors, filtered = [], {}, {}
    for source in payload["records"]:
        try:
            row = lookup.get(source["docked_pose_pdbqt"])
            if row is None:
                raise KeyError("missing local-minimum row")
            molecule, raw_z, raw_absolute = load_raw(resolve(row["sdf"], [args.path_root, Path(args.local_csv).parent]))
            prepared_z, prepared_pos = parse_pdbqt(resolve(row["prepared_raw_pdbqt"], [args.path_root, Path(args.local_csv).parent]))
            local_z, local_pos = parse_pdbqt(resolve(row["local_pose_pdbqt"], [args.path_root, Path(args.local_csv).parent]))
            mapped_z, mapped_local = align_flexible_pose(raw_z, raw_absolute, prepared_z, prepared_pos, local_z, local_pos)
            if not torch.equal(mapped_z, source["ligand_z"]):
                raise ValueError("local/source atom order mismatch")
            center = raw_absolute.mean(0)
            local_centered = mapped_local - center
            raw_score = float(row["raw_score_only"])
            local_score = float(row["local_vina_score"])
            global_score = float(source["vina_score"])
            if args.require_strict_order and not global_score < local_score < raw_score:
                filtered["non_strict_energy_order"] = filtered.get("non_strict_energy_order", 0) + 1
                continue
            raw_local_rmsd = float(
                (source["raw_pos"] - local_centered).square().sum(-1).mean().sqrt()
            )
            if args.max_raw_local_rmsd is not None and raw_local_rmsd > args.max_raw_local_rmsd:
                filtered["raw_local_rmsd"] = filtered.get("raw_local_rmsd", 0) + 1
                continue
            record = dict(source)
            record["global_target_pos"] = source["target_pos"]
            record["target_pos"] = local_centered
            record["local_pos"] = local_centered
            record["raw_score_only"] = raw_score
            record["local_vina_score"] = local_score
            record["global_vina_score"] = global_score
            record["raw_local_rmsd"] = raw_local_rmsd
            record["raw_view"] = make_view(record, record["raw_pos"])
            record["local_view"] = make_view(record, local_centered)
            record["global_view"] = make_view(record, source["target_pos"])
            records.append(record)
        except Exception as exc:
            key = f"{type(exc).__name__}: {exc}"
            errors[key] = errors.get(key, 0) + 1
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records}, output)
    if records:
        raw_local = torch.tensor([(r["raw_pos"] - r["local_pos"]).square().sum(-1).mean().sqrt() for r in records])
        local_global = torch.tensor([(r["local_pos"] - r["global_target_pos"]).square().sum(-1).mean().sqrt() for r in records])
        ordering = torch.tensor([r["global_vina_score"] < r["local_vina_score"] < r["raw_score_only"] for r in records]).float()
        summary = {"records": len(records), "errors": errors, "filtered": filtered, "raw_local_rmsd_mean": float(raw_local.mean()), "raw_local_rmsd_max": float(raw_local.max()), "local_global_rmsd_mean": float(local_global.mean()), "strict_energy_order_fraction": float(ordering.mean())}
    else:
        summary = {"records": 0, "errors": errors, "filtered": filtered}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
