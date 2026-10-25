import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_atomic_affinity_dataset import resolve
from scripts.build_atomic_refiner_dataset import load_raw
from utils.geometry_reference import build_geometry_reference


def main():
    parser = argparse.ArgumentParser(description="Add force-field bond/angle references to refiner records.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = torch.load(args.data, map_location="cpu")
    counts = {"mmff": 0, "uff": 0, "failed": 0}
    records = []
    source_records = payload["records"][:args.limit] if args.limit else payload["records"]
    for source in source_records:
        record = dict(source)
        try:
            path_name = record.get("raw_sdf") or record.get("sdf")
            if not path_name:
                raise KeyError("record has no raw_sdf or sdf")
            molecule, _, absolute = load_raw(resolve(path_name, [args.path_root, Path(args.data).parent]))
            center = absolute.mean(0)
            reference, method, status = build_geometry_reference(
                molecule, absolute - center, max_iters=args.max_iters
            )
            if reference.shape != record["raw_pos"].shape:
                raise ValueError("geometry reference atom count mismatch")
            record["geometry_reference_pos"] = reference
            record["geometry_reference_method"] = method
            record["geometry_reference_status"] = status
            counts[method] = counts.get(method, 0) + 1
        except Exception as error:
            record["geometry_reference_pos"] = record["raw_pos"].clone()
            record["geometry_reference_method"] = "failed"
            record["geometry_reference_error"] = f"{type(error).__name__}: {error}"[:500]
            counts["failed"] += 1
        records.append(record)
    result = dict(payload)
    result["records"] = records
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(json.dumps({"records": len(records), "methods": counts, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
