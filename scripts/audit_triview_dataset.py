import argparse
import json

import torch


def quantiles(values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    levels = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {str(float(level)): float(value) for level, value in zip(levels, torch.quantile(tensor, levels))}


def main():
    parser = argparse.ArgumentParser(description="Audit three-view coordinate and energy targets.")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    records = torch.load(args.data, map_location="cpu")["records"]
    raw_local = [
        float((record["raw_pos"] - record["local_pos"]).square().sum(-1).mean().sqrt())
        for record in records
    ]
    local_global = [
        float((record["local_pos"] - record["global_target_pos"]).square().sum(-1).mean().sqrt())
        for record in records
    ]
    counts = {}
    for record in records:
        pocket = str(record["pocket_index"])
        counts[pocket] = counts.get(pocket, 0) + 1
    print(json.dumps({
        "records": len(records),
        "raw_local_rmsd": quantiles(raw_local),
        "local_global_rmsd": quantiles(local_global),
        "records_per_pocket": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
