import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scored_lookup(path):
    result = {}
    for row in load_csv(path):
        if not row.get("vina_score", "").strip():
            continue
        key = (int(row["pocket_index"]), int(row["sample_index"]))
        result[key] = row
    return result


def summarize(rows):
    pockets = sorted({int(row["pocket_index"]) for row in rows})
    per_pocket = {}
    for pocket in pockets:
        values = [row for row in rows if int(row["pocket_index"]) == pocket]
        vina = [float(row["vina_score"]) for row in values]
        per_pocket[str(pocket)] = {
            "n": len(values),
            "vina": float(np.mean(vina)),
            "ha7": float(np.mean([score <= -7.0 for score in vina])),
            "ha_reference": float(np.mean([int(row["high_affinity_reference"]) for row in values])),
        }
    return {
        "n": len(rows),
        "vina_micro": float(np.mean([float(row["vina_score"]) for row in rows])),
        "vina_macro": float(np.mean([value["vina"] for value in per_pocket.values()])),
        "ha7_micro": float(np.mean([float(row["vina_score"]) <= -7.0 for row in rows])),
        "ha7_macro": float(np.mean([value["ha7"] for value in per_pocket.values()])),
        "ha_reference_micro": float(np.mean([int(row["high_affinity_reference"]) for row in rows])),
        "ha_reference_macro": float(np.mean([value["ha_reference"] for value in per_pocket.values()])),
        "per_pocket": per_pocket,
    }


def bootstrap(pairs, repeats, seed):
    rng = np.random.default_rng(seed)
    by_pocket = {}
    for key, raw, refined in pairs:
        by_pocket.setdefault(key[0], []).append((raw, refined))
    pockets = sorted(by_pocket)
    draws = {name: [] for name in ("vina", "ha7", "ha_reference")}
    centers = {name: [] for name in draws}
    for values in by_pocket.values():
        centers["vina"].append(np.mean([float(b["vina_score"]) - float(a["vina_score"]) for a, b in values]))
        centers["ha7"].append(np.mean([(float(b["vina_score"]) <= -7) - (float(a["vina_score"]) <= -7) for a, b in values]))
        centers["ha_reference"].append(np.mean([int(b["high_affinity_reference"]) - int(a["high_affinity_reference"]) for a, b in values]))
    for _ in range(repeats):
        sampled_pockets = rng.choice(pockets, len(pockets), replace=True)
        aggregate = {name: [] for name in draws}
        for pocket in sampled_pockets:
            values = by_pocket[int(pocket)]
            indices = rng.integers(0, len(values), len(values))
            aggregate["vina"].append(np.mean([float(values[i][1]["vina_score"]) - float(values[i][0]["vina_score"]) for i in indices]))
            aggregate["ha7"].append(np.mean([(float(values[i][1]["vina_score"]) <= -7) - (float(values[i][0]["vina_score"]) <= -7) for i in indices]))
            aggregate["ha_reference"].append(np.mean([int(values[i][1]["high_affinity_reference"]) - int(values[i][0]["high_affinity_reference"]) for i in indices]))
        for name in draws:
            draws[name].append(float(np.mean(aggregate[name])))
    return {
        name: {
            "mean_refined_minus_raw": float(np.mean(centers[name])),
            "ci95": [float(np.percentile(draws[name], 2.5)), float(np.percentile(draws[name], 97.5))],
        }
        for name in draws
    }


def main():
    parser = argparse.ArgumentParser(description="Compare raw and topology-refined validation poses.")
    parser.add_argument("--raw-docking", required=True)
    parser.add_argument("--refined-docking", required=True)
    parser.add_argument("--raw-posebusters", required=True)
    parser.add_argument("--refined-posebusters", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    raw_lookup = scored_lookup(args.raw_docking)
    refined_lookup = scored_lookup(args.refined_docking)
    keys = sorted(set(raw_lookup) & set(refined_lookup))
    pairs = [(key, raw_lookup[key], refined_lookup[key]) for key in keys]
    raw_rows = [raw for _, raw, _ in pairs]
    refined_rows = [refined for _, _, refined in pairs]
    pockets = sorted({key[0] for key in keys})
    raw_posebusters = json.loads(Path(args.raw_posebusters).read_text(encoding="utf-8"))["posebusters"]
    refined_posebusters = json.loads(Path(args.refined_posebusters).read_text(encoding="utf-8"))["posebusters"]
    result = {
        "protocol": (
            "same molecules, fixed held-out pockets "
            + ",".join(map(str, pockets))
            + ", qvina2.1 box20 exhaustiveness8 seed17"
        ),
        "held_out_pockets": pockets,
        "paired_n": len(pairs),
        "same_smiles": sum(a.get("smiles") == b.get("smiles") for _, a, b in pairs),
        "raw": summarize(raw_rows),
        "refined": summarize(refined_rows),
        "paired_bootstrap": bootstrap(pairs, args.bootstrap, args.seed),
        "posebusters": {"raw": raw_posebusters, "refined": refined_posebusters},
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
