import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for report_path in sorted(root.glob("w_*/seed_*/metrics.json")):
        docking = json.loads(report_path.read_text(encoding="utf-8"))
        light = json.loads((report_path.parent / "light_metrics.json").read_text(encoding="utf-8"))
        scale = float(report_path.parents[1].name.removeprefix("w_").replace("p", "."))
        seed = int(report_path.parent.name.removeprefix("seed_"))
        records.append(
            {
                "scale": scale,
                "seed": seed,
                "vina": float(docking["vina_macro"]),
                "ha7": float(docking["ha_fixed_7_macro"]),
                "ha_reference": float(
                    docking["ha_reference_macro"]
                ),
                "qed": float(light["qed"]),
                "validity": float(light["validity"]),
                "coverage": float(docking["overall_scored_rate"]),
            }
        )

    if not records:
        summary_path = root / "cfg_pareto_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"No metric reports found below {root}")
        records = json.loads(summary_path.read_text(encoding="utf-8"))["per_run"]

    summary = []
    for scale in sorted({row["scale"] for row in records}):
        rows = [row for row in records if row["scale"] == scale]
        item = {"scale": scale, "seeds": [row["seed"] for row in rows]}
        for metric in ("vina", "ha7", "ha_reference", "qed", "validity", "coverage"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            item[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary.append(item)

    payload = {
        "design": {
            "scales": [row["scale"] for row in summary],
            "seeds": sorted({row["seed"] for row in records}),
            "pockets": list(range(10)),
            "dataset_split": "train",
            "samples_per_pocket": 30,
            "selection_policy": "validation-only Pareto sweep; public test excluded",
        },
        "per_run": records,
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=180)
    qeds = [row["qed"]["mean"] for row in summary]
    vinas = [row["vina"]["mean"] for row in summary]
    qerr = [row["qed"]["std"] for row in summary]
    verr = [row["vina"]["std"] for row in summary]
    ax.errorbar(qeds, vinas, xerr=qerr, yerr=verr, fmt="none", ecolor="#90A4AE", capsize=3)
    points = ax.scatter(qeds, vinas, c=[row["scale"] for row in summary], cmap="viridis", s=65)
    for row, x, y in zip(summary, qeds, vinas):
        ax.annotate(f"w={row['scale']:g}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("QED (higher is better)")
    ax.set_ylabel("Pocket-macro Vina Dock (kcal/mol; lower is better)")
    ax.set_title("Vina Dock-QED Pareto sweep")
    ax.grid(alpha=0.22)
    fig.colorbar(points, ax=ax, label="CFG scale w")
    fig.tight_layout()
    fig.savefig(args.plot, bbox_inches="tight")


if __name__ == "__main__":
    main()
