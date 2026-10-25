import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select constrained energy guidance on validation only.")
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collision-tolerance", type=float, default=0.0)
    parser.add_argument("--posebusters-tolerance", type=float, default=0.005)
    parser.add_argument("--geometry-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    candidates = {}
    baseline = None
    for name, path in args.candidate:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        refined = result["refined"]
        pose = result["posebusters"]["refined"]
        metrics = {
            "vina": refined["vina_macro"],
            "ha7": refined["ha7_macro"],
            "ha_reference": refined["ha_reference_macro"],
            "posebusters_all": pose["all_selected_tests_pass_rate"],
            "bond_lengths": pose["test_pass_rates"]["bond_lengths"],
            "bond_angles": pose["test_pass_rates"]["bond_angles"],
            "protein_collision": pose["protein_ligand_collision_rate"],
        }
        if baseline is None:
            baseline = dict(metrics)
        metrics["eligible"] = bool(
            metrics["protein_collision"]
            <= baseline["protein_collision"] + args.collision_tolerance
            and metrics["posebusters_all"]
            >= baseline["posebusters_all"] - args.posebusters_tolerance
            and metrics["bond_lengths"]
            >= baseline["bond_lengths"] - args.geometry_tolerance
            and metrics["bond_angles"]
            >= baseline["bond_angles"] - args.geometry_tolerance
        )
        metrics["selection_score"] = (
            -0.10 * metrics["vina"]
            + 0.35 * metrics["ha_reference"]
            + 0.20 * metrics["ha7"]
            + 0.25 * metrics["posebusters_all"]
            - 0.20 * metrics["protein_collision"]
        )
        candidates[name] = metrics
    eligible = [name for name, row in candidates.items() if row["eligible"]]
    selected = max(
        eligible,
        key=lambda name: (
            candidates[name]["selection_score"],
            -candidates[name]["protein_collision"],
        ),
    )
    output = {
        "scope": "validation-only; baseline is an explicit candidate; no test metrics used",
        "score_formula": "-.10*Vina + .35*HAref + .20*HA7 + .25*PB_all - .20*collision",
        "constraints": {
            "collision_tolerance": args.collision_tolerance,
            "posebusters_tolerance": args.posebusters_tolerance,
            "geometry_tolerance": args.geometry_tolerance,
        },
        "selected": selected,
        "eligible_candidates": eligible,
        "candidates": candidates,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
