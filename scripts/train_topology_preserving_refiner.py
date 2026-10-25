import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.topology_preserving_refiner import (
    DEFAULT_PHYSICS_WEIGHTS,
    TopologyPreservingAtomicRefiner,
    physical_objective,
    physical_terms,
)


def parse_ids(value):
    return {int(item) for item in value.split(",") if item.strip()}


def move(record, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in record.items()}


def batches(records, batch_size, rng):
    order = list(range(len(records)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start:start + batch_size]]


def prepare_record(source, target_key):
    record = dict(source)
    if target_key not in record:
        raise KeyError(f"record has no target key: {target_key}")
    record["refiner_target_pos"] = record[target_key]
    record.setdefault("geometry_reference_pos", record["raw_pos"])
    return record


def training_loss(model, record, noise, physics_weights, protein_clash_scale,
                  protein_safety_weight):
    time = torch.rand((), device=record["raw_pos"].device)
    raw, target = record["raw_pos"], record["refiner_target_pos"]
    position = (
        (1.0 - time) * raw + time * target
        + torch.randn_like(raw) * noise * time * (1.0 - time)
    )
    velocity = model.forward_graph(record, position, time)
    target_velocity = target - raw
    endpoint = position + (1.0 - time) * velocity
    flow = F.smooth_l1_loss(velocity, target_velocity)
    coordinate = F.smooth_l1_loss(endpoint, target)
    terms = physical_terms(
        endpoint,
        record,
        record["geometry_reference_pos"],
        protein_clash_scale=protein_clash_scale,
    )
    physics = physical_objective(terms, physics_weights)

    raw_terms = physical_terms(
        raw,
        record,
        record["geometry_reference_pos"],
        protein_clash_scale=protein_clash_scale,
    )
    non_regression = sum(
        F.relu(terms[key] - raw_terms[key].detach())
        for key in ("bond", "angle", "torsion", "conformer", "intramolecular_clash")
    )
    target_contact = torch.cdist(target, record["protein_pos"]).min(1).values
    endpoint_contact = torch.cdist(endpoint, record["protein_pos"]).min(1).values
    contact = F.smooth_l1_loss(endpoint_contact, target_contact)
    protein_safety = (
        F.relu(terms["protein_overlap_max"] - raw_terms["protein_overlap_max"].detach())
        + F.relu(raw_terms["protein_clearance"].detach() - terms["protein_clearance"])
    )
    total = (
        flow + 0.5 * coordinate + physics + 2.0 * non_regression
        + 0.5 * contact + protein_safety_weight * protein_safety
    )
    return total, {"flow": flow, "coordinate": coordinate, "physics": physics,
                   "non_regression": non_regression, "contact": contact,
                   "protein_safety": protein_safety, **terms}


def graph_metrics(position, record, protein_clash_scale=0.75):
    terms = physical_terms(
        position,
        record,
        record["geometry_reference_pos"],
        protein_clash_scale=protein_clash_scale,
    )
    target = record["refiner_target_pos"]
    return {
        "target_rmsd": float((position - target).square().sum(-1).mean().sqrt()),
        "bond": float(terms["bond"]),
        "angle": float(terms["angle"]),
        "torsion": float(terms["torsion"]),
        "conformer": float(terms["conformer"]),
        "intramolecular_clash": float(terms["intramolecular_clash"]),
        "protein_clash": float(terms["protein_clash"]),
        "protein_clash_rate": float(terms["protein_clash_rate"]),
    }


@torch.no_grad()
def evaluate(model, records, device, steps, strength, limit_per_pocket=None,
             refine_kwargs=None):
    model.eval()
    selected, counts = [], {}
    for record in records:
        pocket = int(record["pocket_index"])
        if limit_per_pocket is None or counts.get(pocket, 0) < limit_per_pocket:
            selected.append(record)
            counts[pocket] = counts.get(pocket, 0) + 1
    per_pocket = {}
    for source in selected:
        record = move(source, device)
        protein_clash_scale = (refine_kwargs or {}).get("protein_clash_scale", 0.75)
        raw = graph_metrics(record["raw_pos"], record, protein_clash_scale)
        refined, status = model.refine(
            record, steps=steps, strength=strength, **(refine_kwargs or {})
        )
        result = graph_metrics(refined, record, protein_clash_scale)
        row = {f"raw_{key}": value for key, value in raw.items()}
        row.update({f"refined_{key}": value for key, value in result.items()})
        row["acceptance_rate"] = status["accepted_steps"] / max(status["attempted_steps"], 1)
        per_pocket.setdefault(int(record["pocket_index"]), []).append(row)
    if not per_pocket:
        return {}
    keys = next(iter(next(iter(per_pocket.values())))).keys()
    macro = {}
    for key in keys:
        pocket_values = [sum(row[key] for row in rows) / len(rows) for rows in per_pocket.values()]
        macro[key] = sum(pocket_values) / len(pocket_values)
    return macro


def main():
    parser = argparse.ArgumentParser(description="Train a topology-preserving atom-level pose refiner.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-key", default="local_pos")
    parser.add_argument("--train-pockets")
    parser.add_argument("--val-pockets")
    parser.add_argument("--test-pockets")
    parser.add_argument(
        "--partition-field",
        help=(
            "Record field containing train/val/test labels. When set, this takes "
            "precedence over pocket-index lists and prevents local-index collisions."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--eval-strength", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit-per-pocket", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--protein-clash-weight", type=float, default=4.0)
    parser.add_argument("--protein-clash-scale", type=float, default=0.75)
    parser.add_argument("--protein-safety-weight", type=float, default=0.0)
    parser.add_argument("--preserve-protein-clearance", action="store_true")
    parser.add_argument("--clearance-tolerance", type=float, default=0.0)
    parser.add_argument("--selection-protein-clash-weight", type=float, default=2.0)
    parser.add_argument("--skip-test-eval", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = torch.load(args.data, map_location="cpu")
    records = [prepare_record(record, args.target_key) for record in payload["records"]]
    if args.partition_field:
        missing = sum(args.partition_field not in row for row in records)
        if missing:
            raise ValueError(
                f"{missing} records have no partition field {args.partition_field!r}"
            )
        train = [row for row in records if row[args.partition_field] == "train"]
        val = [row for row in records if row[args.partition_field] == "val"]
        test = [row for row in records if row[args.partition_field] == "test"]
    else:
        if not all((args.train_pockets, args.val_pockets, args.test_pockets)):
            raise ValueError(
                "provide --partition-field or all three pocket-index lists"
            )
        train_ids, val_ids, test_ids = map(
            parse_ids, (args.train_pockets, args.val_pockets, args.test_pockets)
        )
        train = [row for row in records if int(row["pocket_index"]) in train_ids]
        val = [row for row in records if int(row["pocket_index"]) in val_ids]
        test = [row for row in records if int(row["pocket_index"]) in test_ids]
    if args.train_limit and len(train) > args.train_limit:
        train = random.Random(args.seed).sample(train, args.train_limit)
    if not train or not val or not test:
        raise ValueError(f"empty split: train={len(train)}, val={len(val)}, test={len(test)}")

    model = TopologyPreservingAtomicRefiner(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers
    ).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=args.device)
        model.load_state_dict(initial["model"])
    physics_weights = dict(DEFAULT_PHYSICS_WEIGHTS)
    physics_weights["protein_clash"] = args.protein_clash_weight
    refine_kwargs = {
        "weights": physics_weights,
        "protein_clash_scale": args.protein_clash_scale,
        "preserve_protein_clearance": args.preserve_protein_clearance,
        "clearance_tolerance": args.clearance_tolerance,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng, best = random.Random(args.seed), math.inf
    print(json.dumps({"train": len(train), "val": len(val), "test": len(test), "config": vars(args)}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, term_totals = [], {}
        for batch in batches(train, args.batch_size, rng):
            losses, rows = [], []
            for source in batch:
                loss, terms = training_loss(
                    model,
                    move(source, args.device),
                    args.noise,
                    physics_weights,
                    args.protein_clash_scale,
                    args.protein_safety_weight,
                )
                losses.append(loss)
                rows.append(terms)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach()))
            for key in ("flow", "physics", "non_regression", "protein_clash"):
                term_totals.setdefault(key, []).append(float(torch.stack([row[key].detach() for row in rows]).mean()))

        should_evaluate = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        val_metrics = None
        if should_evaluate:
            val_metrics = evaluate(
                model, val, args.device, args.eval_steps, args.eval_strength,
                args.eval_limit_per_pocket,
                refine_kwargs,
            )
            selection = (
                val_metrics["refined_target_rmsd"]
                + args.selection_protein_clash_weight
                * val_metrics["refined_protein_clash_rate"]
                + 2.0 * val_metrics["refined_bond"]
                + val_metrics["refined_angle"]
            )
            if selection < best:
                best = selection
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "physics_weights": physics_weights,
                    "best_val_selection": best,
                    "config": vars(args),
                }, args.out)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps({
                "epoch": epoch,
                "loss": sum(totals) / len(totals),
                "terms": {key: sum(values) / len(values) for key, values in term_totals.items()},
                "val": val_metrics,
            }), flush=True)

    checkpoint = torch.load(args.out, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    final = {
        "best_val_selection": best,
        "val": evaluate(model, val, args.device, args.eval_steps, args.eval_strength, args.eval_limit_per_pocket, refine_kwargs),
    }
    if not args.skip_test_eval:
        final["test"] = evaluate(
            model, test, args.device, args.eval_steps, args.eval_strength,
            args.eval_limit_per_pocket, refine_kwargs,
        )
    print("FINAL " + json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
