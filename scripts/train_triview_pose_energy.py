import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shared_pose_energy import SharedPoseEnergy
from scripts.train_atomic_affinity import pack


VIEWS = ("raw", "local", "global")


def parse_ids(value):
    return {int(item) for item in value.split(",") if item}


def batches(records, batch_size, rng, shuffle=True):
    groups = {}
    for record in records:
        groups.setdefault(int(record["pocket_index"]), []).append(record)
    result = []
    for values in groups.values():
        values = list(values)
        if shuffle:
            rng.shuffle(values)
        result.extend(values[start:start + batch_size] for start in range(0, len(values), batch_size))
    if shuffle:
        rng.shuffle(result)
    return result


def targets(batch, device):
    keys = ("raw_score_only", "local_vina_score", "global_vina_score")
    return {view: torch.tensor([float(record[key]) for record in batch], device=device) for view, key in zip(VIEWS, keys)}


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp(min=1e-8))


def ordered_pair_loss(better_prediction, worse_prediction, better_target, worse_target, margin):
    valid = better_target < worse_target
    if not valid.any():
        return better_prediction.sum() * 0.0
    return F.relu(margin + better_prediction[valid] - worse_prediction[valid]).mean()


def within_pocket_pair_loss(prediction, target, minimum_gap=0.5):
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    left, right = torch.triu_indices(prediction.numel(), prediction.numel(), offset=1, device=prediction.device)
    target_delta = target[right] - target[left]
    valid = target_delta.abs() >= minimum_gap
    if not valid.any():
        return prediction.sum() * 0.0
    direction = target_delta[valid].sign()
    predicted_delta = prediction[right[valid]] - prediction[left[valid]]
    return F.softplus(-direction * predicted_delta).mean()


def listwise_loss(prediction, target, temperature=0.25):
    if prediction.numel() < 2:
        return prediction.sum() * 0.0
    target_distribution = torch.softmax(-target / temperature, dim=0)
    return F.kl_div(
        torch.log_softmax(-prediction / temperature, dim=0),
        target_distribution,
        reduction="batchmean",
    )


def pair_accuracy(prediction, target, minimum_gap=0.5):
    if prediction.numel() < 2:
        return float("nan")
    left, right = torch.triu_indices(prediction.numel(), prediction.numel(), offset=1)
    target_delta = target[right] - target[left]
    valid = target_delta.abs() >= minimum_gap
    if not valid.any():
        return float("nan")
    predicted_delta = prediction[right[valid]] - prediction[left[valid]]
    return float((predicted_delta.sign() == target_delta[valid].sign()).float().mean())


def coordinate_gradient_losses(model, batch, device, maximum_norm):
    raw_positions = [
        record["raw_pos"].to(device).detach().requires_grad_(True) for record in batch
    ]
    raw_views = [
        {
            "protein_z": record["protein_z"],
            "protein_pos": record["protein_pos"],
            "ligand_z": record["ligand_z"],
            "ligand_pos": position,
        }
        for record, position in zip(batch, raw_positions)
    ]
    energy = model(pack(raw_views, device))
    gradients = torch.autograd.grad(energy.sum(), raw_positions, create_graph=True)
    cosine_losses, norm_losses, cosines, norms = [], [], [], []
    for record, position, gradient in zip(batch, raw_positions, gradients):
        displacement = record["local_pos"].to(device) - position.detach()
        cosine = F.cosine_similarity(
            -gradient.flatten(), displacement.flatten(), dim=0, eps=1e-8
        )
        norm = (gradient.square().sum() + 1e-12).sqrt()
        cosine_losses.append(1.0 - cosine)
        norm_losses.append(F.relu(norm - maximum_norm).square())
        cosines.append(cosine.detach())
        norms.append(norm.detach())
    return (
        torch.stack(cosine_losses).mean(),
        torch.stack(norm_losses).mean(),
        torch.stack(cosines).mean(),
        torch.stack(norms).mean(),
    )


def path_monotonic_loss(model, batch, device, points, margin):
    fractions = torch.linspace(0.0, 1.0, points, device=device)
    views = []
    for record in batch:
        raw = record["raw_pos"].to(device)
        local = record["local_pos"].to(device)
        for fraction in fractions:
            views.append({
                "protein_z": record["protein_z"],
                "protein_pos": record["protein_pos"],
                "ligand_z": record["ligand_z"],
                "ligand_pos": (1.0 - fraction) * raw + fraction * local,
            })
    energy = model(pack(views, device)).reshape(len(batch), points)
    violations = F.relu(margin + energy[:, 1:] - energy[:, :-1])
    return violations.mean(), (energy[:, 1:] < energy[:, :-1]).float().mean().detach()


def evaluate_gradient(model, records, device, limit_per_pocket=10):
    model.eval()
    selected, counts = [], {}
    for record in records:
        pocket = int(record["pocket_index"])
        if counts.get(pocket, 0) >= limit_per_pocket:
            continue
        counts[pocket] = counts.get(pocket, 0) + 1
        selected.append(record)
    cosines, norms = [], []
    with torch.enable_grad():
        for record in selected:
            position = record["raw_pos"].to(device).detach().requires_grad_(True)
            view = {
                "protein_z": record["protein_z"].to(device),
                "protein_pos": record["protein_pos"].to(device),
                "ligand_z": record["ligand_z"].to(device),
                "ligand_pos": position,
            }
            energy = model.forward_records([view])[0]
            gradient = torch.autograd.grad(energy, position)[0]
            displacement = record["local_pos"].to(device) - position.detach()
            cosines.append(float(F.cosine_similarity(
                -gradient.flatten(), displacement.flatten(), dim=0, eps=1e-8
            )))
            norms.append(float(gradient.norm()))
    return {
        "n": len(selected),
        "gradient_cosine": sum(cosines) / len(cosines),
        "positive_alignment": sum(value > 0 for value in cosines) / len(cosines),
        "gradient_norm": sum(norms) / len(norms),
    }


@torch.no_grad()
def evaluate(model, records, device, batch_size):
    model.eval()
    pockets = []
    for pocket in sorted({int(record["pocket_index"]) for record in records}):
        selected = [record for record in records if int(record["pocket_index"]) == pocket]
        predictions = {view: [] for view in VIEWS}; truth = {view: [] for view in VIEWS}
        for batch in batches(selected, batch_size, random.Random(0), shuffle=False):
            target = targets(batch, "cpu")
            for view in VIEWS:
                predictions[view].append(model(pack([record[f"{view}_view"] for record in batch], device)).cpu())
                truth[view].append(target[view])
        predictions = {key: torch.cat(value) for key, value in predictions.items()}
        truth = {key: torch.cat(value) for key, value in truth.items()}
        physical = {key: model.inverse_energy(value) for key, value in predictions.items()}
        pockets.append({
            "pocket": pocket, "n": len(selected),
            "raw_to_local_accuracy": float((predictions["local"] < predictions["raw"]).float().mean()),
            "local_to_global_accuracy": float((predictions["global"] < predictions["local"]).float().mean()),
            "strict_order_accuracy": float(((predictions["global"] < predictions["local"]) & (predictions["local"] < predictions["raw"])).float().mean()),
            "raw_mae_transformed": float((predictions["raw"] - model.transform_energy(truth["raw"])).abs().mean()),
            "local_mae_kcal": float((physical["local"] - truth["local"]).abs().mean()),
            "global_mae_kcal": float((physical["global"] - truth["global"]).abs().mean()),
            "global_pearson": pearson(physical["global"], truth["global"]),
            "raw_pair_accuracy": pair_accuracy(physical["raw"], truth["raw"]),
            "local_pair_accuracy": pair_accuracy(physical["local"], truth["local"]),
            "global_pair_accuracy": pair_accuracy(physical["global"], truth["global"]),
        })
    keys = ("raw_to_local_accuracy", "local_to_global_accuracy", "strict_order_accuracy", "raw_mae_transformed", "local_mae_kcal", "global_mae_kcal", "global_pearson", "raw_pair_accuracy", "local_pair_accuracy", "global_pair_accuracy")
    macro = {}
    for key in keys:
        values = [row[key] for row in pockets if not math.isnan(row[key])]
        macro[key] = sum(values) / len(values) if values else float("nan")
    return {"macro": macro, "per_pocket": pockets}


def main():
    parser = argparse.ArgumentParser(description="Train shared raw/local/global pose energy.")
    parser.add_argument("--data", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--train-pockets"); parser.add_argument("--val-pockets"); parser.add_argument("--test-pockets")
    parser.add_argument("--partition-field")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--physical-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96); parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pair-margin", type=float, default=0.25); parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--within-pair-weight", type=float, default=0.0)
    parser.add_argument("--raw-pair-scale", type=float, default=0.25)
    parser.add_argument("--minimum-pair-gap", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=0.25)
    parser.add_argument("--gradient-alignment-weight", type=float, default=0.0)
    parser.add_argument("--gradient-norm-weight", type=float, default=0.0)
    parser.add_argument("--gradient-maximum-norm", type=float, default=0.05)
    parser.add_argument("--path-monotonic-weight", type=float, default=0.0)
    parser.add_argument("--path-points", type=int, default=5)
    parser.add_argument("--path-margin", type=float, default=0.01)
    parser.add_argument("--selection-gradient-weight", type=float, default=0.0)
    parser.add_argument("--selection-gradient-norm-weight", type=float, default=0.0)
    parser.add_argument("--gradient-eval-limit-per-pocket", type=int, default=10)
    parser.add_argument("--init", default=None)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=109)
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    records = torch.load(args.data, map_location="cpu")["records"]
    if args.partition_field:
        missing = sum(args.partition_field not in record for record in records)
        if missing:
            raise RuntimeError(f"{missing} records lack partition field {args.partition_field!r}")
        train = [r for r in records if r[args.partition_field] == "train"]
        val = [r for r in records if r[args.partition_field] == "val"]
        test = [r for r in records if r[args.partition_field] == "test"]
    else:
        if not all((args.train_pockets, args.val_pockets, args.test_pockets)):
            raise RuntimeError("provide --partition-field or all pocket lists")
        train_ids, val_ids, test_ids = map(parse_ids, (args.train_pockets, args.val_pockets, args.test_pockets))
        train = [r for r in records if int(r["pocket_index"]) in train_ids]; val = [r for r in records if int(r["pocket_index"]) in val_ids]; test = [r for r in records if int(r["pocket_index"]) in test_ids]
    if not train or not val or not test: raise RuntimeError("Empty target-disjoint split")
    model = SharedPoseEnergy(
        hidden_dim=args.hidden_dim, physical_features=args.physical_features
    ).to(args.device)
    if args.init:
        initial = torch.load(args.init, map_location=args.device)
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng, best, best_state = random.Random(args.seed), -math.inf, None
    print(json.dumps({"train": len(train), "val": len(val), "test": len(test), "config": vars(args)}), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; diagnostics = {}
        for batch in batches(train, args.batch_size, rng):
            target = targets(batch, args.device)
            prediction = {view: model(pack([record[f"{view}_view"] for record in batch], args.device)) for view in VIEWS}
            regression = sum(F.smooth_l1_loss(prediction[view], model.transform_energy(target[view])) for view in VIEWS)
            ordering = ordered_pair_loss(prediction["local"], prediction["raw"], target["local"], target["raw"], args.pair_margin)
            ordering = ordering + ordered_pair_loss(prediction["global"], prediction["local"], target["global"], target["local"], args.pair_margin)
            transformed_target = {view: model.transform_energy(target[view]) for view in VIEWS}
            within_pair = (
                args.raw_pair_scale * within_pocket_pair_loss(prediction["raw"], transformed_target["raw"], args.minimum_pair_gap / 10.0)
                + within_pocket_pair_loss(prediction["local"], transformed_target["local"], args.minimum_pair_gap / 10.0)
                + within_pocket_pair_loss(prediction["global"], transformed_target["global"], args.minimum_pair_gap / 10.0)
            )
            listwise = sum(
                listwise_loss(prediction[view], transformed_target[view], args.listwise_temperature)
                for view in ("local", "global")
            )
            loss = (
                regression
                + args.pair_weight * ordering
                + args.within_pair_weight * within_pair
                + args.listwise_weight * listwise
            )
            if args.gradient_alignment_weight > 0 or args.gradient_norm_weight > 0:
                alignment, norm_penalty, gradient_cosine, gradient_norm = coordinate_gradient_losses(
                    model, batch, args.device, args.gradient_maximum_norm
                )
                loss = (
                    loss
                    + args.gradient_alignment_weight * alignment
                    + args.gradient_norm_weight * norm_penalty
                )
                diagnostics.setdefault("gradient_cosine", []).append(float(gradient_cosine))
                diagnostics.setdefault("gradient_norm", []).append(float(gradient_norm))
                diagnostics.setdefault("gradient_alignment_loss", []).append(float(alignment.detach()))
                diagnostics.setdefault("gradient_norm_penalty", []).append(float(norm_penalty.detach()))
            if args.path_monotonic_weight > 0:
                path_loss, monotonic_fraction = path_monotonic_loss(
                    model, batch, args.device, args.path_points, args.path_margin
                )
                loss = loss + args.path_monotonic_weight * path_loss
                diagnostics.setdefault("path_loss", []).append(float(path_loss.detach()))
                diagnostics.setdefault("path_monotonic_fraction", []).append(float(monotonic_fraction))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        metrics = evaluate(model, val, args.device, args.batch_size); macro = metrics["macro"]
        gradient_metrics = evaluate_gradient(
            model, val, args.device, args.gradient_eval_limit_per_pocket
        )
        score = (
            macro["strict_order_accuracy"]
            + 0.125 * macro["local_pair_accuracy"]
            + 0.125 * macro["global_pair_accuracy"]
            + 0.1 * macro["global_pearson"]
            + args.selection_gradient_weight * gradient_metrics["gradient_cosine"]
            - args.selection_gradient_norm_weight * max(
                0.0, gradient_metrics["gradient_norm"] - args.gradient_maximum_norm
            )
        )
        if score > best:
            best = score; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs: print(json.dumps({"epoch": epoch, "loss": sum(losses) / len(losses), "train_diagnostics": {key: sum(values) / len(values) for key, values in diagnostics.items()}, "val": macro, "val_gradient": gradient_metrics}), flush=True)
    model.load_state_dict(best_state); result = {"best_selection": best, "val": evaluate(model, val, args.device, args.batch_size), "val_gradient": evaluate_gradient(model, val, args.device, args.gradient_eval_limit_per_pocket)}
    if not args.skip_test_eval:
        result["test"] = evaluate(model, test, args.device, args.batch_size)
        result["test_gradient"] = evaluate_gradient(model, test, args.device, args.gradient_eval_limit_per_pocket)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); torch.save({"model": best_state, "hidden_dim": args.hidden_dim, "physical_features": args.physical_features, "config": vars(args), "metrics": result}, args.out)
    print("FINAL " + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
