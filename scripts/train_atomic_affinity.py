import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.atomic_affinity import AtomicPocketAffinity


def move(record, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in record.items()}


def pack(records, device, coordinate_noise=0.0, translation_noise=0.0):
    batch_size = len(records)
    max_protein = max(record["protein_z"].numel() for record in records)
    max_ligand = max(record["ligand_z"].numel() for record in records)
    protein_z = torch.zeros(batch_size, max_protein, dtype=torch.long, device=device)
    protein_pos = torch.zeros(batch_size, max_protein, 3, device=device)
    protein_mask = torch.zeros(batch_size, max_protein, dtype=torch.bool, device=device)
    ligand_z = torch.zeros(batch_size, max_ligand, dtype=torch.long, device=device)
    ligand_pos = torch.zeros(batch_size, max_ligand, 3, device=device)
    ligand_mask = torch.zeros(batch_size, max_ligand, dtype=torch.bool, device=device)
    for index, record in enumerate(records):
        num_protein, num_ligand = record["protein_z"].numel(), record["ligand_z"].numel()
        protein_z[index, :num_protein] = record["protein_z"].to(device)
        protein_pos[index, :num_protein] = record["protein_pos"].to(device)
        protein_mask[index, :num_protein] = True
        ligand_z[index, :num_ligand] = record["ligand_z"].to(device)
        ligand_pos[index, :num_ligand] = record["ligand_pos"].to(device)
        if coordinate_noise > 0:
            ligand_pos[index, :num_ligand] += torch.randn_like(ligand_pos[index, :num_ligand]) * coordinate_noise
        if translation_noise > 0:
            ligand_pos[index, :num_ligand] += torch.randn(1, 3, device=device) * translation_noise
        ligand_mask[index, :num_ligand] = True
    return {"protein_z": protein_z, "protein_pos": protein_pos, "protein_mask": protein_mask,
            "ligand_z": ligand_z, "ligand_pos": ligand_pos, "ligand_mask": ligand_mask}


def batches(records, batch_size, shuffle, rng):
    order = list(range(len(records)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start:start + batch_size]]


def ranking_loss(prediction, target, records, margin=0.1):
    losses = []
    for pocket in {record["pocket_pdb"] for record in records}:
        index = torch.tensor([i for i, record in enumerate(records) if record["pocket_pdb"] == pocket], device=prediction.device)
        if index.numel() < 2:
            continue
        pred, truth = prediction[index], target[index]
        better = truth[:, None] + 0.5 < truth[None, :]
        if better.any():
            losses.append(F.relu(margin + pred[:, None] - pred[None, :])[better].mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


@torch.no_grad()
def evaluate(model, records, device, mean, std, batch_size):
    model.eval()
    predictions, logits, targets, labels = [], [], [], []
    rng = random.Random(0)
    for batch in batches(records, batch_size, False, rng):
        prediction, logit = model(pack(batch, device))
        predictions.extend((prediction * std + mean).cpu().tolist())
        logits.extend(logit.cpu().tolist())
        targets.extend(record["vina_score"] for record in batch)
        labels.extend(record["high_affinity"] for record in batch)
    p, y = torch.tensor(predictions), torch.tensor(targets)
    centered_p, centered_y = p - p.mean(), y - y.mean()
    pearson = float((centered_p * centered_y).sum() / (centered_p.square().sum().sqrt() * centered_y.square().sum().sqrt()).clamp(min=1e-8))
    auc = float(roc_auc_score(labels, [-value for value in logits])) if len(set(labels)) > 1 else float("nan")
    return {"n": len(y), "mae": float((p-y).abs().mean()), "rmse": float((p-y).square().mean().sqrt()), "pearson": pearson, "ha_auc": auc, "pred_mean": float(p.mean()), "target_mean": float(y.mean())}


def main():
    parser = argparse.ArgumentParser(description="Train atom-level 3D qvina affinity predictor.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coordinate-noise", type=float, default=0.0)
    parser.add_argument("--translation-noise", type=float, default=0.0)
    parser.add_argument("--init", default=None, help="Optional atomic-affinity checkpoint to fine-tune.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    payload = torch.load(args.data, map_location="cpu")
    pockets = sorted(payload["pockets"])
    if len(pockets) < 3:
        raise ValueError("at least three pockets are required for pocket-held-out evaluation")
    test_pocket, val_pocket = pockets[-1], pockets[-2]
    train = [r for r in payload["records"] if r["pocket_pdb"] not in {val_pocket, test_pocket}]
    val = [r for r in payload["records"] if r["pocket_pdb"] == val_pocket]
    test = [r for r in payload["records"] if r["pocket_pdb"] == test_pocket]
    mean = torch.tensor([r["vina_score"] for r in train]).mean().to(args.device)
    std = torch.tensor([r["vina_score"] for r in train]).std().clamp(min=0.25).to(args.device)
    positives = sum(r["high_affinity"] for r in train)
    pos_weight = torch.tensor((len(train)-positives) / max(positives, 1), device=args.device).clamp(max=10.0)
    model = AtomicPocketAffinity(hidden_dim=args.hidden_dim).to(args.device)
    if args.init:
        initial = torch.load(args.init, map_location=args.device)
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng, best = random.Random(args.seed), math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in batches(train, args.batch_size, True, rng):
            target = torch.tensor([record["vina_score"] for record in batch], device=args.device)
            label = torch.tensor([record["high_affinity"] for record in batch], device=args.device)
            coordinate_noise = random.random() * args.coordinate_noise
            translation_noise = random.random() * args.translation_noise
            prediction, logit = model(pack(batch, args.device, coordinate_noise, translation_noise))
            normalized = (target - mean) / std
            regression = F.smooth_l1_loss(prediction, normalized)
            classification = F.binary_cross_entropy_with_logits(-logit, label, pos_weight=pos_weight)
            rank = ranking_loss(prediction, normalized, batch)
            loss = regression + 0.25 * classification + 0.25 * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val, args.device, mean, std, args.batch_size)
        history.append({"epoch": epoch, "loss": sum(losses)/len(losses), "val": metrics})
        if metrics["mae"] < best:
            best = metrics["mae"]
            torch.save({"model": model.state_dict(), "mean": float(mean), "std": float(std), "config": vars(args), "pockets": pockets, "val_pocket": val_pocket, "test_pocket": test_pocket}, args.out)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(history[-1]))
    checkpoint = torch.load(args.out, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    result = {"best_val_mae": best, "train": evaluate(model, train, args.device, mean, std, args.batch_size), "val": evaluate(model, val, args.device, mean, std, args.batch_size), "test": evaluate(model, test, args.device, mean, std, args.batch_size), "val_pocket": val_pocket, "test_pocket": test_pocket}
    print("FINAL " + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
