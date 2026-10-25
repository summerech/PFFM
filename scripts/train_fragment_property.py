import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import FragmentCacheDataset, get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.fragment_property import FragmentPropertyPredictor


def build_size_lookup(vocabulary):
    lookup = torch.ones(int(vocabulary["vocab_size"]), dtype=torch.float32)
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            continue
        fragment = Chem.MolFromSmiles(token)
        if fragment is not None:
            lookup[int(token_id)] = sum(
                atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()
            )
    return lookup


def metrics(prediction, target):
    error = prediction - target
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    correlation = (
        (target_centered * prediction_centered).sum()
        / (
            target_centered.square().sum().sqrt()
            * prediction_centered.square().sum().sqrt()
        ).clamp(min=1e-8)
    )
    r2 = 1.0 - error.square().sum() / target_centered.square().sum().clamp(min=1e-8)
    return {
        "mae": error.abs().mean(),
        "rmse": error.square().mean().sqrt(),
        "r2": r2,
        "pearson": correlation,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain a condition-free rBRICS QED predictor.")
    parser.add_argument("config")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=50)
    args = parser.parse_args()

    config = misc.load_config(args.config)
    misc.seed_all(config.train.seed)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    _, subsets = get_dataset(config.data, transform=transform)
    train_set = FragmentCacheDataset(subsets["train"], args.cache, args.vocab)
    validation_subset = subsets.get("val")
    if validation_subset is None or len(validation_subset) == 0:
        validation_subset = subsets["test"]
    val_set = FragmentCacheDataset(validation_subset, args.cache, args.vocab)
    with open(args.vocab, "r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    token_size_lookup = build_size_lookup(vocabulary).to(args.device)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    train_iterator = iter(train_loader)
    model = FragmentPropertyPredictor(int(vocabulary["vocab_size"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for iteration in range(1, args.max_iters + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = batch.to(args.device)
        model.train()
        optimizer.zero_grad()
        prediction = model(
            batch.ligand_fragment_token,
            batch.ligand_fragment_attachment_count,
            batch.ligand_fragment_pos_batch,
            token_size_lookup,
        )
        target = batch.ligand_qed_target.float()
        loss = F.smooth_l1_loss(prediction, target, beta=0.05)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if iteration == 1 or iteration % args.report_every == 0:
            values = metrics(prediction.detach(), target)
            report = " ".join(f"{key}={float(value):.4f}" for key, value in values.items())
            print(
                f"iter={iteration} loss={float(loss):.4f} {report} grad={float(grad_norm):.3f}",
                flush=True,
            )

    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= args.val_batches:
                break
            batch = batch.to(args.device)
            predictions.append(
                model(
                    batch.ligand_fragment_token,
                    batch.ligand_fragment_attachment_count,
                    batch.ligand_fragment_pos_batch,
                    token_size_lookup,
                )
            )
            targets.append(batch.ligand_qed_target.float())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    validation = {key: float(value) for key, value in metrics(prediction, target).items()}
    print(
        "validation " + " ".join(f"{key}={value:.4f}" for key, value in validation.items()),
        flush=True,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": int(vocabulary["vocab_size"]),
            "hidden_dim": model.hidden_dim,
            "validation": validation,
            "property": "qed",
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
