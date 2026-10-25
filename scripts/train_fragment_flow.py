import argparse
import json
import os

import torch
from rdkit import Chem
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import FragmentCacheDataset, get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.fragment_flow import FragmentPocketFlow
from models.fragment_property import FragmentPropertyPredictor


def main():
    parser = argparse.ArgumentParser(description="Train pocket-conditioned rBRICS fragment flow.")
    parser.add_argument("config")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--condition-dropout", type=float, default=0.15)
    parser.add_argument("--property-checkpoint")
    parser.add_argument("--property-loss-weight", type=float, default=2.0)
    parser.add_argument("--junction-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--protein-repulsion-loss-weight", type=float, default=1.0)
    parser.add_argument("--residue-mix-floor", type=float, default=0.0)
    parser.add_argument("--atom-branch-dropout", type=float, default=0.0)
    parser.add_argument("--atom-branch-dropout-weight", type=float, default=0.5)
    parser.add_argument("--fusion-gate-bias", type=float, default=4.0)
    parser.add_argument("--hypergraph-geometry", action="store_true")
    parser.add_argument(
        "--esm2-model-name", default="facebook/esm2_t6_8M_UR50D"
    )
    parser.add_argument("--sequence-fusion-rank", type=int, default=32)
    parser.add_argument("--pocket-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--pocket-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--pocket-compatibility-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--pocket-encoder-mode",
        choices=(
            "atom_only",
            "residue_graph",
            "residue_hypergraph",
            "esm2_residue_hypergraph",
        ),
        default="atom_only",
    )
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
    vocab_size = int(vocabulary["vocab_size"])
    token_class_weight = torch.ones(vocab_size, dtype=torch.float32)
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            token_class_weight[token_id] = 0.2
            continue
        frequency = max(1, int(vocabulary["frequency"].get(token, 1)))
        token_class_weight[token_id] = frequency ** -0.5
    known_weight = token_class_weight[1:]
    token_class_weight[1:] = (
        known_weight / known_weight.mean().clamp(min=1e-8)
    ).clamp(max=5.0)
    token_class_weight = token_class_weight.to(args.device)
    token_size_lookup = torch.ones(vocab_size, dtype=torch.float32)
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            continue
        fragment = Chem.MolFromSmiles(token)
        if fragment is not None:
            token_size_lookup[token_id] = sum(
                atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()
            )
    token_size_lookup = token_size_lookup.to(args.device)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    train_iterator = iter(train_loader)
    model = FragmentPocketFlow(
        protein_feature_dim=protein_featurizer.feature_dim,
        vocab_size=vocab_size,
        pocket_encoder_mode=args.pocket_encoder_mode,
        residue_mix_floor=args.residue_mix_floor,
        atom_branch_dropout=args.atom_branch_dropout,
        atom_branch_dropout_weight=args.atom_branch_dropout_weight,
        fusion_gate_bias=args.fusion_gate_bias,
        hypergraph_geometry=args.hypergraph_geometry,
        esm2_model_name=args.esm2_model_name,
        sequence_fusion_rank=args.sequence_fusion_rank,
    ).to(args.device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=args.device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        print(
            f"initialized from {args.init_checkpoint}; missing={missing}; unexpected={unexpected}",
            flush=True,
        )
    property_predictor = None
    if args.property_checkpoint:
        property_checkpoint = torch.load(args.property_checkpoint, map_location=args.device)
        property_predictor = FragmentPropertyPredictor(
            property_checkpoint["vocab_size"],
            hidden_dim=property_checkpoint.get("hidden_dim", 192),
        ).to(args.device)
        property_predictor.load_state_dict(property_checkpoint["model"])
        property_predictor.eval()
        property_predictor.requires_grad_(False)
        print(
            f"loaded frozen property predictor from {args.property_checkpoint}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
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
        losses = model.get_loss(
            protein_pos=batch.protein_pos,
            protein_feature=batch.protein_atom_feature.float(),
            protein_batch=batch.protein_element_batch,
            target_fragment_pos=batch.ligand_fragment_pos,
            target_fragment_token=batch.ligand_fragment_token,
            target_attachment_count=batch.ligand_fragment_attachment_count,
            fragment_batch=batch.ligand_fragment_pos_batch,
            target_edge_index=batch.ligand_fragment_edge_index,
            token_class_weight=token_class_weight,
            token_size_lookup=token_size_lookup,
            target_heavy_atom_count=batch.ligand_heavy_atom_count,
            target_qed=batch.ligand_qed_target,
            condition_dropout=args.condition_dropout,
            property_predictor=property_predictor,
            property_loss_weight=args.property_loss_weight,
            junction_distance_loss_weight=args.junction_distance_loss_weight,
            protein_repulsion_loss_weight=args.protein_repulsion_loss_weight,
            pocket_contrastive_loss_weight=args.pocket_contrastive_loss_weight,
            pocket_contrastive_temperature=args.pocket_contrastive_temperature,
            pocket_compatibility_loss_weight=args.pocket_compatibility_loss_weight,
        )
        losses["loss"].backward()
        grad_norm = clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if iteration == 1 or iteration % args.report_every == 0:
            values = " ".join(f"{key}={float(value):.4f}" for key, value in losses.items())
            print(f"iter={iteration} {values} grad={float(grad_norm):.3f}", flush=True)

    model.eval()
    validation = {}
    validation_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= args.val_batches:
                break
            batch = batch.to(args.device)
            losses = model.get_loss(
                protein_pos=batch.protein_pos,
                protein_feature=batch.protein_atom_feature.float(),
                protein_batch=batch.protein_element_batch,
                target_fragment_pos=batch.ligand_fragment_pos,
                target_fragment_token=batch.ligand_fragment_token,
                target_attachment_count=batch.ligand_fragment_attachment_count,
                fragment_batch=batch.ligand_fragment_pos_batch,
                target_edge_index=batch.ligand_fragment_edge_index,
                token_class_weight=token_class_weight,
                token_size_lookup=token_size_lookup,
                target_heavy_atom_count=batch.ligand_heavy_atom_count,
                target_qed=batch.ligand_qed_target,
                condition_dropout=0.0,
                property_predictor=property_predictor,
                property_loss_weight=args.property_loss_weight,
                junction_distance_loss_weight=args.junction_distance_loss_weight,
                protein_repulsion_loss_weight=args.protein_repulsion_loss_weight,
                pocket_contrastive_loss_weight=args.pocket_contrastive_loss_weight,
                pocket_contrastive_temperature=args.pocket_contrastive_temperature,
                pocket_compatibility_loss_weight=args.pocket_compatibility_loss_weight,
            )
            for key, value in losses.items():
                validation[key] = validation.get(key, 0.0) + float(value)
            validation_count += 1
    validation = {
        key: value / max(validation_count, 1) for key, value in validation.items()
    }
    print(
        "validation " + " ".join(
            f"{key}={value:.4f}" for key, value in validation.items()
        ),
        flush=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "vocab_size": vocab_size,
            "model_family": "pocket_rbrics_fragment_flow",
            "pocket_encoder_mode": args.pocket_encoder_mode,
            "validation": validation,
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
