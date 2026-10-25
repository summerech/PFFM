import argparse
import json
import platform
import time

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


def make_inputs(batch, token_class_weight, token_size_lookup, property_predictor):
    return dict(
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
        condition_dropout=0.15,
        property_predictor=property_predictor,
        property_loss_weight=2.0,
        junction_distance_loss_weight=1.0,
        protein_repulsion_loss_weight=1.0,
        pocket_contrastive_loss_weight=0.0,
        pocket_contrastive_temperature=0.1,
        pocket_compatibility_loss_weight=0.0,
    )


def benchmark_mode(args, mode, checkpoint_path, batches, protein_feature_dim, vocab_size,
                   token_class_weight, token_size_lookup, property_predictor):
    torch.cuda.empty_cache()
    model = FragmentPocketFlow(
        protein_feature_dim=protein_feature_dim,
        vocab_size=vocab_size,
        pocket_encoder_mode=mode,
    ).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)

    def train_step(batch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model.get_loss(**make_inputs(
            batch, token_class_weight, token_size_lookup, property_predictor
        ))
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    for index in range(args.warmup):
        train_step(batches[index % len(batches)])
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for index in range(args.train_steps):
        train_step(batches[index % len(batches)])
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    model.eval()
    with torch.no_grad():
        for index in range(args.warmup):
            batch = batches[index % len(batches)]
            model.encode_pocket(
                batch.protein_pos,
                batch.protein_atom_feature.float(),
                batch.protein_element_batch,
            )
        torch.cuda.synchronize()
        timings = []
        for index in range(args.inference_steps):
            batch = batches[index % len(batches)]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model.encode_pocket(
                batch.protein_pos,
                batch.protein_atom_feature.float(),
                batch.protein_element_batch,
            )
            end_event.record()
            end_event.synchronize()
            timings.append(start_event.elapsed_time(end_event))

    total_pockets = args.train_steps * args.batch_size
    inference_pockets = args.inference_steps * args.batch_size
    result = {
        "mode": mode,
        "checkpoint": checkpoint_path,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "training_seconds": train_seconds,
        "training_pockets_per_second": total_pockets / train_seconds,
        "encoder_latency_ms_per_batch": sum(timings) / len(timings),
        "encoder_latency_ms_per_pocket": sum(timings) / inference_pockets,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--graph-checkpoint", required=True)
    parser.add_argument("--hypergraph-checkpoint", required=True)
    parser.add_argument("--property-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, default=100)
    args = parser.parse_args()

    config = misc.load_config(args.config)
    misc.seed_all(2027)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    _, subsets = get_dataset(config.data, transform=transform)
    train_set = FragmentCacheDataset(subsets["train"], args.cache, args.vocab)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    batches = []
    for batch in loader:
        batches.append(batch.to(args.device))
        # A fixed real batch isolates model compute and avoids counting a
        # prefetched benchmark corpus as persistent GPU memory.
        if len(batches) >= 1:
            break

    with open(args.vocab, "r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    vocab_size = int(vocabulary["vocab_size"])
    token_class_weight = torch.ones(vocab_size, device=args.device)
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            token_class_weight[token_id] = 0.2
        else:
            token_class_weight[token_id] = max(
                1, int(vocabulary["frequency"].get(token, 1))
            ) ** -0.5
    known_weight = token_class_weight[1:]
    token_class_weight[1:] = (
        known_weight / known_weight.mean().clamp(min=1e-8)
    ).clamp(max=5.0)
    token_size_lookup = torch.ones(vocab_size, device=args.device)
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            continue
        fragment = Chem.MolFromSmiles(token)
        if fragment is not None:
            token_size_lookup[token_id] = sum(
                atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()
            )

    property_checkpoint = torch.load(args.property_checkpoint, map_location=args.device)
    property_predictor = FragmentPropertyPredictor(
        property_checkpoint["vocab_size"],
        hidden_dim=property_checkpoint.get("hidden_dim", 192),
    ).to(args.device)
    property_predictor.load_state_dict(property_checkpoint["model"])
    property_predictor.eval().requires_grad_(False)

    results = []
    for mode, checkpoint_path in (
        ("residue_graph", args.graph_checkpoint),
        ("residue_hypergraph", args.hypergraph_checkpoint),
    ):
        results.append(benchmark_mode(
            args, mode, checkpoint_path, batches, protein_featurizer.feature_dim,
            vocab_size, token_class_weight, token_size_lookup, property_predictor,
        ))

    payload = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "protocol": {
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "train_steps": args.train_steps,
            "inference_steps": args.inference_steps,
            "training_scope": "full model forward, backward, and optimizer step",
            "inference_scope": "pocket encoder forward only",
        },
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
