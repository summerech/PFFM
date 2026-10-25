import argparse
import json
import random
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.fragment_flow import FragmentPocketFlow
from models.flexible_fragment_affinity import (
    FlexibleFragmentAffinity,
    FlexibleFragmentAffinityEnsemble,
)
from models.flow_affinity_head import FlowAffinityHead
from models.fragment_property import FragmentPropertyPredictor
from models.fragment_multiproperty import FragmentMultiPropertyPredictor


def main():
    parser = argparse.ArgumentParser(description="Sample coarse rBRICS graphs for pockets.")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--pockets", default="0,1,2,3,4")
    parser.add_argument(
        "--dataset-split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset subset whose local pocket indices are listed by --pockets.",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--samples-per-pocket", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-temperature", type=float, default=0.9)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--max-token-rings", type=int, default=-1)
    parser.add_argument("--forbid-token-bridgeheads", action="store_true")
    parser.add_argument("--desired-heavy-atoms", type=int, default=28)
    parser.add_argument("--desired-qed", type=float, default=0.7)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--size-tolerance", type=int, default=1)
    parser.add_argument("--token-search-topk", type=int, default=128)
    parser.add_argument(
        "--disable-constrained-decoding",
        action="store_true",
        help=(
            "Ablation only: skip size/capacity token decoding and connected "
            "capacity-aware edge decoding; retain thresholded predicted edges."
        ),
    )
    parser.add_argument("--property-checkpoint")
    parser.add_argument("--affinity-checkpoint")
    parser.add_argument("--affinity-scale", type=float, default=0.0)
    parser.add_argument("--affinity-start", type=float, default=0.5)
    parser.add_argument("--affinity-grad-clip", type=float, default=3.0)
    parser.add_argument("--fragment-affinity-checkpoint")
    parser.add_argument("--fragment-affinity-checkpoints", nargs="+")
    parser.add_argument("--fragment-affinity-scale", type=float, default=0.0)
    parser.add_argument("--fragment-affinity-steps", type=int, default=8)
    parser.add_argument("--fragment-affinity-grad-clip", type=float, default=3.0)
    parser.add_argument("--fragment-affinity-uncertainty-weight", type=float, default=0.35)
    parser.add_argument(
        "--pocket-overrides",
        help=(
            "JSON mapping pocket indices to per-pocket sampling overrides. Supported "
            "keys: token_temperature, desired_heavy_atoms, desired_qed, guidance_scale, "
            "fragment_affinity_scale, fragment_affinity_steps, and "
            "fragment_affinity_grad_clip, edge_threshold, max_token_rings, and "
            "forbid_token_bridgeheads."
        ),
    )
    parser.add_argument(
        "--conditioning-pocket-map",
        help=(
            "JSON mapping each reported/evaluation pocket index to the test-pocket "
            "index used as the model condition. Intended for pocket-shuffling controls."
        ),
    )
    parser.add_argument("--multiproperty-checkpoint")
    parser.add_argument("--property-qed-floor", type=float, default=0.58)
    parser.add_argument("--property-qed-weight", type=float, default=4.0)
    parser.add_argument("--property-logp-min", type=float, default=2.0)
    parser.add_argument("--property-logp-max", type=float, default=3.3)
    parser.add_argument("--property-logp-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--pocket-encoder-mode",
        choices=(
            "atom_only",
            "residue_graph",
            "residue_hypergraph",
            "esm2_residue_hypergraph",
        ),
        help="Override the pocket encoder recorded in the checkpoint.",
    )
    args = parser.parse_args()

    pocket_overrides = {}
    if args.pocket_overrides:
        with open(args.pocket_overrides, "r", encoding="utf-8") as handle:
            pocket_overrides = json.load(handle)
    conditioning_pocket_map = {}
    if args.conditioning_pocket_map:
        with open(args.conditioning_pocket_map, "r", encoding="utf-8") as handle:
            conditioning_pocket_map = json.load(handle)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = misc.load_config(args.config)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    _, subsets = get_dataset(config.data, transform=transform)
    with open(args.vocab, "r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    id_to_token = {int(value): key for key, value in vocabulary["token_to_id"].items()}
    token_capacity_lookup = torch.full(
        (int(vocabulary["vocab_size"]),), -1, dtype=torch.long, device=args.device
    )
    token_size_lookup = torch.ones(
        int(vocabulary["vocab_size"]), dtype=torch.long, device=args.device
    )
    token_feature_lookup = torch.zeros(
        (int(vocabulary["vocab_size"]), 10), dtype=torch.float32, device=args.device
    )
    token_ring_lookup = torch.zeros(
        int(vocabulary["vocab_size"]), dtype=torch.long, device=args.device
    )
    token_bridgehead_lookup = torch.zeros(
        int(vocabulary["vocab_size"]), dtype=torch.long, device=args.device
    )
    for token, token_id in vocabulary["token_to_id"].items():
        if token == "<unk>":
            continue
        fragment = Chem.MolFromSmiles(token)
        if fragment is None:
            continue
        capacity = sum(atom.GetAtomicNum() == 0 for atom in fragment.GetAtoms())
        token_capacity_lookup[int(token_id)] = capacity
        token_size_lookup[int(token_id)] = sum(
            atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()
        )
        atoms = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() > 0]
        count = max(len(atoms), 1)
        atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
        token_feature_lookup[int(token_id)] = torch.tensor(
            [
                atomic_numbers.count(6) / count,
                atomic_numbers.count(7) / count,
                atomic_numbers.count(8) / count,
                atomic_numbers.count(16) / count,
                atomic_numbers.count(15) / count,
                sum(number in {9, 17, 35, 53} for number in atomic_numbers) / count,
                sum(number not in {6, 7, 8, 16, 15, 9, 17, 35, 53} for number in atomic_numbers) / count,
                sum(atom.GetIsAromatic() for atom in atoms) / count,
                sum(atom.IsInRing() for atom in atoms) / count,
                sum(atom.GetFormalCharge() for atom in atoms) / 4.0,
            ],
            device=args.device,
        )
        token_ring_lookup[int(token_id)] = rdMolDescriptors.CalcNumRings(fragment)
        token_bridgehead_lookup[int(token_id)] = rdMolDescriptors.CalcNumBridgeheadAtoms(
            fragment
        )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    pocket_encoder_mode = args.pocket_encoder_mode or checkpoint.get(
        "pocket_encoder_mode", checkpoint.get("config", {}).get("pocket_encoder_mode", "atom_only")
    )
    checkpoint_config = checkpoint.get("config", {})
    model = FragmentPocketFlow(
        protein_feature_dim=protein_featurizer.feature_dim,
        vocab_size=int(checkpoint["vocab_size"]),
        pocket_encoder_mode=pocket_encoder_mode,
        residue_mix_floor=float(checkpoint_config.get("residue_mix_floor", 0.0)),
        atom_branch_dropout=0.0,
        atom_branch_dropout_weight=float(
            checkpoint_config.get("atom_branch_dropout_weight", 0.5)
        ),
        fusion_gate_bias=float(checkpoint_config.get("fusion_gate_bias", 4.0)),
        hypergraph_geometry=bool(checkpoint_config.get("hypergraph_geometry", False)),
        esm2_model_name=checkpoint_config.get(
            "esm2_model_name", "facebook/esm2_t6_8M_UR50D"
        ),
        sequence_fusion_rank=int(checkpoint_config.get("sequence_fusion_rank", 32)),
    ).to(args.device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = all(
        key.startswith("multiscale_pocket_encoder.")
        or ".residue_message." in key
        or ".residue_coordinate." in key
        or ".residue_update." in key
        or ".residue_norm." in key
        or key.startswith("pocket_compatibility_head.")
        for key in missing
    )
    if unexpected or (missing and not allowed_missing):
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    property_predictor = None
    if args.property_checkpoint:
        property_checkpoint = torch.load(args.property_checkpoint, map_location=args.device)
        property_predictor = FragmentPropertyPredictor(
            property_checkpoint["vocab_size"],
            hidden_dim=property_checkpoint.get("hidden_dim", 192),
        ).to(args.device)
        property_predictor.load_state_dict(property_checkpoint["model"])
        property_predictor.eval()
    affinity_head = None
    affinity_mean = None
    affinity_std = None
    if args.affinity_checkpoint:
        affinity_checkpoint = torch.load(args.affinity_checkpoint, map_location=args.device)
        affinity_head = FlowAffinityHead(
            hidden_dim=affinity_checkpoint.get("hidden_dim", model.hidden_dim)
        ).to(args.device)
        affinity_head.load_state_dict(affinity_checkpoint["model"])
        affinity_head.eval()
        for parameter in affinity_head.parameters():
            parameter.requires_grad_(False)
        affinity_mean = float(affinity_checkpoint["mean"])
        affinity_std = float(affinity_checkpoint["std"])
    fragment_affinity_ranker = None
    fragment_affinity_mean = None
    fragment_affinity_std = None
    checkpoint_paths = args.fragment_affinity_checkpoints
    if checkpoint_paths is None and args.fragment_affinity_checkpoint:
        checkpoint_paths = [args.fragment_affinity_checkpoint]
    if checkpoint_paths:
        members, means, stds = [], [], []
        for checkpoint_path in checkpoint_paths:
            fragment_checkpoint = torch.load(checkpoint_path, map_location=args.device)
            member = FlexibleFragmentAffinity(
                fragment_checkpoint["vocab_size"],
                hidden_dim=fragment_checkpoint.get("hidden_dim", 192),
            ).to(args.device)
            member.load_state_dict(fragment_checkpoint["model"])
            member.eval()
            member.requires_grad_(False)
            members.append(member)
            means.append(float(fragment_checkpoint["mean"]))
            stds.append(float(fragment_checkpoint["std"]))
        if len(members) == 1:
            fragment_affinity_ranker = members[0]
            fragment_affinity_mean = means[0]
            fragment_affinity_std = stds[0]
        else:
            fragment_affinity_ranker = FlexibleFragmentAffinityEnsemble(
                members,
                means,
                stds,
                uncertainty_weight=args.fragment_affinity_uncertainty_weight,
            ).to(args.device)
            fragment_affinity_ranker.eval()
            fragment_affinity_ranker.requires_grad_(False)
            fragment_affinity_mean = fragment_affinity_ranker.output_mean
            fragment_affinity_std = fragment_affinity_ranker.output_std
    multiproperty_predictor = None
    multiproperty_logp_mean = 0.0
    multiproperty_logp_std = 1.0
    if args.multiproperty_checkpoint:
        multiproperty_checkpoint = torch.load(
            args.multiproperty_checkpoint, map_location=args.device
        )
        multiproperty_predictor = FragmentMultiPropertyPredictor(
            multiproperty_checkpoint["vocab_size"],
            hidden_dim=multiproperty_checkpoint.get("hidden_dim", 192),
        ).to(args.device)
        multiproperty_predictor.load_state_dict(multiproperty_checkpoint["model"])
        multiproperty_predictor.eval()
        multiproperty_predictor.requires_grad_(False)
        multiproperty_logp_mean = float(multiproperty_checkpoint["logp_mean"])
        multiproperty_logp_std = float(multiproperty_checkpoint["logp_std"])

    samples = []
    for pocket_index in [int(value) for value in args.pockets.split(",") if value]:
        override = pocket_overrides.get(str(pocket_index), {})
        token_temperature = float(
            override.get("token_temperature", args.token_temperature)
        )
        desired_heavy_atoms = int(
            override.get("desired_heavy_atoms", args.desired_heavy_atoms)
        )
        desired_qed = float(override.get("desired_qed", args.desired_qed))
        guidance_scale = float(
            override.get("guidance_scale", args.guidance_scale)
        )
        fragment_affinity_scale = float(
            override.get("fragment_affinity_scale", args.fragment_affinity_scale)
        )
        fragment_affinity_steps = int(
            override.get("fragment_affinity_steps", args.fragment_affinity_steps)
        )
        fragment_affinity_grad_clip = float(
            override.get(
                "fragment_affinity_grad_clip", args.fragment_affinity_grad_clip
            )
        )
        edge_threshold = float(override.get("edge_threshold", args.edge_threshold))
        max_token_rings = int(override.get("max_token_rings", args.max_token_rings))
        forbid_token_bridgeheads = bool(
            override.get("forbid_token_bridgeheads", args.forbid_token_bridgeheads)
        )
        token_allowed_lookup = torch.ones(
            int(vocabulary["vocab_size"]), dtype=torch.bool, device=args.device
        )
        token_allowed_lookup[0] = False
        if max_token_rings >= 0:
            token_allowed_lookup &= token_ring_lookup <= max_token_rings
        if forbid_token_bridgeheads:
            token_allowed_lookup &= token_bridgehead_lookup == 0
        conditioning_pocket_index = int(
            conditioning_pocket_map.get(str(pocket_index), pocket_index)
        )
        data = subsets[args.dataset_split][conditioning_pocket_index].to(args.device)
        for sample_index in range(args.samples_per_pocket):
            started = time.perf_counter()
            result = model.sample(
                data.protein_pos,
                data.protein_atom_feature.float(),
                protein_element=data.protein_element,
                num_steps=args.num_steps,
                edge_threshold=edge_threshold,
                token_temperature=token_temperature,
                token_capacity_lookup=token_capacity_lookup,
                token_size_lookup=token_size_lookup,
                token_allowed_lookup=token_allowed_lookup,
                desired_heavy_atoms=desired_heavy_atoms,
                desired_qed=desired_qed,
                guidance_scale=guidance_scale,
                size_tolerance=args.size_tolerance,
                token_search_topk=args.token_search_topk,
                affinity_head=affinity_head,
                affinity_scale=args.affinity_scale,
                affinity_start=args.affinity_start,
                affinity_grad_clip=args.affinity_grad_clip,
                fragment_affinity_ranker=fragment_affinity_ranker,
                token_feature_lookup=token_feature_lookup,
                fragment_affinity_scale=fragment_affinity_scale,
                fragment_affinity_steps=fragment_affinity_steps,
                fragment_affinity_grad_clip=fragment_affinity_grad_clip,
                multiproperty_predictor=multiproperty_predictor,
                multiproperty_logp_mean=multiproperty_logp_mean,
                multiproperty_logp_std=multiproperty_logp_std,
                property_qed_floor=args.property_qed_floor,
                property_qed_weight=args.property_qed_weight,
                property_logp_min=args.property_logp_min,
                property_logp_max=args.property_logp_max,
                property_logp_weight=args.property_logp_weight,
                constrained_decoding=not args.disable_constrained_decoding,
            )
            property_prediction = result["qed_prediction"]
            if property_predictor is not None:
                fragment_batch = torch.zeros_like(result["fragment_token"])
                property_prediction = float(
                    property_predictor(
                        result["fragment_token"],
                        result["attachment_capacity"],
                        fragment_batch,
                        token_size_lookup,
                    ).item()
                )
            samples.append(
                {
                    "pocket_index": pocket_index,
                    "dataset_split": args.dataset_split,
                    "conditioning_pocket_index": conditioning_pocket_index,
                    "sample_index": sample_index,
                    "fragment_tokens": [
                        id_to_token.get(int(token), "<unk>")
                        for token in result["fragment_token"].cpu().tolist()
                    ],
                    "fragment_positions": result["fragment_pos"].cpu().tolist(),
                    "attachment_capacity": result["attachment_capacity"].cpu().tolist(),
                    "edges": result["edges"],
                    "predicted_heavy_atoms": result["predicted_heavy_atoms"],
                    "heavy_atom_condition": result["heavy_atom_condition"],
                    "qed_prediction": property_prediction,
                    "decoded_heavy_atoms": result["decoded_heavy_atoms"],
                    "guidance_scale": guidance_scale,
                    "token_temperature": token_temperature,
                    "desired_qed": desired_qed,
                    "affinity_scale": args.affinity_scale,
                    "fragment_affinity_scale": fragment_affinity_scale,
                    "fragment_affinity_steps": fragment_affinity_steps,
                    "fragment_affinity_grad_clip": fragment_affinity_grad_clip,
                    "edge_threshold": edge_threshold,
                    "max_token_rings": max_token_rings,
                    "forbid_token_bridgeheads": forbid_token_bridgeheads,
                    "constrained_decoding": not args.disable_constrained_decoding,
                    "fragment_affinity_members": len(checkpoint_paths or []),
                    "predicted_vina": (
                        result["normalized_vina_prediction"] * affinity_std + affinity_mean
                        if result["normalized_vina_prediction"] is not None else None
                    ),
                    "fragment_predicted_vina": (
                        result["fragment_normalized_vina_prediction"] * fragment_affinity_std
                        + fragment_affinity_mean
                        if result["fragment_normalized_vina_prediction"] is not None
                        else None
                    ),
                    "multiproperty_qed_prediction": result[
                        "multiproperty_qed_prediction"
                    ],
                    "multiproperty_logp_prediction": result[
                        "multiproperty_logp_prediction"
                    ],
                    "generation_time": time.perf_counter() - started,
                }
            )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2)
    print(json.dumps(samples, indent=2))
    print(f"saved samples to {args.output}")


if __name__ == "__main__":
    main()
