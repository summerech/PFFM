import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


def _pair_index(batch, include_self=False):
    pairs = []
    for graph_index in range(int(batch.max().item()) + 1):
        nodes = torch.where(batch == graph_index)[0]
        if nodes.numel() == 0:
            continue
        src = nodes.repeat_interleave(nodes.numel())
        dst = nodes.repeat(nodes.numel())
        if not include_self:
            keep = src != dst
            src, dst = src[keep], dst[keep]
        pairs.append(torch.stack([src, dst]))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(pairs, dim=1)


def _cross_index(fragment_batch, protein_batch):
    edges = []
    for graph_index in range(int(fragment_batch.max().item()) + 1):
        fragments = torch.where(fragment_batch == graph_index)[0]
        proteins = torch.where(protein_batch == graph_index)[0]
        if fragments.numel() == 0 or proteins.numel() == 0:
            continue
        edges.append(
            torch.stack(
                [
                    fragments.repeat_interleave(proteins.numel()),
                    proteins.repeat(fragments.numel()),
                ]
            )
        )
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=fragment_batch.device)
    return torch.cat(edges, dim=1)


def _protein_atom_to_residue(protein_feature, protein_batch):
    """Recover residue membership from ordered pocket atoms.

    CrossDocked pocket PDBs preserve residue atom order. Each standard residue
    starts with a backbone nitrogen, represented by element-N (feature 2) and
    the final backbone flag. The fallback keeps malformed pockets usable by
    assigning atoms before the first detected nitrogen to the first residue.
    """
    atom_to_residue = torch.empty_like(protein_batch)
    residue_batches = []
    offset = 0
    for graph_index in range(int(protein_batch.max().item()) + 1):
        atoms = torch.where(protein_batch == graph_index)[0]
        if atoms.numel() == 0:
            continue
        is_backbone_n = (
            (protein_feature[atoms, 2] > 0.5)
            & (protein_feature[atoms, -1] > 0.5)
        )
        local = is_backbone_n.long().cumsum(0) - 1
        if not bool(is_backbone_n.any()):
            local.zero_()
        local.clamp_(min=0)
        count = int(local.max().item()) + 1
        atom_to_residue[atoms] = local + offset
        residue_batches.append(
            torch.full((count,), graph_index, dtype=torch.long, device=protein_batch.device)
        )
        offset += count
    return atom_to_residue, torch.cat(residue_batches, dim=0)


def _residue_ca_positions(protein_pos, protein_feature, atom_to_residue):
    """Recover C-alpha coordinates from ordered backbone carbon atoms."""
    num_residues = int(atom_to_residue.max().item()) + 1
    fallback = scatter_mean(protein_pos, atom_to_residue, dim=0)
    result = fallback.clone()
    is_backbone_carbon = (protein_feature[:, 1] > 0.5) & (protein_feature[:, -1] > 0.5)
    for residue_index in range(num_residues):
        candidates = torch.where(
            (atom_to_residue == residue_index) & is_backbone_carbon
        )[0]
        if candidates.numel() > 0:
            result[residue_index] = protein_pos[candidates[0]]
    return result


def _residue_graph_index(residue_pos, residue_batch, cutoff=10.0):
    edge_index = _pair_index(residue_batch)
    if edge_index.numel() == 0:
        return edge_index
    src, dst = edge_index
    keep = (residue_pos[src] - residue_pos[dst]).norm(dim=-1) <= cutoff
    return edge_index[:, keep]


def _residue_hypergraph_incidence(residue_pos, residue_batch, spatial_cutoff=8.0):
    """Build sequence-window and spatial-neighborhood residue hyperedges."""
    node_parts, edge_parts = [], []
    edge_offset = 0
    for graph_index in range(int(residue_batch.max().item()) + 1):
        nodes = torch.where(residue_batch == graph_index)[0]
        if nodes.numel() == 0:
            continue
        local_count = nodes.numel()
        local_ids = torch.arange(local_count, device=nodes.device)
        for center in range(local_count):
            sequence_mask = (local_ids - center).abs() <= 2
            sequence_nodes = nodes[sequence_mask]
            node_parts.append(sequence_nodes)
            edge_parts.append(
                torch.full_like(sequence_nodes, edge_offset, dtype=torch.long)
            )
            edge_offset += 1

            distances = (residue_pos[nodes] - residue_pos[nodes[center]]).norm(dim=-1)
            spatial_nodes = nodes[distances <= spatial_cutoff]
            if spatial_nodes.numel() >= 2:
                node_parts.append(spatial_nodes)
                edge_parts.append(
                    torch.full_like(spatial_nodes, edge_offset, dtype=torch.long)
                )
                edge_offset += 1
    if not node_parts:
        empty = torch.empty(0, dtype=torch.long, device=residue_pos.device)
        return empty, empty
    return torch.cat(node_parts), torch.cat(edge_parts)


def _pamol_hypergraph_incidence(residue_pos, residue_batch, spatial_cutoff=5.0):
    """Build one PAMol-style spatial hyperedge around each residue."""
    node_parts, edge_parts = [], []
    edge_offset = 0
    for graph_index in range(int(residue_batch.max().item()) + 1):
        nodes = torch.where(residue_batch == graph_index)[0]
        if nodes.numel() == 0:
            continue
        distance = torch.cdist(residue_pos[nodes], residue_pos[nodes])
        for center in range(nodes.numel()):
            members = nodes[distance[center] < spatial_cutoff]
            if members.numel() == 0:
                members = nodes[center : center + 1]
            node_parts.append(members)
            edge_parts.append(torch.full_like(members, edge_offset))
            edge_offset += 1
    if not node_parts:
        empty = torch.empty(0, dtype=torch.long, device=residue_pos.device)
        return empty, empty
    return torch.cat(node_parts), torch.cat(edge_parts)


class FrozenESM2ResidueEncoder(nn.Module):
    """Contextual ESM2 embeddings for ordered pocket-residue sequences."""

    AA_SYMBOLS = tuple("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        try:
            from transformers import EsmModel, EsmTokenizer
        except ImportError as exc:
            raise ImportError(
                "ESM2 pocket encoding requires transformers; install transformers>=4.30"
            ) from exc
        tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
        self.esm = EsmModel.from_pretrained(model_name, local_files_only=True)
        self.output_dim = int(self.esm.config.hidden_size)
        self.model_name = model_name
        token_ids = [tokenizer.convert_tokens_to_ids(symbol) for symbol in self.AA_SYMBOLS]
        self.register_buffer("aa_token_ids", torch.tensor(token_ids, dtype=torch.long))
        self.cls_token_id = int(tokenizer.cls_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.esm.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.esm.eval()
        return self

    def forward(self, residue_aa_type, residue_batch):
        batch_size = int(residue_batch.max().item()) + 1
        lengths = torch.bincount(residue_batch, minlength=batch_size)
        max_length = int(lengths.max().item())
        input_ids = torch.full(
            (batch_size, max_length + 2),
            self.pad_token_id,
            dtype=torch.long,
            device=residue_batch.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1
        for graph_index, length in enumerate(lengths.tolist()):
            residue_index = torch.where(residue_batch == graph_index)[0]
            input_ids[graph_index, 1 : length + 1] = self.aa_token_ids[
                residue_aa_type[residue_index].clamp(min=0, max=19)
            ]
            input_ids[graph_index, length + 1] = self.eos_token_id
            attention_mask[graph_index, : length + 2] = 1
        self.esm.eval()
        with torch.no_grad():
            hidden = self.esm(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        parts = [hidden[i, 1 : length + 1] for i, length in enumerate(lengths.tolist())]
        return torch.cat(parts, dim=0)


class ResidueGraphLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, residue_h, residue_pos, edge_index):
        if edge_index.numel() == 0:
            return residue_h
        src, dst = edge_index
        distance = (residue_pos[src] - residue_pos[dst]).norm(dim=-1, keepdim=True)
        message = self.message(torch.cat([residue_h[src], residue_h[dst], distance], dim=-1))
        aggregate = scatter_mean(message, src, dim=0, dim_size=residue_h.size(0))
        return self.norm(residue_h + self.update(torch.cat([residue_h, aggregate], dim=-1)))


class ResidueHypergraphLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_to_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_to_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_distance = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_distance = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        residue_h,
        residue_pos,
        incidence_node,
        incidence_edge,
        use_geometry=False,
    ):
        if incidence_node.numel() == 0:
            return residue_h
        edge_pos = scatter_mean(
            residue_pos[incidence_node], incidence_edge, dim=0
        )
        incidence_distance = (
            residue_pos[incidence_node] - edge_pos[incidence_edge]
        ).norm(dim=-1, keepdim=True)
        node_to_edge = self.node_to_edge(residue_h[incidence_node])
        if use_geometry:
            node_to_edge = node_to_edge + self.node_distance(incidence_distance)
        edge_h = scatter_mean(
            node_to_edge, incidence_edge, dim=0
        )
        edge_to_node = self.edge_to_node(edge_h[incidence_edge])
        if use_geometry:
            edge_to_node = edge_to_node + self.edge_distance(incidence_distance)
        node_message = scatter_mean(
            edge_to_node,
            incidence_node,
            dim=0,
            dim_size=residue_h.size(0),
        )
        return self.norm(
            residue_h + self.update(torch.cat([residue_h, node_message], dim=-1))
        )


class MultiscalePocketEncoder(nn.Module):
    MODES = {
        "atom_only",
        "residue_graph",
        "residue_hypergraph",
        "esm2_residue_hypergraph",
    }

    def __init__(
        self,
        hidden_dim,
        mode="atom_only",
        num_layers=2,
        residue_mix_floor=0.0,
        atom_branch_dropout=0.0,
        atom_branch_dropout_weight=0.5,
        fusion_gate_bias=4.0,
        hypergraph_geometry=False,
        esm2_model_name="facebook/esm2_t6_8M_UR50D",
        sequence_fusion_rank=32,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown pocket encoder mode: {mode}")
        self.mode = mode
        self.residue_mix_floor = float(residue_mix_floor)
        self.atom_branch_dropout = float(atom_branch_dropout)
        self.atom_branch_dropout_weight = float(atom_branch_dropout_weight)
        self.hypergraph_geometry = bool(hypergraph_geometry)
        self.esm2_encoder = None
        if mode == "esm2_residue_hypergraph":
            self.esm2_encoder = FrozenESM2ResidueEncoder(esm2_model_name)
            self.esm2_projection = nn.Sequential(
                nn.Linear(self.esm2_encoder.output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.structure_low_rank = nn.Linear(hidden_dim, sequence_fusion_rank)
            self.sequence_low_rank = nn.Linear(hidden_dim, sequence_fusion_rank)
            self.sequence_structure_fusion = nn.Sequential(
                nn.Linear(2 * hidden_dim + sequence_fusion_rank, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        self.graph_layers = nn.ModuleList(
            [ResidueGraphLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.hypergraph_layers = nn.ModuleList(
            [ResidueHypergraphLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        # Preserve the pretrained atom-level encoder at initialization. The
        # residue branch is introduced gradually during matched fine-tuning.
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, fusion_gate_bias)

    def forward(
        self,
        protein_h,
        protein_pos,
        protein_feature,
        protein_batch,
        return_residue_context=False,
    ):
        if self.mode == "atom_only":
            if return_residue_context:
                return protein_h, None
            return protein_h
        atom_to_residue, residue_batch = _protein_atom_to_residue(
            protein_feature, protein_batch
        )
        residue_h = scatter_mean(protein_h, atom_to_residue, dim=0)
        residue_pos = _residue_ca_positions(
            protein_pos, protein_feature, atom_to_residue
        )
        if self.mode == "residue_graph":
            edge_index = _residue_graph_index(residue_pos, residue_batch)
            for layer in self.graph_layers:
                residue_h = layer(residue_h, residue_pos, edge_index)
        else:
            if self.mode == "esm2_residue_hypergraph":
                residue_aa_type = scatter_mean(
                    protein_feature[:, 6:26].float(), atom_to_residue, dim=0
                ).argmax(dim=-1)
                sequence_h = self.esm2_projection(
                    self.esm2_encoder(residue_aa_type, residue_batch)
                )
                bilinear_h = self.structure_low_rank(residue_h) * self.sequence_low_rank(
                    sequence_h
                )
                residue_h = self.sequence_structure_fusion(
                    torch.cat([residue_h, sequence_h, bilinear_h], dim=-1)
                )
                incidence_node, incidence_edge = _pamol_hypergraph_incidence(
                    residue_pos, residue_batch
                )
            else:
                incidence_node, incidence_edge = _residue_hypergraph_incidence(
                    residue_pos, residue_batch
                )
            for layer in self.hypergraph_layers:
                residue_h = layer(
                    residue_h,
                    residue_pos,
                    incidence_node,
                    incidence_edge,
                    use_geometry=self.hypergraph_geometry,
                )
        broadcast_h = residue_h[atom_to_residue]
        gate = torch.sigmoid(self.fusion_gate(torch.cat([protein_h, broadcast_h], dim=-1)))
        residue_weight = self.residue_mix_floor + (
            1.0 - self.residue_mix_floor
        ) * (1.0 - gate)
        if self.training and self.atom_branch_dropout > 0.0:
            drop_graph = (
                torch.rand(
                    int(protein_batch.max().item()) + 1,
                    device=protein_h.device,
                ) < self.atom_branch_dropout
            )
            forced_weight = torch.full_like(
                residue_weight, self.atom_branch_dropout_weight
            )
            residue_weight = torch.where(
                drop_graph[protein_batch].unsqueeze(-1),
                torch.maximum(residue_weight, forced_weight),
                residue_weight,
            )
        atom_h = protein_h + residue_weight * (broadcast_h - protein_h)
        if return_residue_context:
            return atom_h, (residue_h, residue_pos, residue_batch)
        return atom_h


def sinusoidal_time_embedding(time, dim):
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = time.unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.size(-1) < dim:
        embedding = F.pad(embedding, (0, dim - embedding.size(-1)))
    return embedding


class FragmentEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        message_input = 2 * hidden_dim + 1
        self.fragment_message = nn.Sequential(
            nn.Linear(message_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.protein_message = nn.Sequential(
            nn.Linear(message_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.fragment_coordinate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.protein_coordinate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.residue_message = nn.Sequential(
            nn.Linear(message_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.residue_coordinate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.residue_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.residue_norm = nn.LayerNorm(hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _relative(pos_src, pos_dst):
        relative = pos_dst - pos_src
        distance = relative.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return relative / distance, distance

    def forward(
        self,
        fragment_h,
        fragment_pos,
        protein_h,
        protein_pos,
        ff_index,
        fp_index,
        residue_context=None,
        fr_index=None,
    ):
        ff_src, ff_dst = ff_index
        ff_direction, ff_distance = self._relative(
            fragment_pos[ff_src], fragment_pos[ff_dst]
        )
        ff_message = self.fragment_message(
            torch.cat([fragment_h[ff_src], fragment_h[ff_dst], ff_distance], dim=-1)
        )
        ff_aggregate = scatter_sum(
            ff_message, ff_src, dim=0, dim_size=fragment_h.size(0)
        )
        ff_coordinate = scatter_sum(
            ff_direction * self.fragment_coordinate(ff_message),
            ff_src,
            dim=0,
            dim_size=fragment_h.size(0),
        )

        fp_fragment, fp_protein = fp_index
        fp_direction, fp_distance = self._relative(
            fragment_pos[fp_fragment], protein_pos[fp_protein]
        )
        fp_message = self.protein_message(
            torch.cat(
                [fragment_h[fp_fragment], protein_h[fp_protein], fp_distance], dim=-1
            )
        )
        fp_aggregate = scatter_sum(
            fp_message, fp_fragment, dim=0, dim_size=fragment_h.size(0)
        )
        fp_coordinate = scatter_sum(
            fp_direction * self.protein_coordinate(fp_message),
            fp_fragment,
            dim=0,
            dim_size=fragment_h.size(0),
        )
        degree = scatter_sum(
            torch.ones_like(fp_fragment, dtype=fragment_pos.dtype),
            fp_fragment,
            dim=0,
            dim_size=fragment_h.size(0),
        ).clamp(min=1.0).unsqueeze(-1)
        update = self.node_update(torch.cat([fragment_h, ff_aggregate, fp_aggregate / degree], dim=-1))
        fragment_h = self.norm(fragment_h + update)
        fragment_pos = fragment_pos + 0.1 * (ff_coordinate + fp_coordinate / degree)
        if residue_context is not None and fr_index is not None:
            residue_h, residue_pos, _ = residue_context
            fr_fragment, fr_residue = fr_index
            fr_direction, fr_distance = self._relative(
                fragment_pos[fr_fragment], residue_pos[fr_residue]
            )
            fr_message = self.residue_message(
                torch.cat(
                    [fragment_h[fr_fragment], residue_h[fr_residue], fr_distance], dim=-1
                )
            )
            fr_aggregate = scatter_sum(
                fr_message, fr_fragment, dim=0, dim_size=fragment_h.size(0)
            )
            fr_coordinate = scatter_sum(
                fr_direction * self.residue_coordinate(fr_message),
                fr_fragment,
                dim=0,
                dim_size=fragment_h.size(0),
            )
            fr_degree = scatter_sum(
                torch.ones_like(fr_fragment, dtype=fragment_pos.dtype),
                fr_fragment,
                dim=0,
                dim_size=fragment_h.size(0),
            ).clamp(min=1.0).unsqueeze(-1)
            fragment_h = self.residue_norm(
                fragment_h
                + self.residue_update(
                    torch.cat([fragment_h, fr_aggregate / fr_degree], dim=-1)
                )
            )
            fragment_pos = fragment_pos + 0.1 * fr_coordinate / fr_degree
        return fragment_h, fragment_pos


class FragmentPocketFlow(nn.Module):
    """Pocket-conditioned coarse flow over rBRICS fragment nodes."""

    def __init__(
        self,
        protein_feature_dim,
        vocab_size,
        hidden_dim=128,
        num_layers=4,
        max_fragments=24,
        max_attachments=6,
        max_heavy_atoms=64,
        pocket_encoder_mode="atom_only",
        residue_mix_floor=0.0,
        atom_branch_dropout=0.0,
        atom_branch_dropout_weight=0.5,
        fusion_gate_bias=4.0,
        hypergraph_geometry=False,
        esm2_model_name="facebook/esm2_t6_8M_UR50D",
        sequence_fusion_rank=32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_fragments = max_fragments
        self.max_attachments = max_attachments
        self.max_heavy_atoms = max_heavy_atoms
        self.pocket_encoder_mode = pocket_encoder_mode
        self.protein_embedding = nn.Linear(protein_feature_dim, hidden_dim)
        self.multiscale_pocket_encoder = MultiscalePocketEncoder(
            hidden_dim,
            mode=pocket_encoder_mode,
            residue_mix_floor=residue_mix_floor,
            atom_branch_dropout=atom_branch_dropout,
            atom_branch_dropout_weight=atom_branch_dropout_weight,
            fusion_gate_bias=fusion_gate_bias,
            hypergraph_geometry=hypergraph_geometry,
            esm2_model_name=esm2_model_name,
            sequence_fusion_rank=sequence_fusion_rank,
        )
        self.fragment_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.layers = nn.ModuleList(
            [FragmentEquivariantLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.token_head = nn.Linear(hidden_dim, vocab_size)
        self.attachment_head = nn.Linear(hidden_dim, max_attachments + 1)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_fragments),
        )
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_heavy_atoms),
        )
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.pocket_compatibility_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_pocket(self, protein_pos, protein_feature, protein_batch):
        protein_h = self.protein_embedding(protein_feature)
        return self.multiscale_pocket_encoder(
            protein_h,
            protein_pos,
            protein_feature,
            protein_batch,
            return_residue_context=True,
        )

    def forward(
        self,
        protein_pos,
        protein_feature,
        protein_batch,
        fragment_pos,
        fragment_token,
        fragment_batch,
        flow_time,
        heavy_atom_condition,
        qed_condition,
        condition_mask=None,
        encoded_pocket=None,
    ):
        if encoded_pocket is None:
            protein_h, residue_context = self.encode_pocket(
                protein_pos, protein_feature, protein_batch
            )
        else:
            protein_h, residue_context = encoded_pocket
        graph_condition = self.condition_embedding(
            torch.stack(
                [
                    heavy_atom_condition.float() / self.max_heavy_atoms,
                    qed_condition.float(),
                ],
                dim=-1,
            )
        )
        if condition_mask is not None:
            mask = condition_mask.to(graph_condition).unsqueeze(-1)
            graph_condition = (
                mask * graph_condition
                + (1.0 - mask) * self.null_condition.unsqueeze(0)
            )
        fragment_h = self.fragment_embedding(fragment_token)
        fragment_h = fragment_h + self.time_embedding(
            sinusoidal_time_embedding(flow_time[fragment_batch], self.hidden_dim)
        ) + graph_condition[fragment_batch]
        ff_index = _pair_index(fragment_batch)
        fp_index = _cross_index(fragment_batch, protein_batch)
        fr_index = None
        if residue_context is not None:
            fr_index = _cross_index(fragment_batch, residue_context[2])
        for layer in self.layers:
            fragment_h, fragment_pos = layer(
                fragment_h,
                fragment_pos,
                protein_h,
                protein_pos,
                ff_index,
                fp_index,
                residue_context=residue_context,
                fr_index=fr_index,
            )

        pair_index = torch.triu_indices(
            fragment_h.size(0), fragment_h.size(0), offset=1, device=fragment_h.device
        )
        same_graph = fragment_batch[pair_index[0]] == fragment_batch[pair_index[1]]
        pair_index = pair_index[:, same_graph]
        src, dst = pair_index
        distance = (fragment_pos[src] - fragment_pos[dst]).norm(dim=-1, keepdim=True)
        pair_feature = torch.cat(
            [fragment_h[src] + fragment_h[dst], (fragment_h[src] - fragment_h[dst]).abs(), distance],
            dim=-1,
        )
        protein_graph_h = scatter_mean(protein_h, protein_batch, dim=0)
        fragment_graph_h = scatter_mean(fragment_h, fragment_batch, dim=0)
        return {
            "fragment_pos": fragment_pos,
            "token_logits": self.token_head(fragment_h),
            "attachment_logits": self.attachment_head(fragment_h),
            "pair_index": pair_index,
            "edge_logits": self.edge_head(pair_feature).squeeze(-1),
            "count_logits": self.count_head(
                torch.cat([protein_graph_h, graph_condition], dim=-1)
            ),
            "size_logits": self.size_head(protein_graph_h),
            "qed_prediction": self.property_head(fragment_graph_h).squeeze(-1).sigmoid(),
            "fragment_graph_h": fragment_graph_h,
            "protein_graph_h": protein_graph_h,
        }

    def get_loss(
        self,
        protein_pos,
        protein_feature,
        protein_batch,
        target_fragment_pos,
        target_fragment_token,
        target_attachment_count,
        fragment_batch,
        target_edge_index,
        token_class_weight=None,
        token_size_lookup=None,
        target_heavy_atom_count=None,
        target_qed=None,
        condition_dropout=0.15,
        property_predictor=None,
        property_loss_weight=2.0,
        junction_distance_loss_weight=1.0,
        protein_repulsion_loss_weight=1.0,
        pocket_contrastive_loss_weight=0.0,
        pocket_contrastive_temperature=0.1,
        pocket_compatibility_loss_weight=0.0,
    ):
        num_graphs = int(protein_batch.max().item()) + 1
        protein_center = scatter_mean(protein_pos, protein_batch, dim=0)
        protein_pos = protein_pos - protein_center[protein_batch]
        target_fragment_pos = target_fragment_pos - protein_center[fragment_batch]
        flow_time = torch.rand(num_graphs, device=protein_pos.device)
        source_pos = torch.randn_like(target_fragment_pos)
        atom_time = flow_time[fragment_batch].unsqueeze(-1)
        fragment_pos_t = (1.0 - atom_time) * source_pos + atom_time * target_fragment_pos
        corruption = torch.rand_like(target_fragment_token, dtype=torch.float32) > flow_time[fragment_batch]
        random_token = torch.randint(
            self.vocab_size, target_fragment_token.shape, device=target_fragment_token.device
        )
        fragment_token_t = torch.where(corruption, random_token, target_fragment_token)
        condition_mask = (
            torch.rand(num_graphs, device=protein_pos.device) >= condition_dropout
        ).float()
        output = self(
            protein_pos,
            protein_feature,
            protein_batch,
            fragment_pos_t,
            fragment_token_t,
            fragment_batch,
            flow_time,
            target_heavy_atom_count,
            target_qed,
            condition_mask,
        )
        loss_pos = scatter_mean(
            ((output["fragment_pos"] - target_fragment_pos) ** 2).sum(dim=-1),
            fragment_batch,
            dim=0,
        ).mean()
        loss_token = F.cross_entropy(
            output["token_logits"], target_fragment_token, weight=token_class_weight
        )
        attachment_target = target_attachment_count.clamp(max=self.max_attachments)
        loss_attachment = F.cross_entropy(output["attachment_logits"], attachment_target)

        pair_index = output["pair_index"]
        target_adjacency = torch.zeros(
            (target_fragment_pos.size(0), target_fragment_pos.size(0)),
            dtype=torch.bool,
            device=target_fragment_pos.device,
        )
        if target_edge_index.numel():
            target_adjacency[target_edge_index[0], target_edge_index[1]] = True
            target_adjacency[target_edge_index[1], target_edge_index[0]] = True
        edge_target = target_adjacency[pair_index[0], pair_index[1]].float()
        positive = edge_target.sum().clamp(min=1.0)
        negative = (1.0 - edge_target).sum().clamp(min=1.0)
        positive_weight = (negative / positive).clamp(max=20.0)
        loss_edge = F.binary_cross_entropy_with_logits(
            output["edge_logits"], edge_target, pos_weight=positive_weight
        )
        if target_edge_index.numel():
            junction_src, junction_dst = target_edge_index
            pred_junction_distance = (
                output["fragment_pos"][junction_src]
                - output["fragment_pos"][junction_dst]
            ).norm(dim=-1)
            target_junction_distance = (
                target_fragment_pos[junction_src] - target_fragment_pos[junction_dst]
            ).norm(dim=-1)
            loss_junction_distance = F.smooth_l1_loss(
                pred_junction_distance,
                target_junction_distance,
                beta=0.2,
            )
        else:
            loss_junction_distance = loss_pos.new_zeros(())

        repulsion_losses = []
        for graph_index in range(num_graphs):
            fragment_mask = fragment_batch == graph_index
            protein_mask = protein_batch == graph_index
            if not fragment_mask.any() or not protein_mask.any():
                continue
            predicted_minimum = torch.cdist(
                output["fragment_pos"][fragment_mask], protein_pos[protein_mask]
            ).min(dim=-1).values
            target_minimum = torch.cdist(
                target_fragment_pos[fragment_mask], protein_pos[protein_mask]
            ).min(dim=-1).values.detach().clamp(min=1.5, max=4.0)
            repulsion_losses.append(
                F.relu(target_minimum - predicted_minimum).square().mean()
            )
        loss_protein_repulsion = (
            torch.stack(repulsion_losses).mean()
            if repulsion_losses else loss_pos.new_zeros(())
        )
        fragment_count = scatter_sum(
            torch.ones_like(fragment_batch), fragment_batch, dim=0
        ).clamp(max=self.max_fragments) - 1
        loss_count = F.cross_entropy(output["count_logits"], fragment_count)
        size_target = target_heavy_atom_count.clamp(
            min=1, max=self.max_heavy_atoms
        ) - 1
        loss_size = F.cross_entropy(output["size_logits"], size_target)
        token_probability = output["token_logits"].softmax(dim=-1)
        expected_fragment_size = token_probability @ token_size_lookup.to(
            token_probability
        )
        expected_molecule_size = scatter_sum(
            expected_fragment_size, fragment_batch, dim=0
        )
        loss_size_consistency = F.smooth_l1_loss(
            expected_molecule_size / self.max_heavy_atoms,
            target_heavy_atom_count.float() / self.max_heavy_atoms,
            beta=0.05,
        )
        if property_predictor is None:
            property_prediction = output["qed_prediction"]
        else:
            attachment_probability = output["attachment_logits"].softmax(dim=-1)
            hard_token = F.one_hot(
                token_probability.argmax(dim=-1), num_classes=token_probability.size(-1)
            ).to(token_probability)
            hard_attachment = F.one_hot(
                attachment_probability.argmax(dim=-1),
                num_classes=attachment_probability.size(-1),
            ).to(attachment_probability)
            straight_through_token = (
                hard_token - token_probability.detach() + token_probability
            )
            straight_through_attachment = (
                hard_attachment
                - attachment_probability.detach()
                + attachment_probability
            )
            property_prediction = property_predictor.forward_soft(
                straight_through_token,
                straight_through_attachment,
                fragment_batch,
                token_size_lookup,
            )
        loss_qed = F.mse_loss(property_prediction, target_qed.float())
        if num_graphs > 1 and pocket_contrastive_loss_weight > 0.0:
            fragment_graph_h = F.normalize(output["fragment_graph_h"], dim=-1)
            protein_graph_h = F.normalize(output["protein_graph_h"], dim=-1)
            pocket_logits = (
                fragment_graph_h @ protein_graph_h.transpose(0, 1)
            ) / pocket_contrastive_temperature
            pocket_target = torch.arange(num_graphs, device=protein_pos.device)
            loss_pocket_contrastive = 0.5 * (
                F.cross_entropy(pocket_logits, pocket_target)
                + F.cross_entropy(pocket_logits.transpose(0, 1), pocket_target)
            )
        else:
            loss_pocket_contrastive = loss_pos.new_zeros(())
        if num_graphs > 1 and pocket_compatibility_loss_weight > 0.0:
            fragment_graph_h = output["fragment_graph_h"]
            protein_graph_h = output["protein_graph_h"]

            def compatibility_logits(fragment_h, pocket_h):
                return self.pocket_compatibility_head(
                    torch.cat(
                        [
                            fragment_h,
                            pocket_h,
                            (fragment_h - pocket_h).abs(),
                            fragment_h * pocket_h,
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)

            positive_logits = compatibility_logits(fragment_graph_h, protein_graph_h)
            negative_pocket_h = protein_graph_h.roll(shifts=1, dims=0)
            negative_logits = compatibility_logits(fragment_graph_h, negative_pocket_h)
            loss_pocket_compatibility = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    positive_logits, torch.ones_like(positive_logits)
                )
                + F.binary_cross_entropy_with_logits(
                    negative_logits, torch.zeros_like(negative_logits)
                )
            )
        else:
            loss_pocket_compatibility = loss_pos.new_zeros(())
        loss = (
            loss_pos
            + loss_token
            + loss_attachment
            + loss_edge
            + 1.5 * loss_count
            + 0.5 * loss_size
            + 2.0 * loss_size_consistency
            + property_loss_weight * loss_qed
            + junction_distance_loss_weight * loss_junction_distance
            + protein_repulsion_loss_weight * loss_protein_repulsion
            + pocket_contrastive_loss_weight * loss_pocket_contrastive
            + pocket_compatibility_loss_weight * loss_pocket_compatibility
        )
        edge_prediction = output["edge_logits"] > 0
        edge_truth = edge_target.bool()
        true_positive = (edge_prediction & edge_truth).float().sum()
        return {
            "loss": loss,
            "loss_pos": loss_pos,
            "loss_token": loss_token,
            "loss_attachment": loss_attachment,
            "loss_edge": loss_edge,
            "loss_count": loss_count,
            "loss_size": loss_size,
            "loss_size_consistency": loss_size_consistency,
            "loss_qed": loss_qed,
            "loss_junction_distance": loss_junction_distance,
            "loss_protein_repulsion": loss_protein_repulsion,
            "loss_pocket_contrastive": loss_pocket_contrastive,
            "loss_pocket_compatibility": loss_pocket_compatibility,
            "token_accuracy": (
                output["token_logits"].argmax(dim=-1) == target_fragment_token
            ).float().mean(),
            "attachment_accuracy": (
                output["attachment_logits"].argmax(dim=-1) == attachment_target
            ).float().mean(),
            "edge_precision": true_positive / edge_prediction.float().sum().clamp(min=1.0),
            "edge_recall": true_positive / edge_truth.float().sum().clamp(min=1.0),
            "count_accuracy": (
                output["count_logits"].argmax(dim=-1) == fragment_count
            ).float().mean(),
            "size_accuracy": (
                output["size_logits"].argmax(dim=-1) == size_target
            ).float().mean(),
            "heavy_atom_mae": (
                expected_molecule_size - target_heavy_atom_count.float()
            ).abs().mean(),
            "qed_mae": (property_prediction - target_qed.float()).abs().mean(),
        }

    @staticmethod
    def decode_coarse_edges(pair_index, edge_probability, attachment_capacity, threshold=0.5):
        """Maximum-score connected graph under fragment attachment capacities."""
        num_fragments = len(attachment_capacity)
        remaining = [int(value) for value in attachment_capacity]
        parent = list(range(num_fragments))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        candidates = sorted(
            [
                (float(edge_probability[index]), int(pair_index[0, index]), int(pair_index[1, index]))
                for index in range(pair_index.size(1))
            ],
            reverse=True,
        )
        selected = []
        for probability, src, dst in candidates:
            if find(src) == find(dst) or remaining[src] <= 0 or remaining[dst] <= 0:
                continue
            selected.append((src, dst, probability))
            remaining[src] -= 1
            remaining[dst] -= 1
            parent[find(dst)] = find(src)
        if num_fragments > 1 and len(selected) < num_fragments - 1:
            probability_lookup = {
                (min(src, dst), max(src, dst)): probability
                for probability, src, dst in candidates
            }
            degrees = [1] * num_fragments
            extra_degree = num_fragments - 2
            node_priority = sorted(
                range(num_fragments),
                key=lambda node: (
                    attachment_capacity[node] - 1,
                    sum(
                        probability_lookup.get((min(node, other), max(node, other)), 0.0)
                        for other in range(num_fragments)
                        if other != node
                    ),
                ),
                reverse=True,
            )
            for node in node_priority:
                increment = min(
                    extra_degree, max(0, int(attachment_capacity[node]) - 1)
                )
                degrees[node] += increment
                extra_degree -= increment
                if extra_degree == 0:
                    break
            if extra_degree != 0:
                return []
            prufer = [node for node, degree in enumerate(degrees) for _ in range(degree - 1)]
            tree_degree = degrees[:]
            selected = []
            for parent_node in prufer:
                leaves = [node for node, degree in enumerate(tree_degree) if degree == 1]
                leaf = max(
                    leaves,
                    key=lambda node: probability_lookup.get(
                        (min(node, parent_node), max(node, parent_node)), 0.0
                    ),
                )
                probability = probability_lookup[(min(leaf, parent_node), max(leaf, parent_node))]
                selected.append((min(leaf, parent_node), max(leaf, parent_node), probability))
                tree_degree[leaf] -= 1
                tree_degree[parent_node] -= 1
            leaves = [node for node, degree in enumerate(tree_degree) if degree == 1]
            src, dst = leaves
            selected.append(
                (src, dst, probability_lookup[(min(src, dst), max(src, dst))])
            )
            remaining = [
                int(capacity) - degrees[index]
                for index, capacity in enumerate(attachment_capacity)
            ]
            return selected
        for probability, src, dst in candidates:
            if probability < threshold or remaining[src] <= 0 or remaining[dst] <= 0:
                continue
            if any(a == src and b == dst for a, b, _ in selected):
                continue
            selected.append((src, dst, probability))
            remaining[src] -= 1
            remaining[dst] -= 1
        return selected

    @staticmethod
    def decode_tokens_with_size(
        token_logits,
        token_size_lookup,
        token_capacity_lookup,
        target_size,
        tolerance=1,
        topk=128,
    ):
        """Jointly decode fragment tokens under exact size and tree-feasibility constraints."""
        num_fragments, vocab_size = token_logits.shape
        required_capacity = 2 * max(num_fragments - 1, 0)
        log_probability = token_logits.log_softmax(dim=-1)
        log_probability[:, 0] = -torch.inf
        topk = min(topk, vocab_size - 1)
        option_scores, option_tokens = log_probability.topk(topk, dim=-1)
        option_scores = option_scores.cpu().tolist()
        option_tokens = option_tokens.cpu().tolist()
        token_sizes = token_size_lookup.cpu().tolist()
        token_capacities = token_capacity_lookup.cpu().tolist()
        states = {(0, 0): (0.0, [])}
        max_size = int(target_size) + int(tolerance)
        for node in range(num_fragments):
            next_states = {}
            for (current_size, current_capacity), (score, tokens) in states.items():
                for option_index in range(topk):
                    token = int(option_tokens[node][option_index])
                    size = int(token_sizes[token])
                    capacity = int(token_capacities[token])
                    if size <= 0 or (num_fragments > 1 and capacity <= 0):
                        continue
                    new_size = current_size + size
                    if new_size > max_size:
                        continue
                    new_capacity = min(required_capacity, current_capacity + max(capacity, 0))
                    key = (new_size, new_capacity)
                    new_score = score + float(option_scores[node][option_index])
                    if key not in next_states or new_score > next_states[key][0]:
                        next_states[key] = (new_score, tokens + [token])
            states = next_states
            if not states:
                return None
        candidates = [
            (abs(size - int(target_size)), -score, tokens)
            for (size, capacity), (score, tokens) in states.items()
            if capacity >= required_capacity and abs(size - int(target_size)) <= tolerance
        ]
        if not candidates:
            return None
        tokens = min(candidates)[2]
        return torch.tensor(tokens, dtype=torch.long, device=token_logits.device)

    @torch.no_grad()
    def sample(
        self,
        protein_pos,
        protein_feature,
        protein_element=None,
        num_steps=100,
        edge_threshold=0.5,
        token_temperature=0.9,
        token_capacity_lookup=None,
        token_size_lookup=None,
        token_allowed_lookup=None,
        desired_heavy_atoms=None,
        desired_qed=0.7,
        guidance_scale=2.0,
        size_tolerance=1,
        token_search_topk=128,
        affinity_head=None,
        affinity_scale=0.0,
        affinity_start=0.5,
        affinity_grad_clip=3.0,
        fragment_affinity_ranker=None,
        token_feature_lookup=None,
        fragment_affinity_scale=0.0,
        fragment_affinity_steps=8,
        fragment_affinity_grad_clip=3.0,
        multiproperty_predictor=None,
        multiproperty_logp_mean=0.0,
        multiproperty_logp_std=1.0,
        property_qed_floor=0.58,
        property_qed_weight=4.0,
        property_logp_min=2.0,
        property_logp_max=3.3,
        property_logp_weight=0.25,
        constrained_decoding=True,
    ):
        device = protein_pos.device
        protein_batch = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)
        protein_center = protein_pos.mean(dim=0, keepdim=True)
        centered_protein = protein_pos - protein_center
        encoded_pocket = self.encode_pocket(
            centered_protein, protein_feature, protein_batch
        )
        protein_h = encoded_pocket[0]
        protein_graph_h = protein_h.mean(dim=0, keepdim=True)
        size_logits = self.size_head(protein_graph_h)
        if desired_heavy_atoms is None:
            heavy_atom_condition = size_logits.argmax(dim=-1) + 1
        else:
            heavy_atom_condition = torch.tensor(
                [desired_heavy_atoms], dtype=torch.long, device=device
            ).clamp(min=1, max=self.max_heavy_atoms)
        qed_condition = torch.tensor(
            [desired_qed], dtype=protein_pos.dtype, device=device
        )
        graph_condition = self.condition_embedding(
            torch.stack(
                [
                    heavy_atom_condition.float() / self.max_heavy_atoms,
                    qed_condition,
                ],
                dim=-1,
            )
        )
        count_logits = self.count_head(
            torch.cat([protein_graph_h, graph_condition], dim=-1)
        )
        if guidance_scale != 1.0:
            unconditional_count = self.count_head(
                torch.cat(
                    [protein_graph_h, self.null_condition.unsqueeze(0)], dim=-1
                )
            )
            count_logits = unconditional_count + guidance_scale * (
                count_logits - unconditional_count
            )
        num_fragments = int(count_logits.argmax(dim=-1).item()) + 1
        fragment_batch = torch.zeros(num_fragments, dtype=torch.long, device=device)
        fragment_pos = torch.randn(num_fragments, 3, device=device)
        fragment_token = torch.randint(self.vocab_size, (num_fragments,), device=device)
        output = None
        for step in range(num_steps):
            flow_time = torch.tensor([step / num_steps], dtype=protein_pos.dtype, device=device)
            conditional_output = self(
                centered_protein,
                protein_feature,
                protein_batch,
                fragment_pos,
                fragment_token,
                fragment_batch,
                flow_time,
                heavy_atom_condition,
                qed_condition,
                encoded_pocket=encoded_pocket,
            )
            unconditional_output = self(
                centered_protein,
                protein_feature,
                protein_batch,
                fragment_pos,
                fragment_token,
                fragment_batch,
                flow_time,
                heavy_atom_condition,
                qed_condition,
                torch.zeros(1, device=device),
                encoded_pocket=encoded_pocket,
            )
            output = {}
            for key in (
                "fragment_pos",
                "token_logits",
                "attachment_logits",
                "edge_logits",
            ):
                output[key] = unconditional_output[key] + guidance_scale * (
                    conditional_output[key] - unconditional_output[key]
                )
            output["pair_index"] = conditional_output["pair_index"]
            output["qed_prediction"] = conditional_output["qed_prediction"]
            remaining = max(1.0 - step / num_steps, 1.0 / num_steps)
            velocity = (output["fragment_pos"] - fragment_pos) / remaining
            fragment_pos = fragment_pos + velocity / num_steps
            if affinity_head is not None and affinity_scale > 0.0:
                progress = (step + 1) / num_steps
                if progress >= affinity_start:
                    with torch.enable_grad():
                        guided_pos = fragment_pos.detach().requires_grad_(True)
                        affinity_output = self(
                            centered_protein,
                            protein_feature,
                            protein_batch,
                            guided_pos,
                            fragment_token,
                            fragment_batch,
                            flow_time,
                            heavy_atom_condition,
                            qed_condition,
                            encoded_pocket=encoded_pocket,
                        )
                        normalized_vina, _ = affinity_head(
                            affinity_output["fragment_graph_h"],
                            affinity_output["protein_graph_h"],
                        )
                        affinity_gradient = torch.autograd.grad(
                            normalized_vina.sum(), guided_pos, only_inputs=True
                        )[0]
                    gradient_rms = affinity_gradient.square().mean().sqrt().clamp(min=1e-8)
                    affinity_direction = (affinity_gradient / gradient_rms).clamp(
                        min=-affinity_grad_clip, max=affinity_grad_clip
                    )
                    ramp = (progress - affinity_start) / max(1.0 - affinity_start, 1e-8)
                    fragment_pos = fragment_pos - (
                        affinity_scale * ramp * affinity_direction / num_steps
                    )
            token_probability = F.softmax(
                output["token_logits"] / token_temperature, dim=-1
            )
            token_probability[:, 0] = 0.0
            if token_allowed_lookup is not None:
                token_probability = token_probability * token_allowed_lookup.to(
                    token_probability
                ).unsqueeze(0)
            token_probability = token_probability / token_probability.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-8)
            fragment_token = torch.multinomial(token_probability, num_samples=1).squeeze(-1)
        flow_time = torch.ones(1, dtype=protein_pos.dtype, device=device)
        conditional_output = self(
            centered_protein,
            protein_feature,
            protein_batch,
            fragment_pos,
            fragment_token,
            fragment_batch,
            flow_time,
            heavy_atom_condition,
            qed_condition,
            encoded_pocket=encoded_pocket,
        )
        unconditional_output = self(
            centered_protein,
            protein_feature,
            protein_batch,
            fragment_pos,
            fragment_token,
            fragment_batch,
            flow_time,
            heavy_atom_condition,
            qed_condition,
            torch.zeros(1, device=device),
            encoded_pocket=encoded_pocket,
        )
        output = {}
        for key in ("token_logits", "attachment_logits", "edge_logits"):
            output[key] = unconditional_output[key] + guidance_scale * (
                conditional_output[key] - unconditional_output[key]
            )
        output["pair_index"] = conditional_output["pair_index"]
        output["qed_prediction"] = conditional_output["qed_prediction"]
        if (
            fragment_affinity_ranker is not None
            and fragment_affinity_scale > 0.0
            and protein_element is not None
            and token_size_lookup is not None
            and token_feature_lookup is not None
        ):
            guided_logits = output["token_logits"].detach()
            predicted_attachment = output["attachment_logits"].argmax(dim=-1)
            provisional_edges = self.decode_coarse_edges(
                output["pair_index"].cpu(),
                output["edge_logits"].sigmoid().cpu(),
                predicted_attachment.cpu().tolist(),
                threshold=0.0,
            )
            if provisional_edges:
                undirected = torch.tensor(
                    [[int(edge[0]), int(edge[1])] for edge in provisional_edges],
                    dtype=torch.long,
                    device=device,
                ).t().contiguous()
                rank_edge_index = torch.cat([undirected, undirected.flip(0)], dim=1)
            else:
                rank_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            rank_protein_batch = torch.zeros(
                protein_pos.size(0), dtype=torch.long, device=device
            )
            rank_fragment_batch = torch.zeros(
                num_fragments, dtype=torch.long, device=device
            )
            for _ in range(max(int(fragment_affinity_steps), 1)):
                with torch.enable_grad():
                    logits_for_grad = guided_logits.detach().requires_grad_(True)
                    token_probability = F.softmax(
                        logits_for_grad / token_temperature, dim=-1
                    )
                    token_probability = token_probability.clone()
                    token_probability[:, 0] = 0.0
                    if token_allowed_lookup is not None:
                        token_probability = token_probability * token_allowed_lookup.to(
                            token_probability
                        ).unsqueeze(0)
                    token_probability = token_probability / token_probability.sum(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-8)
                    expected_size = token_probability @ token_size_lookup.float()
                    expected_feature = token_probability @ token_feature_lookup.float()
                    normalized_vina, _ = fragment_affinity_ranker(
                        protein_element.long(),
                        protein_pos,
                        rank_protein_batch,
                        fragment_token,
                        predicted_attachment,
                        expected_size,
                        rank_fragment_batch,
                        rank_edge_index,
                        expected_feature,
                        token_probability=token_probability,
                    )
                    objective = normalized_vina.sum()
                    if multiproperty_predictor is not None:
                        attachment_probability = F.softmax(
                            output["attachment_logits"].detach(), dim=-1
                        )
                        predicted_qed, normalized_logp = multiproperty_predictor.forward_soft(
                            token_probability,
                            attachment_probability,
                            rank_fragment_batch,
                            token_size_lookup,
                        )
                        predicted_logp = (
                            normalized_logp * multiproperty_logp_std
                            + multiproperty_logp_mean
                        )
                        qed_penalty = F.relu(property_qed_floor - predicted_qed).square()
                        logp_penalty = (
                            F.relu(property_logp_min - predicted_logp).square()
                            + F.relu(predicted_logp - property_logp_max).square()
                        )
                        objective = objective + (
                            property_qed_weight * qed_penalty
                            + property_logp_weight * logp_penalty
                        ).sum()
                    gradient = torch.autograd.grad(
                        objective, logits_for_grad, only_inputs=True
                    )[0]
                gradient_rms = gradient.square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                direction = (gradient / gradient_rms).clamp(
                    min=-fragment_affinity_grad_clip,
                    max=fragment_affinity_grad_clip,
                )
                guided_logits = guided_logits - fragment_affinity_scale * direction
            output["token_logits"] = guided_logits
        if (
            constrained_decoding
            and token_size_lookup is not None
            and token_capacity_lookup is not None
        ):
            constrained_logits = output["token_logits"] / token_temperature
            if token_allowed_lookup is not None:
                constrained_logits = constrained_logits.masked_fill(
                    ~token_allowed_lookup.bool().unsqueeze(0), -1e9
                )
            constrained_tokens = self.decode_tokens_with_size(
                constrained_logits,
                token_size_lookup,
                token_capacity_lookup,
                int(heavy_atom_condition.item()),
                tolerance=size_tolerance,
                topk=token_search_topk,
            )
            if constrained_tokens is not None:
                fragment_token = constrained_tokens
                conditional_output = self(
                    centered_protein,
                    protein_feature,
                    protein_batch,
                    fragment_pos,
                    fragment_token,
                    fragment_batch,
                    flow_time,
                    heavy_atom_condition,
                    qed_condition,
                    encoded_pocket=encoded_pocket,
                )
                unconditional_output = self(
                    centered_protein,
                    protein_feature,
                    protein_batch,
                    fragment_pos,
                    fragment_token,
                    fragment_batch,
                    flow_time,
                    heavy_atom_condition,
                    qed_condition,
                    torch.zeros(1, device=device),
                    encoded_pocket=encoded_pocket,
                )
                for key in ("attachment_logits", "edge_logits"):
                    output[key] = unconditional_output[key] + guidance_scale * (
                        conditional_output[key] - unconditional_output[key]
                    )
                output["pair_index"] = conditional_output["pair_index"]
                output["qed_prediction"] = conditional_output["qed_prediction"]
        if not constrained_decoding or token_capacity_lookup is None:
            attachment_capacity = output["attachment_logits"].argmax(dim=-1)
        else:
            final_token_probability = F.softmax(
                output["token_logits"] / token_temperature, dim=-1
            )
            final_token_probability[:, 0] = 0.0
            if token_allowed_lookup is not None:
                final_token_probability = final_token_probability * token_allowed_lookup.to(
                    final_token_probability
                ).unsqueeze(0)
            final_token_probability = final_token_probability / final_token_probability.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-8)
            predicted_capacity = output["attachment_logits"].argmax(dim=-1)
            for _ in range(32):
                attachment_capacity = token_capacity_lookup[fragment_token]
                attachment_capacity = torch.where(
                    attachment_capacity >= 0, attachment_capacity, predicted_capacity
                )
                feasible = (
                    num_fragments == 1
                    or (
                        bool((attachment_capacity > 0).all())
                        and int(attachment_capacity.sum()) >= 2 * (num_fragments - 1)
                    )
                )
                if feasible:
                    break
                fragment_token = torch.multinomial(
                    final_token_probability, num_samples=1
                ).squeeze(-1)
        edge_probability = output["edge_logits"].sigmoid()
        if constrained_decoding:
            edges = self.decode_coarse_edges(
                output["pair_index"].cpu(),
                edge_probability.cpu(),
                attachment_capacity.cpu().tolist(),
                threshold=edge_threshold,
            )
        else:
            pair_index_cpu = output["pair_index"].cpu()
            probability_cpu = edge_probability.cpu()
            edges = [
                (
                    int(pair_index_cpu[0, index]),
                    int(pair_index_cpu[1, index]),
                    float(probability_cpu[index]),
                )
                for index in range(pair_index_cpu.size(1))
                if float(probability_cpu[index]) >= edge_threshold
            ]
        predicted_vina = None
        fragment_predicted_vina = None
        multiproperty_qed = None
        multiproperty_logp = None
        if affinity_head is not None:
            affinity_output = self(
                centered_protein,
                protein_feature,
                protein_batch,
                fragment_pos,
                fragment_token,
                fragment_batch,
                flow_time,
                heavy_atom_condition,
                qed_condition,
                encoded_pocket=encoded_pocket,
            )
            normalized_vina, _ = affinity_head(
                affinity_output["fragment_graph_h"],
                affinity_output["protein_graph_h"],
            )
            predicted_vina = float(normalized_vina.item())
        if (
            fragment_affinity_ranker is not None
            and protein_element is not None
            and token_size_lookup is not None
            and token_feature_lookup is not None
        ):
            final_edge_index = torch.tensor(
                [[int(edge[0]), int(edge[1])] for edge in edges],
                dtype=torch.long,
                device=device,
            ).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long, device=device)
            if final_edge_index.numel():
                final_edge_index = torch.cat([final_edge_index, final_edge_index.flip(0)], dim=1)
            normalized_vina, _ = fragment_affinity_ranker(
                protein_element.long(),
                protein_pos,
                torch.zeros(protein_pos.size(0), dtype=torch.long, device=device),
                fragment_token,
                attachment_capacity,
                token_size_lookup[fragment_token].float(),
                torch.zeros(num_fragments, dtype=torch.long, device=device),
                final_edge_index,
                token_feature_lookup[fragment_token],
            )
            fragment_predicted_vina = float(normalized_vina.item())
        if multiproperty_predictor is not None and token_size_lookup is not None:
            multiproperty_qed_tensor, normalized_logp = multiproperty_predictor(
                fragment_token,
                attachment_capacity,
                torch.zeros(num_fragments, dtype=torch.long, device=device),
                token_size_lookup,
            )
            multiproperty_qed = float(multiproperty_qed_tensor.item())
            multiproperty_logp = float(
                normalized_logp.item() * multiproperty_logp_std
                + multiproperty_logp_mean
            )
        return {
            "fragment_pos": fragment_pos + protein_center,
            "fragment_token": fragment_token,
            "attachment_capacity": attachment_capacity,
            "edges": edges,
            "count_probability": count_logits.softmax(dim=-1),
            "predicted_heavy_atoms": int(size_logits.argmax(dim=-1).item()) + 1,
            "heavy_atom_condition": int(heavy_atom_condition.item()),
            "qed_prediction": float(output["qed_prediction"].item()),
            "decoded_heavy_atoms": (
                int(token_size_lookup[fragment_token].sum())
                if token_size_lookup is not None
                else None
            ),
            "normalized_vina_prediction": predicted_vina,
            "fragment_normalized_vina_prediction": fragment_predicted_vina,
            "multiproperty_qed_prediction": multiproperty_qed,
            "multiproperty_logp_prediction": multiproperty_logp,
        }
