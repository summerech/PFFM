import math

import torch
import torch.nn as nn


def stable_cdist(left, right, epsilon=1e-12):
    """Euclidean distance with a finite Hessian at coincident coordinates."""
    difference = left.unsqueeze(-2) - right.unsqueeze(-3)
    return (difference.square().sum(-1) + epsilon).sqrt()


class RadialBasis(nn.Module):
    def __init__(self, num_basis=24, cutoff=8.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / float((centers[1] - centers[0]).square())
        self.cutoff = float(cutoff)

    def forward(self, distance):
        basis = torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        return basis * envelope.clamp(min=0.0).unsqueeze(-1)


class AtomicPocketAffinity(nn.Module):
    """Rotation/translation-invariant pocket-ligand contact predictor."""

    def __init__(self, hidden_dim=96, num_basis=24, cutoff=8.0, max_atomic_number=100):
        super().__init__()
        self.cutoff = float(cutoff)
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.radial = RadialBasis(num_basis=num_basis, cutoff=cutoff)
        pair_dim = 2 * hidden_dim + num_basis
        self.cross_message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.ligand_message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.atom_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 6, 2 * hidden_dim), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.vina_head = nn.Linear(hidden_dim, 1)
        self.ha_head = nn.Linear(hidden_dim, 1)

    def forward_graph(self, record):
        protein_z = record["protein_z"].long()
        protein_pos = record["protein_pos"].float()
        ligand_z = record["ligand_z"].long()
        ligand_pos = record["ligand_pos"].float()
        protein_h = self.atom_embedding(protein_z.clamp(min=0, max=100))
        ligand_h = self.atom_embedding(ligand_z.clamp(min=0, max=100))

        cross_distance = stable_cdist(ligand_pos, protein_pos)
        cross_index = torch.where(cross_distance < self.cutoff)
        cross_agg = ligand_h.new_zeros(ligand_h.shape)
        if cross_index[0].numel():
            li, pi = cross_index
            message = self.cross_message(torch.cat([
                ligand_h[li], protein_h[pi], self.radial(cross_distance[li, pi])
            ], dim=-1))
            cross_agg.index_add_(0, li, message)
            count = torch.bincount(li, minlength=ligand_h.size(0)).clamp(min=1).sqrt().unsqueeze(-1)
            cross_agg = cross_agg / count

        ligand_distance = stable_cdist(ligand_pos, ligand_pos)
        mask = (ligand_distance < 5.0) & (ligand_distance > 1e-4)
        ligand_index = torch.where(mask)
        ligand_agg = ligand_h.new_zeros(ligand_h.shape)
        if ligand_index[0].numel():
            dst, src = ligand_index
            message = self.ligand_message(torch.cat([
                ligand_h[dst], ligand_h[src], self.radial(ligand_distance[dst, src])
            ], dim=-1))
            ligand_agg.index_add_(0, dst, message)
            count = torch.bincount(dst, minlength=ligand_h.size(0)).clamp(min=1).sqrt().unsqueeze(-1)
            ligand_agg = ligand_agg / count

        ligand_h = self.atom_update(torch.cat([ligand_h, cross_agg, ligand_agg], dim=-1))
        minimum = cross_distance.min(dim=1).values
        contact_stats = torch.stack([
            (minimum < 2.5).float().mean(),
            (minimum < 3.5).float().mean(),
            (minimum < 4.5).float().mean(),
            minimum.mean() / self.cutoff,
            ligand_h.new_tensor(ligand_h.size(0) / 64.0),
            ligand_h.new_tensor(protein_h.size(0) / 256.0),
        ])
        graph = torch.cat([ligand_h.mean(0), ligand_h.max(0).values, contact_stats], dim=-1)
        hidden = self.graph_head(graph)
        return self.vina_head(hidden).squeeze(-1), self.ha_head(hidden).squeeze(-1)

    def forward(self, records):
        if isinstance(records, dict):
            return self.forward_batch(records)
        output = [self.forward_graph(record) for record in records]
        return torch.stack([item[0] for item in output]), torch.stack([item[1] for item in output])

    def encode_batch(self, batch):
        protein_h = self.atom_embedding(batch["protein_z"].clamp(min=0, max=100))
        ligand_h = self.atom_embedding(batch["ligand_z"].clamp(min=0, max=100))
        protein_mask = batch["protein_mask"]
        ligand_mask = batch["ligand_mask"]

        cross_distance = stable_cdist(batch["ligand_pos"], batch["protein_pos"])
        cross_mask = (
            ligand_mask.unsqueeze(2) & protein_mask.unsqueeze(1)
            & (cross_distance < self.cutoff)
        )
        batch_size, num_ligand, num_protein = cross_distance.shape
        cross_input = torch.cat([
            ligand_h.unsqueeze(2).expand(-1, -1, num_protein, -1),
            protein_h.unsqueeze(1).expand(-1, num_ligand, -1, -1),
            self.radial(cross_distance),
        ], dim=-1)
        cross_message = self.cross_message(cross_input) * cross_mask.unsqueeze(-1)
        cross_count = cross_mask.sum(2).clamp(min=1).sqrt().unsqueeze(-1)
        cross_agg = cross_message.sum(2) / cross_count

        ligand_distance = stable_cdist(batch["ligand_pos"], batch["ligand_pos"])
        ligand_pair_mask = (
            ligand_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)
            & (ligand_distance < 5.0) & (ligand_distance > 1e-4)
        )
        ligand_input = torch.cat([
            ligand_h.unsqueeze(2).expand(-1, -1, num_ligand, -1),
            ligand_h.unsqueeze(1).expand(-1, num_ligand, -1, -1),
            self.radial(ligand_distance),
        ], dim=-1)
        ligand_message = self.ligand_message(ligand_input) * ligand_pair_mask.unsqueeze(-1)
        ligand_count = ligand_pair_mask.sum(2).clamp(min=1).sqrt().unsqueeze(-1)
        ligand_agg = ligand_message.sum(2) / ligand_count

        ligand_h = self.atom_update(torch.cat([ligand_h, cross_agg, ligand_agg], dim=-1))
        ligand_h = ligand_h * ligand_mask.unsqueeze(-1)
        count = ligand_mask.sum(1).clamp(min=1).unsqueeze(-1)
        graph_mean = ligand_h.sum(1) / count
        graph_max = ligand_h.masked_fill(~ligand_mask.unsqueeze(-1), -1e4).max(1).values

        masked_distance = cross_distance.masked_fill(~protein_mask.unsqueeze(1), self.cutoff + 1.0)
        minimum = masked_distance.min(2).values
        denominator = ligand_mask.sum(1).clamp(min=1)
        def fraction(threshold):
            return (((minimum < threshold) & ligand_mask).sum(1) / denominator).float()
        contact_stats = torch.stack([
            fraction(2.5), fraction(3.5), fraction(4.5),
            (minimum.masked_fill(~ligand_mask, 0.0).sum(1) / denominator) / self.cutoff,
            denominator.float() / 64.0,
            protein_mask.sum(1).float() / 256.0,
        ], dim=-1)
        return self.graph_head(torch.cat([graph_mean, graph_max, contact_stats], dim=-1))

    def forward_batch(self, batch):
        hidden = self.encode_batch(batch)
        return self.vina_head(hidden).squeeze(-1), self.ha_head(hidden).squeeze(-1)
