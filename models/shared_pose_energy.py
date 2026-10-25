import torch
import torch.nn as nn
import torch.nn.functional as F

from models.atomic_affinity import AtomicPocketAffinity


_VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
_CHARGE_PROXY = {6: 0.10, 7: -0.30, 8: -0.50, 9: -0.20,
                 15: 0.30, 16: -0.20, 17: -0.15, 35: -0.10, 53: -0.08}


def _lookup(atomic_numbers, table, default):
    values = torch.full_like(atomic_numbers, float(default), dtype=torch.float32)
    for number, value in table.items():
        values = torch.where(atomic_numbers == number, values.new_tensor(value), values)
    return values


def physical_pair_features(protein_z, protein_pos, ligand_z, ligand_pos,
                           protein_mask=None, ligand_mask=None):
    """Eight smooth, size-normalized interaction proxies for an energy head."""
    if protein_z.dim() == 1:
        protein_z, protein_pos = protein_z[None], protein_pos[None]
        ligand_z, ligand_pos = ligand_z[None], ligand_pos[None]
        protein_mask = torch.ones_like(protein_z, dtype=torch.bool)
        ligand_mask = torch.ones_like(ligand_z, dtype=torch.bool)
        squeeze = True
    else:
        squeeze = False
    pair_mask = ligand_mask.unsqueeze(2) & protein_mask.unsqueeze(1)
    difference = ligand_pos.unsqueeze(2) - protein_pos.unsqueeze(1)
    distance = (difference.square().sum(-1) + 1e-8).sqrt().clamp(max=12.0)
    ligand_radius = _lookup(ligand_z, _VDW, 1.70).unsqueeze(2)
    protein_radius = _lookup(protein_z, _VDW, 1.70).unsqueeze(1)
    radius_sum = ligand_radius + protein_radius
    overlap = F.relu(0.80 * radius_sum - distance) * pair_mask
    contact = torch.exp(-((distance - radius_sum) / 0.60).square()) * pair_mask
    inverse = pair_mask / distance.clamp(min=1.0)
    ligand_charge = _lookup(ligand_z, _CHARGE_PROXY, 0.0).unsqueeze(2)
    protein_charge = _lookup(protein_z, _CHARGE_PROXY, 0.0).unsqueeze(1)
    electrostatic = ligand_charge * protein_charge * inverse
    ligand_polar = ((ligand_z == 7) | (ligand_z == 8) | (ligand_z == 16)).float().unsqueeze(2)
    protein_polar = ((protein_z == 7) | (protein_z == 8) | (protein_z == 16)).float().unsqueeze(1)
    polar_contact = contact * ligand_polar * protein_polar
    ligand_hydrophobic = ((ligand_z == 6) | (ligand_z == 16)).float().unsqueeze(2)
    protein_hydrophobic = ((protein_z == 6) | (protein_z == 16)).float().unsqueeze(1)
    hydrophobic_contact = contact * ligand_hydrophobic * protein_hydrophobic
    denominator = pair_mask.sum((1, 2)).clamp(min=1)
    ligand_count = ligand_mask.sum(1).clamp(min=1)
    masked_overlap = overlap.masked_fill(~pair_mask, -1.0)
    features = torch.stack([
        overlap.sum((1, 2)) / denominator,
        masked_overlap.amax((1, 2)).clamp(min=0.0),
        (overlap > 0).sum((1, 2)) / ligand_count,
        contact.sum((1, 2)) / ligand_count,
        inverse.sum((1, 2)) / denominator,
        electrostatic.sum((1, 2)) / ligand_count,
        polar_contact.sum((1, 2)) / ligand_count,
        hydrophobic_contact.sum((1, 2)) / ligand_count,
    ], dim=-1)
    return features[0] if squeeze else features


class SharedPoseEnergy(nn.Module):
    """One invariant energy surface shared by raw and docked poses."""

    def __init__(self, hidden_dim=96, physical_features=False):
        super().__init__()
        self.physical_features = bool(physical_features)
        self.encoder = AtomicPocketAffinity(hidden_dim=hidden_dim)
        self.physical_head = (
            nn.Sequential(
                nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1, bias=False)
            )
            if self.physical_features
            else None
        )

    def forward(self, batch):
        hidden = self.encoder.encode_batch(batch)
        energy = self.encoder.vina_head(hidden).squeeze(-1)
        if self.physical_head is not None:
            features = physical_pair_features(
                batch["protein_z"], batch["protein_pos"],
                batch["ligand_z"], batch["ligand_pos"],
                batch["protein_mask"], batch["ligand_mask"],
            )
            energy = energy + self.physical_head(features).squeeze(-1)
        return energy

    def forward_records(self, records):
        values, _ = self.encoder(records)
        if self.physical_head is not None:
            features = torch.stack([
                physical_pair_features(
                    record["protein_z"], record["protein_pos"],
                    record["ligand_z"], record["ligand_pos"],
                )
                for record in records
            ])
            values = values + self.physical_head(features).squeeze(-1)
        return values

    @staticmethod
    def transform_energy(value, scale=10.0):
        return torch.asinh(value / scale)

    @staticmethod
    def inverse_energy(value, scale=10.0):
        return torch.sinh(value) * scale
