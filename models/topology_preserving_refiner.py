import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.atomic_affinity import RadialBasis
from utils.junction_torsion import dihedral


_VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}


def vdw_radii(atomic_numbers, default=1.70):
    radii = torch.full_like(atomic_numbers, float(default), dtype=torch.float32)
    for atomic_number, radius in _VDW_RADII.items():
        radii = torch.where(
            atomic_numbers == atomic_number,
            radii.new_tensor(radius),
            radii,
        )
    return radii


def topology_masks(bond_type):
    adjacency = bond_type > 0
    count = adjacency.size(0)
    eye = torch.eye(count, dtype=torch.bool, device=adjacency.device)
    within_three = eye | adjacency
    reach = adjacency.float()
    for _ in range(2):
        reach = reach @ adjacency.float()
        within_three = within_three | (reach > 0)
    nonbond = torch.triu(~within_three, diagonal=1)
    bonds = torch.triu(adjacency, diagonal=1)
    return adjacency, bonds, nonbond


def angle_cosines(position, adjacency):
    values = []
    for center in range(position.size(0)):
        neighbors = torch.where(adjacency[center])[0]
        for left in range(neighbors.numel()):
            for right in range(left + 1, neighbors.numel()):
                first = F.normalize(position[neighbors[left]] - position[center], dim=0)
                second = F.normalize(position[neighbors[right]] - position[center], dim=0)
                values.append((first * second).sum())
    return torch.stack(values) if values else position.new_empty(0)


def torsions(position, indices):
    if indices is None or indices.numel() == 0:
        return position.new_empty(0)
    return torch.stack([dihedral(position, row) for row in indices])


def physical_terms(
    position,
    record,
    reference=None,
    clash_scale=0.75,
    protein_clash_scale=None,
):
    """Differentiable PoseBusters-oriented geometry proxies for one complex."""
    reference = record.get("geometry_reference_pos", record["raw_pos"]) if reference is None else reference
    conformation_reference = record.get("conformation_reference_pos", record["raw_pos"])
    adjacency, bond_mask, nonbond_mask = topology_masks(record["bond_type"])
    distance = torch.cdist(position, position)
    reference_distance = torch.cdist(reference, reference)
    zero = position.sum() * 0.0

    bond = (
        F.smooth_l1_loss(distance[bond_mask], reference_distance[bond_mask])
        if bond_mask.any()
        else zero
    )
    predicted_angles = angle_cosines(position, adjacency)
    reference_angles = angle_cosines(reference, adjacency)
    angle = (
        F.smooth_l1_loss(predicted_angles, reference_angles)
        if predicted_angles.numel()
        else zero
    )

    junction_index = record.get("junction_index")
    predicted_torsions = torsions(position, junction_index)
    reference_torsions = torsions(conformation_reference, junction_index)
    torsion = (
        (1.0 - torch.cos(predicted_torsions - reference_torsions)).mean()
        if predicted_torsions.numel()
        else zero
    )

    # Preserve the complete local conformer, not only covalent edges.
    conformation_distance = torch.cdist(conformation_reference, conformation_reference)
    # Pairs separated by at least four bonds capture fold/conformation without
    # fighting the bond and angle repair objectives.
    local_mask = (conformation_distance < 6.0) & nonbond_mask
    conformer = (
        F.smooth_l1_loss(distance[local_mask], conformation_distance[local_mask])
        if local_mask.any()
        else zero
    )

    ligand_radii = vdw_radii(record["ligand_z"]).to(position)
    if nonbond_mask.any():
        nonbond_limit = clash_scale * (
            ligand_radii[:, None] + ligand_radii[None, :]
        )
        intramolecular_clash = F.relu(
            nonbond_limit[nonbond_mask] - distance[nonbond_mask]
        ).square().mean()
    else:
        intramolecular_clash = zero

    protein_clash_scale = (
        clash_scale if protein_clash_scale is None else float(protein_clash_scale)
    )
    protein_pos = record["protein_pos"]
    if protein_pos.numel():
        protein_radii = vdw_radii(record["protein_z"]).to(position)
        cross_distance = torch.cdist(position, protein_pos)
        cross_limit = protein_clash_scale * (
            ligand_radii[:, None] + protein_radii[None, :]
        )
        protein_overlap = F.relu(cross_limit - cross_distance)
        protein_clash = protein_overlap.square().mean()
        minimum_contact = cross_distance.min(dim=1).values
        protein_clash_mask = cross_distance < cross_limit
        protein_clash_rate = protein_clash_mask.any(dim=1).float().mean()
        protein_clash_count = protein_clash_mask.float().sum()
        protein_overlap_max = protein_overlap.max()
        protein_clearance = (cross_distance - cross_limit).min()
    else:
        protein_clash = zero
        minimum_contact = position.new_zeros(position.size(0))
        protein_clash_rate = zero
        protein_clash_count = zero
        protein_overlap_max = zero
        protein_clearance = zero

    displacement = (position - conformation_reference).square().sum(dim=-1).mean()
    return {
        "bond": bond,
        "angle": angle,
        "torsion": torsion,
        "conformer": conformer,
        "intramolecular_clash": intramolecular_clash,
        "protein_clash": protein_clash,
        "protein_clash_rate": protein_clash_rate,
        "protein_clash_count": protein_clash_count,
        "protein_overlap_max": protein_overlap_max,
        "protein_clearance": protein_clearance,
        "minimum_contact": minimum_contact.mean(),
        "displacement": displacement,
    }


DEFAULT_PHYSICS_WEIGHTS = {
    "bond": 4.0,
    "angle": 2.0,
    "torsion": 1.5,
    "conformer": 1.0,
    "intramolecular_clash": 2.0,
    "protein_clash": 4.0,
    "displacement": 0.02,
}


def physical_objective(terms, weights=None):
    weights = DEFAULT_PHYSICS_WEIGHTS if weights is None else weights
    return sum(terms[key] * float(weight) for key, weight in weights.items())


class TopologyRefinerLayer(nn.Module):
    def __init__(self, hidden_dim, num_basis, cross_cutoff, ligand_cutoff):
        super().__init__()
        self.cross_cutoff = float(cross_cutoff)
        self.ligand_cutoff = float(ligand_cutoff)
        self.bond_embedding = nn.Embedding(5, hidden_dim)
        self.cross_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_basis, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ligand_message = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2 * num_basis, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.cross_coordinate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False))
        self.ligand_coordinate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False))
        self.hidden_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.cross_coordinate[-1].weight, std=1e-3)
        nn.init.normal_(self.ligand_coordinate[-1].weight, std=1e-3)

    def forward(self, ligand_h, protein_h, position, protein_pos, reference, bond_type, radial):
        cross_vector = position[:, None] - protein_pos[None]
        cross_distance = cross_vector.norm(dim=-1).clamp(min=1e-6)
        cross_mask = cross_distance < self.cross_cutoff
        cross_input = torch.cat([
            ligand_h[:, None].expand(-1, protein_h.size(0), -1),
            protein_h[None].expand(ligand_h.size(0), -1, -1),
            radial(cross_distance),
        ], dim=-1)
        cross_message = self.cross_message(cross_input) * cross_mask.unsqueeze(-1)
        cross_count = cross_mask.sum(1).clamp(min=1).sqrt().unsqueeze(-1)
        cross_aggregate = cross_message.sum(1) / cross_count
        cross_velocity = (
            cross_vector / cross_distance.unsqueeze(-1)
            * self.cross_coordinate(cross_message)
            * cross_mask.unsqueeze(-1)
        ).sum(1) / cross_count

        vector = position[:, None] - position[None]
        distance = vector.norm(dim=-1).clamp(min=1e-6)
        reference_distance = torch.cdist(reference, reference)
        eye = torch.eye(position.size(0), dtype=torch.bool, device=position.device)
        ligand_mask = (~eye) & ((distance < self.ligand_cutoff) | (bond_type > 0))
        ligand_input = torch.cat([
            ligand_h[:, None].expand(-1, ligand_h.size(0), -1),
            ligand_h[None].expand(ligand_h.size(0), -1, -1),
            self.bond_embedding(bond_type.clamp(0, 4)),
            radial(distance),
            radial(reference_distance),
        ], dim=-1)
        ligand_message = self.ligand_message(ligand_input) * ligand_mask.unsqueeze(-1)
        ligand_count = ligand_mask.sum(1).clamp(min=1).sqrt().unsqueeze(-1)
        ligand_aggregate = ligand_message.sum(1) / ligand_count
        ligand_velocity = (
            vector / distance.unsqueeze(-1)
            * self.ligand_coordinate(ligand_message)
            * ligand_mask.unsqueeze(-1)
        ).sum(1) / ligand_count
        update = self.hidden_update(torch.cat([ligand_h, cross_aggregate, ligand_aggregate], dim=-1))
        return self.norm(ligand_h + update), cross_velocity + ligand_velocity


class TopologyPreservingAtomicRefiner(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6, num_basis=24,
                 cross_cutoff=8.0, ligand_cutoff=6.0, internal_scale=0.15,
                 max_atomic_number=100):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.internal_scale = float(internal_scale)
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.protein_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.radial = RadialBasis(num_basis=num_basis, cutoff=max(cross_cutoff, ligand_cutoff))
        self.layers = nn.ModuleList([
            TopologyRefinerLayer(hidden_dim, num_basis, cross_cutoff, ligand_cutoff)
            for _ in range(num_layers)
        ])
        self.layer_scale = nn.Parameter(torch.full((num_layers,), 0.1))

    @staticmethod
    def remove_bond_stretch(position, velocity, bond_type, iterations=2):
        src, dst = torch.where(torch.triu(bond_type > 0, diagonal=1))
        if src.numel() == 0:
            return velocity
        degree = torch.bincount(torch.cat([src, dst]), minlength=position.size(0)).clamp(min=1).unsqueeze(-1)
        for _ in range(iterations):
            axis = F.normalize(position[src] - position[dst], dim=-1)
            relative = ((velocity[src] - velocity[dst]) * axis).sum(-1, keepdim=True) * axis
            correction = torch.zeros_like(velocity)
            correction.index_add_(0, src, -0.5 * relative)
            correction.index_add_(0, dst, 0.5 * relative)
            velocity = velocity + correction / degree
        return velocity

    @staticmethod
    def project_bond_lengths(position, reference, bond_type, iterations=12):
        src, dst = torch.where(torch.triu(bond_type > 0, diagonal=1))
        if src.numel() == 0:
            return position
        desired = (reference[src] - reference[dst]).norm(dim=-1).clamp(min=1e-6)
        degree = torch.bincount(torch.cat([src, dst]), minlength=position.size(0)).clamp(min=1).unsqueeze(-1)
        for _ in range(iterations):
            vector = position[src] - position[dst]
            current = vector.norm(dim=-1).clamp(min=1e-6)
            correction = 0.5 * (1.0 - desired / current).unsqueeze(-1) * vector
            accumulated = torch.zeros_like(position)
            accumulated.index_add_(0, src, -correction)
            accumulated.index_add_(0, dst, correction)
            position = position + accumulated / degree
        return position

    def forward_graph(self, record, position, flow_time):
        reference = record.get("geometry_reference_pos", record["raw_pos"])
        time_features = torch.stack([
            flow_time, torch.sin(math.pi * flow_time), torch.cos(math.pi * flow_time)
        ]).reshape(1, 3)
        ligand_h = self.atom_embedding(record["ligand_z"].long().clamp(0, 100))
        ligand_h = ligand_h + self.time_embedding(time_features)
        protein_h = self.protein_embedding(record["protein_z"].long().clamp(0, 100))
        velocity = torch.zeros_like(position)
        for index, layer in enumerate(self.layers):
            ligand_h, layer_velocity = layer(
                ligand_h, protein_h, position, record["protein_pos"], reference,
                record["bond_type"], self.radial,
            )
            velocity = velocity + self.layer_scale[index] * layer_velocity
        translation = velocity.mean(dim=0, keepdim=True)
        velocity = translation + self.internal_scale * (velocity - translation)
        return self.remove_bond_stretch(position, velocity, record["bond_type"])

    def forward(self, records, positions, flow_times):
        return [
            self.forward_graph(record, position, flow_time)
            for record, position, flow_time in zip(records, positions, flow_times)
        ]

    def _physics_descent(
        self, position, record, reference, step_size, weights, protein_clash_scale
    ):
        with torch.enable_grad():
            variable = position.detach().requires_grad_(True)
            objective = physical_objective(
                physical_terms(
                    variable,
                    record,
                    reference,
                    protein_clash_scale=protein_clash_scale,
                ),
                weights,
            )
            gradient = torch.autograd.grad(objective, variable)[0]
        gradient_norm = gradient.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        correction = -step_size * gradient * (1.0 / gradient_norm).clamp(max=1.0)
        return self.project_bond_lengths(
            position + correction.detach(), reference, record["bond_type"], iterations=1
        )

    @staticmethod
    def _rigid_repulsion_candidates(position, record, step_size=0.08, clash_scale=0.75):
        if not record["protein_pos"].numel():
            return []
        ligand_radii = vdw_radii(record["ligand_z"]).to(position)
        protein_radii = vdw_radii(record["protein_z"]).to(position)
        vector = position[:, None] - record["protein_pos"][None]
        distance = vector.norm(dim=-1).clamp(min=1e-6)
        limit = clash_scale * (ligand_radii[:, None] + protein_radii[None, :])
        overlap = F.relu(limit - distance)
        force = (vector / distance.unsqueeze(-1) * overlap.unsqueeze(-1)).sum(dim=(0, 1))
        directions = torch.eye(3, dtype=position.dtype, device=position.device)
        directions = torch.cat([directions, -directions], dim=0)
        if force.norm() >= 1e-8:
            directions = torch.cat([F.normalize(force, dim=0).unsqueeze(0), directions], dim=0)
        return [position + step_size * direction for direction in directions]

    def refine(self, record, steps=12, strength=0.5, max_step=0.25,
               physics_step=0.08, physics_iterations=2, weights=None,
               protein_clash_scale=0.75, preserve_protein_clearance=False,
               clearance_tolerance=0.0):
        """Refine coordinates while monotonically reducing physical proxy energy."""
        reference = record.get("geometry_reference_pos", record["raw_pos"])
        position = record["raw_pos"].clone()
        weights = DEFAULT_PHYSICS_WEIGHTS if weights is None else weights
        accepted = 0

        def measure(value):
            with torch.no_grad():
                terms = physical_terms(
                    value,
                    record,
                    reference,
                    protein_clash_scale=protein_clash_scale,
                )
                return terms, physical_objective(terms, weights)

        baseline_terms, _ = measure(position)

        def acceptable(before_terms, before_score, after_terms, after_score):
            geometry_before = 4.0 * before_terms["bond"] + 2.0 * before_terms["angle"]
            geometry_after = 4.0 * after_terms["bond"] + 2.0 * after_terms["angle"]
            protein_safe = True
            if preserve_protein_clearance:
                protein_safe = bool(
                    after_terms["protein_clash_count"]
                    <= baseline_terms["protein_clash_count"] + 1e-7
                    and after_terms["protein_overlap_max"]
                    <= baseline_terms["protein_overlap_max"] + 1e-7
                    and after_terms["protein_clearance"]
                    >= baseline_terms["protein_clearance"] - clearance_tolerance
                )
            return bool(
                torch.isfinite(after_score)
                and after_score <= before_score + 1e-7
                and after_terms["protein_clash"] <= baseline_terms["protein_clash"] + 1e-7
                and after_terms["intramolecular_clash"] <= baseline_terms["intramolecular_clash"] + 1e-7
                and geometry_after <= geometry_before + 1e-7
                and after_terms["torsion"] <= baseline_terms["torsion"] + 0.02
                and after_terms["conformer"] <= baseline_terms["conformer"] + 0.02
                and protein_safe
            )

        def descend_monotonically(value):
            current = value
            current_terms, current_score = measure(current)
            for _ in range(physics_iterations):
                trial = self._physics_descent(
                    current,
                    record,
                    reference,
                    physics_step,
                    weights,
                    protein_clash_scale,
                )
                trial_terms, trial_score = measure(trial)
                if acceptable(current_terms, current_score, trial_terms, trial_score):
                    current, current_terms, current_score = trial.detach(), trial_terms, trial_score
            return current, current_terms, current_score

        for step in range(steps):
            with torch.no_grad():
                time = position.new_tensor(step / max(steps, 1))
                velocity = self.forward_graph(record, position, time)
                delta = strength * velocity / max(steps, 1)
                norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                delta = delta * (max_step / norm).clamp(max=1.0)
                candidate = self.project_bond_lengths(
                    position + delta, reference, record["bond_type"], iterations=1
                )
            before_terms, before = measure(position)
            candidate, after_terms, after = descend_monotonically(candidate)
            if not acceptable(before_terms, before, after_terms, after):
                # A checkpoint can be used safely before convergence: rejected
                # learned moves fall back to a physics-only correction.
                candidate, after_terms, after = descend_monotonically(position)
            if not acceptable(before_terms, before, after_terms, after):
                rigid_options = []
                for rigid in self._rigid_repulsion_candidates(
                    position, record, physics_step, protein_clash_scale
                ):
                    rigid_terms, rigid_score = measure(rigid)
                    if acceptable(before_terms, before, rigid_terms, rigid_score):
                        rigid_options.append((float(rigid_score), rigid, rigid_terms, rigid_score))
                if rigid_options:
                    _, candidate, after_terms, after = min(rigid_options, key=lambda item: item[0])
            if acceptable(before_terms, before, after_terms, after):
                shift = (candidate - position).norm(dim=-1).mean()
                if shift > 1e-7:
                    position = candidate.detach()
                    accepted += 1
        projected = self.project_bond_lengths(
            position, reference, record["bond_type"], iterations=24
        )
        before_terms, before = measure(position)
        after_terms, after = measure(projected)
        if acceptable(before_terms, before, after_terms, after):
            position = projected
        return position, {"accepted_steps": accepted, "attempted_steps": steps}
