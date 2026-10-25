import math

import torch

from utils.fragmentation import decompose_rbrics


def dihedral(position, indices):
    a, src, dst, b = [position[index] for index in indices]
    axis = dst - src
    axis = axis / axis.norm().clamp(min=1e-8)
    left = a - src
    right = b - dst
    left = left - (left * axis).sum() * axis
    right = right - (right * axis).sum() * axis
    left = left / left.norm().clamp(min=1e-8)
    right = right / right.norm().clamp(min=1e-8)
    return torch.atan2((torch.cross(axis, left, dim=0) * right).sum(), (left * right).sum())


def circular_delta(target, source):
    return torch.atan2(torch.sin(target - source), torch.cos(target - source))


def build_junction_torsions(molecule, raw_pos, target_pos=None):
    decomposition = decompose_rbrics(molecule)
    atom_to_fragment = decomposition["atom_to_fragment"]
    num_fragments = int(decomposition["num_fragments"])
    fragment_sizes = torch.bincount(atom_to_fragment, minlength=num_fragments)
    root = int(fragment_sizes.argmax())
    edges = decomposition["junction_fragment_index"].t().tolist()
    atoms = decomposition["junction_atom_index"].t().tolist()
    adjacency = [[] for _ in range(num_fragments)]
    for edge_index, (src_fragment, dst_fragment) in enumerate(edges):
        adjacency[src_fragment].append((dst_fragment, edge_index))
        adjacency[dst_fragment].append((src_fragment, edge_index))
    parent = {root: -1}
    parent_edge = {}
    order = [root]
    for fragment in order:
        for neighbor, edge_index in adjacency[fragment]:
            if neighbor in parent:
                continue
            parent[neighbor] = fragment
            parent_edge[neighbor] = edge_index
            order.append(neighbor)
    children = [[] for _ in range(num_fragments)]
    for child, parent_fragment in parent.items():
        if parent_fragment >= 0:
            children[parent_fragment].append(child)

    descendants = {}
    for fragment in reversed(order):
        values = {fragment}
        for child in children[fragment]:
            values.update(descendants[child])
        descendants[fragment] = values

    junction_indices, rotate_masks, target_deltas = [], [], []
    for child in order[1:]:
        edge_index = parent_edge[child]
        parent_fragment = parent[child]
        src_atom, dst_atom = atoms[edge_index]
        src_fragment, dst_fragment = edges[edge_index]
        if src_fragment == child:
            src_atom, dst_atom = dst_atom, src_atom
            src_fragment, dst_fragment = dst_fragment, src_fragment
        if src_fragment != parent_fragment or dst_fragment != child:
            continue
        src_neighbors = [
            atom.GetIdx()
            for atom in molecule.GetAtomWithIdx(src_atom).GetNeighbors()
            if atom.GetIdx() != dst_atom
            and int(atom_to_fragment[atom.GetIdx()]) == parent_fragment
        ]
        dst_neighbors = [
            atom.GetIdx()
            for atom in molecule.GetAtomWithIdx(dst_atom).GetNeighbors()
            if atom.GetIdx() != src_atom
            and int(atom_to_fragment[atom.GetIdx()]) == child
        ]
        if not src_neighbors or not dst_neighbors:
            continue
        a = max(src_neighbors, key=lambda index: molecule.GetAtomWithIdx(index).GetDegree())
        b = max(dst_neighbors, key=lambda index: molecule.GetAtomWithIdx(index).GetDegree())
        indices = [a, src_atom, dst_atom, b]
        mask = torch.tensor(
            [int(fragment) in descendants[child] for fragment in atom_to_fragment],
            dtype=torch.bool,
        )
        junction_indices.append(indices)
        rotate_masks.append(mask)
        if target_pos is not None:
            target_deltas.append(
                circular_delta(dihedral(target_pos, indices), dihedral(raw_pos, indices))
            )
    if not junction_indices:
        return {
            "junction_index": torch.empty((0, 4), dtype=torch.long),
            "rotate_mask": torch.empty((0, molecule.GetNumAtoms()), dtype=torch.bool),
            "target_delta": torch.empty((0,), dtype=torch.float32),
            "atom_to_fragment": decomposition["atom_to_fragment"],
            "fragment_atom_index": decomposition["fragment_atom_index"],
            "fragment_ptr": decomposition["fragment_ptr"],
            "junction_atom_index": decomposition["junction_atom_index"],
            "junction_fragment_index": decomposition["junction_fragment_index"],
            "internal_bond_index": decomposition["internal_bond_index"],
        }
    return {
        "junction_index": torch.tensor(junction_indices, dtype=torch.long),
        "rotate_mask": torch.stack(rotate_masks),
        "target_delta": (
            torch.stack(target_deltas).float()
            if target_pos is not None
            else torch.empty((len(junction_indices),), dtype=torch.float32)
        ),
        "atom_to_fragment": decomposition["atom_to_fragment"],
        "fragment_atom_index": decomposition["fragment_atom_index"],
        "fragment_ptr": decomposition["fragment_ptr"],
        "junction_atom_index": decomposition["junction_atom_index"],
        "junction_fragment_index": decomposition["junction_fragment_index"],
        "internal_bond_index": decomposition["internal_bond_index"],
    }


def rotate_subtree(position, src, dst, mask, angle):
    axis = position[dst] - position[src]
    axis = axis / axis.norm().clamp(min=1e-8)
    vector = position[mask] - position[src]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = (
        vector * cosine
        + torch.cross(axis.expand_as(vector), vector, dim=-1) * sine
        + axis * (vector * axis).sum(-1, keepdim=True) * (1.0 - cosine)
    )
    output = position.clone()
    output[mask] = rotated + position[src]
    return output
