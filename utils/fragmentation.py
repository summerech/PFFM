from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem

from utils import rbrics_public as rBRICS


BOND_TYPE_TO_INDEX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
INDEX_TO_BOND_TYPE = {value: key for key, value in BOND_TYPE_TO_INDEX.items()}


def _undirected_bonds(edge_index, edge_type):
    bonds = {}
    for (src, dst), bond_type in zip(edge_index.t().tolist(), edge_type.tolist()):
        if src == dst:
            continue
        key = (min(src, dst), max(src, dst))
        previous = bonds.setdefault(key, int(bond_type))
        if previous != int(bond_type):
            raise ValueError(f"Conflicting bond types for atom pair {key}.")
    return bonds


def _graph_mapping(data_elements, data_bonds, template):
    num_atoms = len(data_elements)
    data_adjacency = [dict() for _ in range(num_atoms)]
    for (src, dst), bond_type in data_bonds.items():
        data_adjacency[src][dst] = bond_type
        data_adjacency[dst][src] = bond_type

    template_adjacency = [dict() for _ in range(template.GetNumAtoms())]
    for bond in template.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_TO_INDEX[bond.GetBondType()]
        template_adjacency[src][dst] = bond_type
        template_adjacency[dst][src] = bond_type

    def signature(element, adjacency):
        return int(element), len(adjacency), tuple(sorted(adjacency.values()))

    template_signatures = [
        signature(atom.GetAtomicNum(), template_adjacency[atom.GetIdx()])
        for atom in template.GetAtoms()
    ]
    candidates = []
    for atom_index, element in enumerate(data_elements):
        expected = signature(element, data_adjacency[atom_index])
        matches = [
            index for index, value in enumerate(template_signatures) if value == expected
        ]
        if not matches:
            raise ValueError(f"No SMILES atom matches data atom {atom_index}.")
        candidates.append(matches)

    mapping = {}
    used = set()

    def search(depth):
        if depth == num_atoms:
            return True
        unmapped = [index for index in range(num_atoms) if index not in mapping]
        data_atom = min(
            unmapped,
            key=lambda index: (
                -sum(neighbor in mapping for neighbor in data_adjacency[index]),
                len(candidates[index]),
                -len(data_adjacency[index]),
            ),
        )
        for template_atom in candidates[data_atom]:
            if template_atom in used:
                continue
            compatible = True
            for data_neighbor, data_bond_type in data_adjacency[data_atom].items():
                if data_neighbor not in mapping:
                    continue
                template_neighbor = mapping[data_neighbor]
                if template_adjacency[template_atom].get(template_neighbor) != data_bond_type:
                    compatible = False
                    break
            if not compatible:
                continue
            mapping[data_atom] = template_atom
            used.add(template_atom)
            if search(depth + 1):
                return True
            used.remove(template_atom)
            del mapping[data_atom]
        return False

    if not search(0):
        raise ValueError("Ligand tensors are not graph-isomorphic to ligand_smiles.")
    return mapping


def mol_from_targetdiff_data(data):
    """Recover an indexed RDKit molecule from a processed TargetDiff sample."""
    template = Chem.MolFromSmiles(data.ligand_smiles)
    if template is None:
        raise ValueError(f"Invalid ligand SMILES: {data.ligand_smiles}")
    elements = [int(value) for value in data.ligand_element.tolist()]
    if template.GetNumAtoms() != len(elements):
        raise ValueError("SMILES and ligand tensors contain different atom counts.")
    bonds = _undirected_bonds(data.ligand_bond_index, data.ligand_bond_type)
    mapping = _graph_mapping(elements, bonds, template)

    editable = Chem.RWMol()
    for data_index in range(len(elements)):
        editable.AddAtom(Chem.Atom(template.GetAtomWithIdx(mapping[data_index])))
    for (src, dst), bond_type in bonds.items():
        editable.AddBond(src, dst, INDEX_TO_BOND_TYPE[bond_type])
        bond = editable.GetBondBetweenAtoms(src, dst)
        template_bond = template.GetBondBetweenAtoms(mapping[src], mapping[dst])
        bond.SetIsAromatic(template_bond.GetIsAromatic())
        bond.SetBondDir(template_bond.GetBondDir())

    mol = editable.GetMol()
    conformer = Chem.Conformer(len(elements))
    for atom_index, position in enumerate(data.ligand_pos.tolist()):
        conformer.SetAtomPosition(atom_index, position)
    mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(mol)
    return mol


def _connected_components(num_atoms, bonds, cut_bond_indices):
    adjacency = [[] for _ in range(num_atoms)]
    cut_bond_indices = set(cut_bond_indices)
    for bond in bonds:
        if bond.GetIdx() in cut_bond_indices:
            continue
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[src].append(dst)
        adjacency[dst].append(src)

    components = []
    visited = set()
    for start in range(num_atoms):
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in adjacency[atom]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda atoms: atoms[0])


def _fragment_key(mol, atom_indices, junction_count):
    atom_set = set(atom_indices)
    editable = Chem.RWMol()
    old_to_new = {}
    for atom_index in atom_indices:
        old_to_new[atom_index] = editable.AddAtom(
            Chem.Atom(mol.GetAtomWithIdx(atom_index))
        )
    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if src in atom_set and dst in atom_set:
            editable.AddBond(
                old_to_new[src], old_to_new[dst], bond.GetBondType()
            )
    for atom_index in atom_indices:
        for _ in range(int(junction_count[atom_index])):
            dummy_index = editable.AddAtom(Chem.Atom(0))
            editable.AddBond(
                old_to_new[atom_index], dummy_index, Chem.BondType.SINGLE
            )
    fragment = editable.GetMol()
    Chem.SanitizeMol(fragment)
    return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)


def decompose_rbrics(mol):
    """Decompose an RDKit molecule without changing its atom indexing."""
    if mol is None:
        raise ValueError("Cannot decompose a null molecule.")
    mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(mol)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError("Cannot decompose an empty molecule.")

    cut_bond_indices = []
    for bond_info in rBRICS.FindrBRICSBonds(mol):
        src, dst = bond_info[0]
        bond = mol.GetBondBetweenAtoms(int(src), int(dst))
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            cut_bond_indices.append(bond.GetIdx())
    cut_bond_indices = sorted(set(cut_bond_indices))

    # rBRICS may propose a bond on a redundant path. FragFM removes candidates
    # whose endpoints remain in the same component after all proposed cuts.
    tentative_components = _connected_components(
        num_atoms, list(mol.GetBonds()), cut_bond_indices
    )
    tentative_assignment = np.empty(num_atoms, dtype=np.int64)
    for fragment_index, atoms in enumerate(tentative_components):
        tentative_assignment[atoms] = fragment_index
    cut_bond_indices = [
        bond_index
        for bond_index in cut_bond_indices
        if tentative_assignment[mol.GetBondWithIdx(bond_index).GetBeginAtomIdx()]
        != tentative_assignment[mol.GetBondWithIdx(bond_index).GetEndAtomIdx()]
    ]

    components = _connected_components(
        num_atoms, list(mol.GetBonds()), cut_bond_indices
    )
    atom_to_fragment = np.empty(num_atoms, dtype=np.int64)
    for fragment_index, atoms in enumerate(components):
        atom_to_fragment[atoms] = fragment_index

    junction_count = np.zeros(num_atoms, dtype=np.int64)
    junction_valence = np.zeros(num_atoms, dtype=np.float32)
    junction_atom_pairs = []
    junction_fragment_pairs = []
    junction_bond_types = []
    for bond_index in cut_bond_indices:
        bond = mol.GetBondWithIdx(bond_index)
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_fragment = int(atom_to_fragment[src])
        dst_fragment = int(atom_to_fragment[dst])
        if src_fragment == dst_fragment:
            raise ValueError("An rBRICS cut did not separate two fragments.")
        if src_fragment > dst_fragment:
            src, dst = dst, src
            src_fragment, dst_fragment = dst_fragment, src_fragment
        junction_atom_pairs.append((src, dst))
        junction_fragment_pairs.append((src_fragment, dst_fragment))
        bond_type = BOND_TYPE_TO_INDEX[bond.GetBondType()]
        junction_bond_types.append(bond_type)
        junction_count[src] += 1
        junction_count[dst] += 1
        junction_valence[src] += bond.GetBondTypeAsDouble()
        junction_valence[dst] += bond.GetBondTypeAsDouble()

    internal_bonds = []
    internal_bond_types = []
    cut_set = set(cut_bond_indices)
    for bond in mol.GetBonds():
        if bond.GetIdx() in cut_set:
            continue
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        internal_bonds.append((src, dst))
        internal_bond_types.append(BOND_TYPE_TO_INDEX[bond.GetBondType()])

    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    periodic_table = Chem.GetPeriodicTable()
    fragment_positions = []
    fragment_sizes = []
    fragment_attachment_counts = []
    fragment_keys = []
    flat_fragment_atoms = []
    fragment_ptr = [0]
    for atoms in components:
        masses = np.asarray(
            [periodic_table.GetAtomicWeight(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in atoms],
            dtype=np.float32,
        )
        fragment_positions.append(np.average(positions[atoms], axis=0, weights=masses))
        fragment_sizes.append(len(atoms))
        fragment_attachment_counts.append(int(junction_count[atoms].sum()))
        fragment_keys.append(_fragment_key(mol, atoms, junction_count))
        flat_fragment_atoms.extend(atoms)
        fragment_ptr.append(len(flat_fragment_atoms))

    record = {
        "num_atoms": num_atoms,
        "num_fragments": len(components),
        "atom_to_fragment": torch.from_numpy(atom_to_fragment),
        "fragment_atom_index": torch.tensor(flat_fragment_atoms, dtype=torch.long),
        "fragment_ptr": torch.tensor(fragment_ptr, dtype=torch.long),
        "fragment_pos": torch.tensor(np.asarray(fragment_positions), dtype=torch.float32),
        "fragment_size": torch.tensor(fragment_sizes, dtype=torch.long),
        "fragment_attachment_count": torch.tensor(
            fragment_attachment_counts, dtype=torch.long
        ),
        "fragment_keys": fragment_keys,
        "junction_count": torch.from_numpy(junction_count),
        "junction_valence": torch.from_numpy(junction_valence),
        "junction_atom_index": _edge_tensor(junction_atom_pairs),
        "junction_fragment_index": _edge_tensor(junction_fragment_pairs),
        "junction_bond_type": torch.tensor(junction_bond_types, dtype=torch.long),
        "internal_bond_index": _edge_tensor(internal_bonds),
        "internal_bond_type": torch.tensor(internal_bond_types, dtype=torch.long),
    }
    validate_fragment_record(record)
    return record


def _edge_tensor(edges):
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def validate_fragment_record(record):
    num_atoms = int(record["num_atoms"])
    num_fragments = int(record["num_fragments"])
    atom_to_fragment = record["atom_to_fragment"]
    if atom_to_fragment.shape != (num_atoms,):
        raise ValueError("atom_to_fragment has an invalid shape.")
    if atom_to_fragment.min() < 0 or atom_to_fragment.max() >= num_fragments:
        raise ValueError("atom_to_fragment contains an invalid fragment index.")
    junction_atoms = record["junction_atom_index"]
    junction_fragments = record["junction_fragment_index"]
    if junction_atoms.shape != junction_fragments.shape:
        raise ValueError("Junction atom and fragment edges must align.")
    if junction_atoms.size(1):
        mapped = atom_to_fragment[junction_atoms]
        if not torch.equal(mapped, junction_fragments):
            raise ValueError("Junction atom endpoints do not match fragment endpoints.")
        if torch.any(junction_fragments[0] == junction_fragments[1]):
            raise ValueError("A junction edge cannot be internal to one fragment.")
    expected_count = torch.bincount(
        junction_atoms.flatten(), minlength=num_atoms
    ) if junction_atoms.numel() else torch.zeros(num_atoms, dtype=torch.long)
    if not torch.equal(expected_count, record["junction_count"]):
        raise ValueError("Junction capacities do not match junction edges.")


def reconstruct_from_fragments(mol, record):
    """Rebuild the molecular graph using only internal and junction bonds."""
    source = Chem.RemoveHs(Chem.Mol(mol))
    editable = Chem.RWMol()
    for atom in source.GetAtoms():
        editable.AddAtom(Chem.Atom(atom))
    for edge, bond_type in zip(
        record["internal_bond_index"].t().tolist(),
        record["internal_bond_type"].tolist(),
    ):
        editable.AddBond(edge[0], edge[1], INDEX_TO_BOND_TYPE[bond_type])
    for edge, bond_type in zip(
        record["junction_atom_index"].t().tolist(),
        record["junction_bond_type"].tolist(),
    ):
        editable.AddBond(edge[0], edge[1], INDEX_TO_BOND_TYPE[bond_type])
    rebuilt = editable.GetMol()
    Chem.SanitizeMol(rebuilt)
    return rebuilt


def summarize_fragment_records(records):
    fragment_counts = [int(record["num_fragments"]) for record in records]
    junction_counts = [record["junction_atom_index"].size(1) for record in records]
    vocabulary = defaultdict(int)
    for record in records:
        for key in record["fragment_keys"]:
            vocabulary[key] += 1
    return {
        "num_molecules": len(records),
        "mean_fragments": float(np.mean(fragment_counts)) if records else 0.0,
        "max_fragments": max(fragment_counts, default=0),
        "single_fragment_rate": (
            float(np.mean(np.asarray(fragment_counts) == 1)) if records else 0.0
        ),
        "mean_junctions": float(np.mean(junction_counts)) if records else 0.0,
        "vocabulary_size": len(vocabulary),
        "fragment_frequency": dict(
            sorted(vocabulary.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
