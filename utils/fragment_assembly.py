import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _ideal_angle(atom):
    hybridization = atom.GetHybridization()
    if hybridization == Chem.HybridizationType.SP:
        return 180.0
    if hybridization == Chem.HybridizationType.SP2:
        return 120.0
    return 109.5


def _ideal_single_bond_length(molecule, src, dst):
    table = Chem.GetPeriodicTable()
    length = (
        table.GetRcovalent(molecule.GetAtomWithIdx(src).GetAtomicNum())
        + table.GetRcovalent(molecule.GetAtomWithIdx(dst).GetAtomicNum())
    )
    return float(np.clip(length, 1.20, 1.90))


def _geometry_force_field(molecule):
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    if properties is not None:
        return AllChem.MMFFGetMoleculeForceField(molecule, properties), "MMFF"
    return AllChem.UFFGetMoleculeForceField(molecule), "UFF"


def _random_rotation(rng):
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rigid_clash_search(aligned, protein_positions, minimum, candidates=1024):
    rng = np.random.default_rng(17)
    center = aligned.mean(axis=0)
    centered = aligned - center
    best_score = np.inf
    best = aligned
    for index in range(candidates):
        rotation = np.eye(3) if index == 0 else _random_rotation(rng)
        translation = np.zeros(3) if index == 0 else rng.uniform(-5.0, 5.0, 3)
        candidate = centered @ rotation + center + translation
        distance = np.linalg.norm(
            candidate[:, None, :] - protein_positions[None, :, :], axis=-1
        ).clip(min=1e-6)
        overlap = np.maximum(minimum - distance, 0.0)
        # A single residual clash is more important than a shorter translation.
        score = float(
            100.0 * (overlap > 0).sum()
            + (overlap * overlap).sum()
            + 0.001 * (translation * translation).sum()
        )
        if score < best_score:
            best_score = score
            best = candidate
    return best


def optimize_junction_geometry(
    molecule,
    junctions,
    protein_positions=None,
    protein_atomic_numbers=None,
    max_iters=500,
    protein_repulsion=True,
    random_seed=0,
):
    """Rebuild valid covalent geometry, align it to the pose, then remove clashes."""
    original_positions = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    rebuilt = Chem.Mol(molecule)
    rebuilt.RemoveAllConformers()
    rebuilt_h = Chem.AddHs(rebuilt)
    embedded = False
    for attempt in range(3):
        rebuilt_h.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed + 17 + 7919 * attempt)
        params.maxIterations = 100
        params.useRandomCoords = attempt >= 2
        if AllChem.EmbedMolecule(rebuilt_h, params) == 0:
            embedded = True
            break
    if not embedded:
        raise ValueError("Could not rebuild the assembled molecular conformer.")
    rebuilt = Chem.RemoveHs(rebuilt_h)
    force_field, force_field_type = _geometry_force_field(rebuilt)
    if force_field is None:
        raise ValueError("No molecular force field is available for the assembled molecule.")

    for src, dst in junctions:
        target = _ideal_single_bond_length(rebuilt, src, dst)
        force_field.AddDistanceConstraint(src, dst, target - 0.02, target + 0.02, 1000.0)
        for center, opposite in ((src, dst), (dst, src)):
            atom = rebuilt.GetAtomWithIdx(center)
            target_angle = _ideal_angle(atom)
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index == opposite:
                    continue
                if force_field_type == "MMFF":
                    force_field.MMFFAddAngleConstraint(
                        neighbor_index, center, opposite, False,
                        target_angle - 8.0, target_angle + 8.0, 100.0,
                    )
                else:
                    force_field.UFFAddAngleConstraint(
                        neighbor_index, center, opposite, False,
                        target_angle - 8.0, target_angle + 8.0, 100.0,
                    )

    force_field.Initialize()
    force_field.Minimize(maxIts=int(max_iters), forceTol=1e-4, energyTol=1e-6)

    # Preserve the generated center and best global orientation without
    # preserving its malformed inter-fragment distances.
    rebuilt_positions = np.asarray(
        rebuilt.GetConformer().GetPositions(), dtype=np.float64
    )
    rebuilt_center = rebuilt_positions.mean(axis=0)
    original_center = original_positions.mean(axis=0)
    moving = rebuilt_positions - rebuilt_center
    target = original_positions - original_center
    u, _, vt = np.linalg.svd(moving.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = moving @ rotation + original_center

    if protein_repulsion and protein_positions is not None:
        protein_positions = np.asarray(protein_positions, dtype=np.float64)
        if protein_atomic_numbers is None:
            protein_atomic_numbers = np.full(len(protein_positions), 6, dtype=np.int64)
        protein_atomic_numbers = np.asarray(protein_atomic_numbers, dtype=np.int64)
        table = Chem.GetPeriodicTable()
        ligand_radii = np.asarray(
            [table.GetRvdw(atom.GetAtomicNum()) for atom in rebuilt.GetAtoms()]
        )
        protein_radii = np.asarray(
            [table.GetRvdw(int(np.clip(value, 1, 118))) for value in protein_atomic_numbers]
        )
        minimum = 0.90 * (ligand_radii[:, None] + protein_radii[None, :])
        aligned = _rigid_clash_search(aligned, protein_positions, minimum)
        translation = np.zeros(3, dtype=np.float64)
        for _ in range(200):
            vector = aligned[:, None, :] + translation - protein_positions[None, :, :]
            distance = np.linalg.norm(vector, axis=-1).clip(min=1e-6)
            overlap = np.maximum(minimum - distance, 0.0)
            active = overlap > 0
            if not active.any():
                break
            gradient = (
                -2.0 * overlap[..., None] * vector / distance[..., None]
            )[active].mean(axis=0)
            gradient += 0.01 * translation
            step = np.clip(-0.10 * gradient, -0.20, 0.20)
            translation += step
        aligned = aligned + translation

    conformer = Chem.Conformer(rebuilt.GetNumAtoms())
    for atom_index, position in enumerate(aligned):
        conformer.SetAtomPosition(atom_index, position)
    rebuilt.RemoveAllConformers()
    rebuilt.AddConformer(conformer, assignId=True)
    return rebuilt


def _instantiate_fragment(token, center, random_seed=0):
    template = Chem.MolFromSmiles(token)
    if template is None:
        raise ValueError(f"Invalid fragment token: {token}")
    real_atoms = [atom.GetIdx() for atom in template.GetAtoms() if atom.GetAtomicNum() != 0]
    old_to_new = {}
    editable = Chem.RWMol()
    for old_index in real_atoms:
        old_to_new[old_index] = editable.AddAtom(Chem.Atom(template.GetAtomWithIdx(old_index)))
    for bond in template.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if src in old_to_new and dst in old_to_new:
            editable.AddBond(old_to_new[src], old_to_new[dst], bond.GetBondType())
            editable.GetBondBetweenAtoms(
                old_to_new[src], old_to_new[dst]
            ).SetIsAromatic(bond.GetIsAromatic())
    attachment_slots = []
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetIdx() not in old_to_new:
            raise ValueError(f"Malformed attachment dummy in token: {token}")
        attachment_slots.append(old_to_new[neighbors[0].GetIdx()])
    slot_counts = {slot: attachment_slots.count(slot) for slot in set(attachment_slots)}
    base_hydrogens = {
        old_to_new[old_index]: template.GetAtomWithIdx(old_index).GetTotalNumHs()
        for old_index in real_atoms
    }
    for slot, count in slot_counts.items():
        attachment_atom = editable.GetAtomWithIdx(slot)
        attachment_atom.SetNoImplicit(True)
        attachment_atom.SetNumExplicitHs(base_hydrogens[slot] + count)

    fragment = editable.GetMol()
    Chem.SanitizeMol(fragment)
    if fragment.GetNumAtoms() == 1:
        coordinates = np.zeros((1, 3), dtype=np.float64)
    else:
        fragment_h = Chem.AddHs(fragment)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        if AllChem.EmbedMolecule(fragment_h, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(fragment_h, params) != 0:
                raise ValueError(f"Could not embed fragment token: {token}")
        try:
            AllChem.UFFOptimizeMolecule(fragment_h, maxIters=100)
        except ValueError:
            pass
        fragment = Chem.RemoveHs(fragment_h)
        for slot, count in slot_counts.items():
            fragment.GetAtomWithIdx(slot).SetNoImplicit(True)
            fragment.GetAtomWithIdx(slot).SetNumExplicitHs(
                base_hydrogens[slot] + count
            )
        coordinates = np.asarray(fragment.GetConformer().GetPositions(), dtype=np.float64)
    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    coordinates = coordinates + np.asarray(center, dtype=np.float64)
    return fragment, coordinates, attachment_slots


def assemble_fragment_graph(
    tokens,
    fragment_positions,
    edges,
    random_seed=0,
    optimize_geometry=False,
    protein_positions=None,
    protein_atomic_numbers=None,
    geometry_max_iters=500,
    protein_repulsion=True,
):
    """Instantiate fragment templates and connect each dummy slot at most once."""
    if len(tokens) != len(fragment_positions):
        raise ValueError("Fragment token and position counts differ.")
    fragments = [
        _instantiate_fragment(token, center, random_seed + index)
        for index, (token, center) in enumerate(zip(tokens, fragment_positions))
    ]
    editable = Chem.RWMol()
    coordinates = []
    offsets = []
    global_slots = []
    for fragment, fragment_coordinates, local_slots in fragments:
        offset = editable.GetNumAtoms()
        offsets.append(offset)
        for atom in fragment.GetAtoms():
            editable.AddAtom(Chem.Atom(atom))
        for bond in fragment.GetBonds():
            editable.AddBond(
                offset + bond.GetBeginAtomIdx(),
                offset + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
            editable.GetBondBetweenAtoms(
                offset + bond.GetBeginAtomIdx(), offset + bond.GetEndAtomIdx()
            ).SetIsAromatic(bond.GetIsAromatic())
        coordinates.extend(fragment_coordinates.tolist())
        global_slots.append([offset + slot for slot in local_slots])

    used_slots = [set() for _ in tokens]
    junctions = []
    for edge in sorted(edges, key=lambda value: float(value[2]), reverse=True):
        src_fragment, dst_fragment = int(edge[0]), int(edge[1])
        candidates = []
        for src_slot_index, src_atom in enumerate(global_slots[src_fragment]):
            if src_slot_index in used_slots[src_fragment]:
                continue
            for dst_slot_index, dst_atom in enumerate(global_slots[dst_fragment]):
                if dst_slot_index in used_slots[dst_fragment]:
                    continue
                distance = np.linalg.norm(
                    np.asarray(coordinates[src_atom]) - np.asarray(coordinates[dst_atom])
                )
                candidates.append(
                    (distance, src_slot_index, dst_slot_index, src_atom, dst_atom)
                )
        if not candidates:
            raise ValueError(
                f"No free junction slots for fragment edge {src_fragment}-{dst_fragment}."
            )
        _, src_slot_index, dst_slot_index, src_atom, dst_atom = min(candidates)
        for atom_index in (src_atom, dst_atom):
            atom = editable.GetAtomWithIdx(atom_index)
            if atom.GetNumExplicitHs() <= 0:
                raise ValueError("A junction slot has no hydrogen cap to replace.")
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
        editable.AddBond(src_atom, dst_atom, Chem.BondType.SINGLE)
        used_slots[src_fragment].add(src_slot_index)
        used_slots[dst_fragment].add(dst_slot_index)
        junctions.append((src_atom, dst_atom))

    molecule = editable.GetMol()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index, position in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, position)
    molecule.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Assembled fragment graph is disconnected.")
    if optimize_geometry:
        molecule = optimize_junction_geometry(
            molecule,
            junctions,
            protein_positions=protein_positions,
            protein_atomic_numbers=protein_atomic_numbers,
            max_iters=geometry_max_iters,
            protein_repulsion=protein_repulsion,
            random_seed=random_seed,
        )
    return molecule, junctions
