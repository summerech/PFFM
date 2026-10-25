import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def kabsch_align(position, reference):
    source_center = position.mean(dim=0, keepdim=True)
    target_center = reference.mean(dim=0, keepdim=True)
    source = position - source_center
    target = reference - target_center
    left, _, right_t = torch.linalg.svd(source.T @ target)
    rotation = left @ right_t
    if torch.linalg.det(rotation) < 0:
        left = left.clone()
        left[:, -1] *= -1
        rotation = left @ right_t
    return source @ rotation + target_center


def build_geometry_reference(molecule, raw_position, max_iters=300):
    """Create a force-field bond/angle teacher aligned to the generated pose."""
    optimized = Chem.AddHs(Chem.Mol(molecule), addCoords=True)
    status, force_field = -1, "none"
    try:
        if AllChem.MMFFHasAllMoleculeParams(optimized):
            status = AllChem.MMFFOptimizeMolecule(optimized, maxIters=max_iters)
            force_field = "mmff"
        else:
            status = AllChem.UFFOptimizeMolecule(optimized, maxIters=max_iters)
            force_field = "uff"
    except Exception:
        optimized = Chem.AddHs(Chem.Mol(molecule), addCoords=True)
        try:
            status = AllChem.UFFOptimizeMolecule(optimized, maxIters=max_iters)
            force_field = "uff"
        except Exception:
            return raw_position.clone(), "failed", -1
    optimized = Chem.RemoveHs(optimized)
    coordinates = torch.tensor(
        optimized.GetConformer().GetPositions(), dtype=raw_position.dtype
    )
    return kabsch_align(coordinates, raw_position), force_field, int(status)
