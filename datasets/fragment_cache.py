import lmdb
import pickle
import json
import os
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from torch.utils.data import Dataset


_READONLY_ENVIRONMENTS = {}


class FragmentCacheDataset(Dataset):
    """Attach precomputed rBRICS tensors to a protein-ligand dataset."""

    TENSOR_FIELDS = {
        "atom_to_fragment": "ligand_atom_to_fragment",
        "fragment_atom_index": "ligand_fragment_atom_index",
        "fragment_pos": "ligand_fragment_pos",
        "fragment_size": "ligand_fragment_size",
        "fragment_attachment_count": "ligand_fragment_attachment_count",
        "junction_count": "ligand_junction_count",
        "junction_valence": "ligand_junction_valence",
        "junction_atom_index": "ligand_junction_atom_index",
        "junction_fragment_index": "ligand_fragment_edge_index",
        "junction_bond_type": "ligand_junction_bond_type",
        "internal_bond_index": "ligand_internal_bond_index",
        "internal_bond_type": "ligand_internal_bond_type",
    }

    def __init__(self, dataset, cache_path, vocab_path=None):
        self.dataset = dataset
        self.cache_path = cache_path
        self.db = None
        self.fragment_token_to_id = None
        if vocab_path is not None:
            with open(vocab_path, "r", encoding="utf-8") as handle:
                vocabulary = json.load(handle)
            self.fragment_token_to_id = vocabulary["token_to_id"]
        discovery_db = lmdb.open(
            self.cache_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with discovery_db.begin() as transaction:
            cached_ids = {
                int(key[3:])
                for key in transaction.cursor().iternext(values=False)
                if key.startswith(b"id:")
            }
        discovery_db.close()
        if hasattr(dataset, "indices"):
            self.available_indices = [
                local_index
                for local_index, dataset_index in enumerate(dataset.indices)
                if int(dataset_index) in cached_ids
            ]
        else:
            self.available_indices = list(range(len(dataset)))

    def _connect(self):
        if self.db is None:
            key = os.path.abspath(self.cache_path)
            if key not in _READONLY_ENVIRONMENTS:
                _READONLY_ENVIRONMENTS[key] = lmdb.open(
                    key,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            self.db = _READONLY_ENVIRONMENTS[key]

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, index):
        data = self.dataset[self.available_indices[index]]
        self._connect()
        key = f"id:{int(data.id)}".encode("ascii")
        value = self.db.begin().get(key)
        if value is None:
            raise KeyError(f"No rBRICS cache entry for dataset index {int(data.id)}")
        record = pickle.loads(value)
        if int(record["num_atoms"]) != data.ligand_element.size(0):
            raise ValueError("Cached fragment graph does not match ligand atom count.")
        for source, destination in self.TENSOR_FIELDS.items():
            setattr(data, destination, record[source])
        if self.fragment_token_to_id is not None:
            data.ligand_fragment_token = torch.tensor(
                [self.fragment_token_to_id.get(key, 0) for key in record["fragment_keys"]],
                dtype=torch.long,
            )
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None:
            raise ValueError("Cached ligand SMILES is invalid.")
        data.ligand_heavy_atom_count = torch.tensor(
            [molecule.GetNumHeavyAtoms()], dtype=torch.long
        )
        data.ligand_qed_target = torch.tensor(
            [QED.qed(molecule)], dtype=torch.float32
        )
        data.ligand_logp_target = torch.tensor(
            [Crippen.MolLogP(molecule)], dtype=torch.float32
        )
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        state["db"] = None
        return state
