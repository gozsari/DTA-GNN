from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MoleculeGraph2D:
    molecule_chembl_id: str
    atom_type: np.ndarray  # (n_atoms,) int64
    atom_feat: np.ndarray  # (n_atoms, 6) float32
    edge_index: np.ndarray  # (2, n_edges) int64
    edge_attr: np.ndarray  # (n_edges, 6) float32


def smiles_to_graph_2d(
    *, molecule_chembl_id: str, smiles: str
) -> MoleculeGraph2D | None:
    """Convert a SMILES string into a simple 2D molecular graph.

    Node features are fixed-size (6) numeric features.
    Edge features are fixed-size (6) numeric features.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_type = np.zeros((n_atoms,), dtype=np.int64)
    atom_feat = np.zeros((n_atoms, 6), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = int(atom.GetAtomicNum())
        atom_type[i] = np.int64(atomic_num)

        # 6 node features (simple, stable)
        atom_feat[i, 0] = float(atomic_num)
        atom_feat[i, 1] = float(atom.GetTotalDegree())
        atom_feat[i, 2] = float(atom.GetFormalCharge())
        atom_feat[i, 3] = float(atom.GetTotalNumHs(includeNeighbors=True))
        atom_feat[i, 4] = 1.0 if atom.GetIsAromatic() else 0.0
        atom_feat[i, 5] = float(atom.GetMass())

    # Build directed edges
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr_rows: list[list[float]] = []

    def _bond_features(bond: "Chem.Bond") -> list[float]:
        bt = bond.GetBondType()
        is_single = 1.0 if bt == Chem.BondType.SINGLE else 0.0
        is_double = 1.0 if bt == Chem.BondType.DOUBLE else 0.0
        is_triple = 1.0 if bt == Chem.BondType.TRIPLE else 0.0
        is_aromatic = 1.0 if bond.GetIsAromatic() else 0.0
        is_conj = 1.0 if bond.GetIsConjugated() else 0.0
        is_ring = 1.0 if bond.IsInRing() else 0.0
        return [is_single, is_double, is_triple, is_aromatic, is_conj, is_ring]

    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        bf = _bond_features(bond)

        edge_src.append(a)
        edge_dst.append(b)
        edge_attr_rows.append(bf)

        edge_src.append(b)
        edge_dst.append(a)
        edge_attr_rows.append(bf)

    if edge_src:
        edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.asarray(edge_attr_rows, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 6), dtype=np.float32)

    return MoleculeGraph2D(
        molecule_chembl_id=str(molecule_chembl_id),
        atom_type=atom_type,
        atom_feat=atom_feat,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def build_graphs_2d(
    *,
    molecules: Iterable[tuple[str, str]],
    drop_failures: bool = True,
) -> list[MoleculeGraph2D]:
    graphs: list[MoleculeGraph2D] = []
    for mid, smi in molecules:
        g = smiles_to_graph_2d(molecule_chembl_id=str(mid), smiles=str(smi))
        if g is None:
            if drop_failures:
                continue
            raise ValueError(f"Failed to parse SMILES for molecule {mid!r}: {smi!r}")
        graphs.append(g)
    return graphs


__all__ = [
    "MoleculeGraph2D",
    "smiles_to_graph_2d",
    "build_graphs_2d",
]
