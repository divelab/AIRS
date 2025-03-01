import warnings
import tempfile

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import openbabel

import utils
from constants import bonds1, bonds2, bonds3, margin1, margin2, margin3, \
    bond_dict


def get_bond_order(atom1, atom2, distance):
    # distance = 100 * distance  # We change the metric

    if atom1 in bonds3 and atom2 in bonds3[atom1] and distance < bonds3[atom1][atom2] + margin3:
        return 3  # Triple

    if atom1 in bonds2 and atom2 in bonds2[atom1] and distance < bonds2[atom1][atom2] + margin2:
        return 2  # Double

    if atom1 in bonds1 and atom2 in bonds1[atom1] and distance < bonds1[atom1][atom2] + margin1:
        return 1  # Single

    return 0      # No bond


def get_bond_order_batch(atoms1, atoms2, distances, dataset_info):
    if isinstance(atoms1, np.ndarray):
        atoms1 = torch.from_numpy(atoms1)
    if isinstance(atoms2, np.ndarray):
        atoms2 = torch.from_numpy(atoms2)
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    # distances = 100 * distances  # We change the metric

    bonds1 = torch.tensor(dataset_info['bonds1'], device=atoms1.device)
    bonds2 = torch.tensor(dataset_info['bonds2'], device=atoms1.device)
    bonds3 = torch.tensor(dataset_info['bonds3'], device=atoms1.device)

    bond_types = torch.zeros_like(atoms1)  # 0: No bond

    # Single
    bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1

    # Double (note that already assigned single bonds will be overwritten)
    bond_types[distances < bonds2[atoms1, atoms2] + margin2] = 2

    # Triple
    bond_types[distances < bonds3[atoms1, atoms2] + margin3] = 3

    return bond_types


def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        utils.write_xyz_file(positions, atom_types, tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)

        obConversion.WriteFile(ob_mol, tmp_file)

        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    bond.GetBondType())

    return mol


def make_mol_edm(positions, atom_types, dataset_info, add_coords):
    """
    Equivalent to EDM's way of building RDKit molecules
    """
    n = len(positions)

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    pos = positions.unsqueeze(0)  # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)  # remove batch dim & flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
    E = torch.tril(E_full, diagonal=-1)  # Warning: the graph should be DIRECTED
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)

    return mol


def build_molecule(positions, atom_types, dataset_info, add_coords=False,
                   use_openbabel=True):
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types,
                                 dataset_info["atom_decoder"])
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol


def process_molecule(rdmol, add_hydrogens=False, sanitize=False, relax_iter=0,
                     largest_frag=False):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """
    # import pdb; pdb.set_trace()
    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            # mol = Chem.AddHs(mol)
            uff_relax(mol, relax_iter)
            # mol = Chem.RemoveHs(mol)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def filter_rd_mol(rdmol):
    """
    Filter out RDMols if they have a 3-3 ring intersection
    adapted from:
    https://github.com/luost26/3D-Generative-SBDD/blob/main/utils/chem.py
    """
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:
            continue
        for j, ring_b in enumerate(rings):
            if i <= j:
                continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0): 
                return False

    return True
