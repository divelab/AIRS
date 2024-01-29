import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from .mat_utils import lattice_params_to_matrix


CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def build_crystal_graph(crystal, lattice_scale=True, graph_method='crystalnn', cutoff=2.0):
    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'cutoff':
        all_nbrs = crystal.get_all_neighbors(cutoff)
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    edge_indices_inv, to_jimages_inv = [], []
    
    if graph_method == 'crystalnn':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices_inv.append([i, j])
            to_jimages_inv.append(tuple(-tj for tj in to_jimage))
    elif graph_method == 'cutoff':
        for i in range(len(all_nbrs)):
            for nbr in all_nbrs[i]:
                if nbr.index < i:
                    edge_indices.append([nbr.index, i])
                    to_jimages.append(nbr.image)
                    edge_indices_inv.append([i, nbr.index])
                    to_jimages_inv.append(tuple(-tj for tj in nbr.image))
    
    edge_indices += edge_indices_inv
    to_jimages += to_jimages_inv

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    scaled_lengths = lengths / float(num_atoms)**(1/3) if lattice_scale else lengths

    return frac_coords, atom_types, scaled_lengths, lengths, angles, edge_indices, to_jimages, num_atoms


def build_crystal(crystal_str, niggli=True, primitive=False):
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    
    frac_coords = crystal.frac_coords
    gt_frac_coords = torch.tensor(frac_coords) % 1.0
    candidate_frac_coords_list = []

    for i in range(3):
        candidate_frac_coords = []
        max_dists = []
        given_frac_coord = gt_frac_coords[:, i].clone()
        for _ in range(len(given_frac_coord)):
            min_id = given_frac_coord.argmin()
            given_frac_coord[min_id] += 1.0
            candidate_frac_coords.append(given_frac_coord.clone())
            max_dists.append((given_frac_coord.max() - given_frac_coord.min()).item())
        
        candidate_id = torch.tensor(max_dists).argmin()
        candidate_frac_coords_list.append(candidate_frac_coords[candidate_id])
    
    gt_frac_coords = torch.stack(candidate_frac_coords_list, dim=1)

    min_frac_coords, _ = gt_frac_coords.min(dim=0, keepdim=True)
    max_frac_coords, _ = gt_frac_coords.max(dim=0, keepdim=True)
    offset_frac_coords = (min_frac_coords + max_frac_coords) / 2.0
    new_frac_coords = gt_frac_coords - offset_frac_coords

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=new_frac_coords,
        coords_are_cartesian=False,
    )

    return canonical_crystal


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""
    EPSILON = 1e-5
    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + self.EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none