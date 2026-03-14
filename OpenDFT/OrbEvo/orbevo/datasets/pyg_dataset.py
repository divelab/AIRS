import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data, Dataset


def ar_transform(data, t, time_cond=1, time_future=1, time_gap=1, rms_0=1.0, rms_t=1.0):  # ar
    """
    Get input and target for auto-regressive prediction
    
    Args:
        t: target start time
    """
    _, num_states, num_atoms, _, _ = data.coef_0.shape

    # 0 is initial state
    # target is computed based on coefs[t] and coefs[0], t = 1, ..., T

    delta_coef_cond = torch.zeros((time_cond, num_states, num_atoms, 9, 2), dtype=torch.complex64)
    state_phase_cond = torch.ones((time_cond, num_states), dtype=torch.complex64)

    available_cond_steps = min(time_cond, t)
    if available_cond_steps > 0:
        delta_coef_cond[time_cond - available_cond_steps :] = data.delta_coef_t[t - available_cond_steps : t]
        state_phase_cond[time_cond - available_cond_steps :] = data.state_phase_t[t - available_cond_steps : t]

    # target and cond have 2*time_future for push forward
    delta_coef_target = torch.zeros((time_future * 2, num_states, num_atoms, 9, 2), dtype=torch.complex64)
    available_time_future = min(data.delta_coef_t.shape[0] - t, time_future * 2)
    delta_coef_target[: available_time_future] = data.delta_coef_t[t : t + available_time_future]

    state_phase_target = torch.ones((time_future * 2, num_states), dtype=torch.complex64)
    state_phase_target[: available_time_future] = data.state_phase_t[t : t + available_time_future]

    # delta_coef_target = data.delta_coef_t[t : t + time_future]
    # state_phase_target = data.state_phase_t[t : t + time_future]

    # efield (not downsampled)
    efield = torch.zeros(((time_cond + time_future * 2) * 10), dtype=torch.float32) # 2 future steps for push forward
    efield[(time_cond - available_cond_steps) * 10 : (time_cond + available_time_future) * 10] = data['efield'][(t - available_cond_steps) * 10 : (t + available_time_future) * 10, 1]
    efield = efield * 50  # scale max to 1

    new_data = Data(
        atom_type=data.atom_type, 
        atom_pos=data.atom_pos,
        coef_0=data.coef_0 / rms_0, 
        delta_coef_cond=delta_coef_cond / rms_t, 
        state_phase_cond=state_phase_cond,
        delta_coef_target=delta_coef_target / rms_t, 
        state_phase_target=state_phase_target,
        occ=data.occ,
        coef_mask=data.coef_mask,
        efield=efield, 
        t=t
        )
    return new_data


class TDDFTv2_pyg(Dataset):

    def __init__(self, root, rms_0, rms_t,
                 time_start=0, time_cond=1, time_future=1, time_gap=1, T=100, 
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.rms_0 = rms_0
        self.rms_t = rms_t
        self.time_start = time_start
        self.time_cond = time_cond
        self.time_future = time_future
        self.time_gap = time_gap
        self.num_targets = T + 1 - self.time_future * self.time_gap
        assert self.num_targets > 0

    @property
    def abacus_to_tfn(self):
        # ssppdd -> spdspd
        abacus_to_tfn = []
        inds_s = torch.tensor([0])
        inds_p = torch.tensor([2, 0, 1])
        inds_d = torch.tensor([4, 2, 0, 1, 3])
        abacus_to_tfn.append(inds_s)
        abacus_to_tfn.append(inds_p + 2)
        abacus_to_tfn.append(inds_d + 2 + 6)
        abacus_to_tfn.append(inds_s + 1)
        abacus_to_tfn.append(inds_p + 2 + 3)
        abacus_to_tfn.append(inds_d + 2 + 6 + 5)
        return torch.cat(abacus_to_tfn)

    @property
    def raw_file_names(self):
        files = sorted([fn for fn in os.listdir(self.raw_dir) if fn.endswith('.pt')])
        return files

    @property
    def processed_file_names(self):
        return self.raw_file_names

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...

    def process(self):
        for raw_path in tqdm(self.raw_paths):
            # Read data from `raw_path`.
            data_torch = torch.load(raw_path)
            # return self.transform(data, t=t, time_cond=self.time_cond, time_future=self.time_future, time_gap=self.time_gap)

            # data_pyg = Data(**data_torch)

            num_atoms = data_torch['atom_type'].shape[0]
            num_orbitals = {1: 5, # ssp
                            6: 13, # ssppd
                            7: 13,
                            8: 13,
                            9: 13
                            }
            orbital_sizes = [num_orbitals[at.item()] for at in data_torch['atom_type']]

            coef_data = {}
            for key in ['coef_0', 'delta_coef_t']:
                coef = data_torch[key]
                coef_split = coef.split(orbital_sizes, dim=-1)
                coef_pad = torch.zeros(coef.shape[0], coef.shape[1], num_atoms, 18, dtype=torch.complex64)
                for i in range(num_atoms):
                    coef_pad[:, :, i, :orbital_sizes[i]] = coef_split[i]
                # T, states, N, 9, 2
                coef_pad = coef_pad[:, :, :, self.abacus_to_tfn].contiguous().reshape(coef.shape[0], coef.shape[1], num_atoms, 2, 9).transpose(-1, -2)
                coef_data[key] = coef_pad

            coef_mask = torch.zeros(num_atoms, 18, dtype=torch.bool)
            for i in range(num_atoms):
                coef_mask[i, :orbital_sizes[i]] = 1
            # N, 9, 2
            coef_mask = coef_mask[:, self.abacus_to_tfn].contiguous().reshape(num_atoms, 2, 9).transpose(-1, -2)

            data_pyg = Data(atom_type=data_torch['atom_type'],
                            atom_pos=data_torch['atom_pos'],
                            coef_0=coef_data['coef_0'],
                            delta_coef_t=coef_data['delta_coef_t'],
                            state_phase_t=data_torch['state_phase_t'],
                            occ=data_torch['occs'][0], # assume the occupancy number does not change with time
                            coef_mask=coef_mask,
                            efield=data_torch['efield'])

            if self.pre_filter is not None and not self.pre_filter(data_pyg):
                continue

            if self.pre_transform is not None:
                data_pyg = self.pre_transform(data_pyg)

            torch.save(data_pyg, osp.join(self.processed_dir, osp.basename(raw_path)))

    def len(self):
        return len(self.processed_file_names) * self.num_targets

    def get(self, idx):
        traj_idx = idx // self.num_targets
        t = idx % self.num_targets + self.time_start
        data = torch.load(osp.join(self.processed_dir, f'{traj_idx:05d}.pt'))

        # legacy naming convention
        data.state_phase_t = data.eband_phase_t

        data = ar_transform(data, t=t, time_cond=self.time_cond, time_future=self.time_future, time_gap=self.time_gap, rms_0=self.rms_0, rms_t=self.rms_t)
        # data.last_step = t == (self.num_targets - 1)
        return data
    

def collate_fn(original_batch):
    """
    Build a two-level minibatch for TDDFT trajectories.

    Input:
        original_batch: List[Data], where each Data is one molecule sample
        with multiple occupied electronic states.

    Output:
        dict with:
        - molecule_data: molecule-level tensors (one entry per molecule in the
          minibatch; atoms are repeated per molecule).
        - state_data: state-level tensors (one entry per occupied state; atoms
          are repeated per state).

    Index mapping conventions:
        - molecule_data.batch:
            atom index (molecule space) -> molecule id.
        - state_data.batch:
            atom-in-state index (state space) -> state id.
        - state_data.state_ind_batch:
            state id -> molecule id.
        - state_data.mol_batch:
            atom-in-state index -> molecule id.
        - state_data.state_atom_batch:
            atom-in-state index -> atom index in molecule_data.

    In short, this collate function keeps both granularities:
    molecule-level context (`molecule_data`) and expanded per-state graphs
    (`state_data`) with explicit cross-index tensors between the two.
    """
    molecule_atom_type = []
    molecule_atom_pos = []
    molecule_batch = []
    molecule_num_atoms = []
    molecule_occ = []
    molecule_occ_batch = []
    efield = []
    molecule_coef_mask = []
    ts = []

    state_atom_type = []
    state_atom_pos = []
    coef_0 = []
    delta_coef_cond = []
    delta_coef_target = []
    state_phase_cond = []
    state_phase_target = []
    state_batch = []
    atom_in_state_batch = []
    state_ind_batch = []  # index from state to molecule in the minibatch
    state_to_molecule_batch = []
    coef_mask = []
    state_num_atoms = []
    molecule_num_states = []
    state_batch_counter = 0
    state_atom_batch_counter = 0

    for data in original_batch:
        # Molecule-level data (one entry per molecule).
        num_atoms = data.atom_type.shape[0]
        molecule_atom_type.append(data.atom_type)
        molecule_atom_pos.append(data.atom_pos)
        molecule_occ.append(data.occ[data.occ > 0])  # occupied states
        molecule_occ_batch.append(torch.ones(data.occ[data.occ > 0].shape[0], dtype=torch.long) * len(molecule_occ_batch))
        molecule_batch.append(torch.ones(num_atoms, dtype=torch.long) * len(molecule_batch))
        molecule_num_atoms.append(num_atoms)
        efield.append(data.efield)
        molecule_coef_mask.append(data.coef_mask)
        ts.append(data.t)

        # Electronic-state-level data (one entry per occupied state).
        num_states = (data.occ > 0).sum().item()  # occupied states
        state_atom_type.append(data.atom_type.repeat(num_states))
        state_atom_pos.append(data.atom_pos.repeat(num_states, 1))
        coef_0.append(data.coef_0[:, data.occ > 0, :, :, :].flatten(start_dim=1, end_dim=2))
        delta_coef_cond.append(data.delta_coef_cond[:, data.occ > 0, :, :, :].flatten(start_dim=1, end_dim=2))
        delta_coef_target.append(data.delta_coef_target[:, data.occ > 0, :, :, :].flatten(start_dim=1, end_dim=2))
        state_phase_cond.append(data.state_phase_cond[:, data.occ > 0])
        state_phase_target.append(data.state_phase_target[:, data.occ > 0])
        state_batch.append(torch.arange(num_states).repeat(num_atoms, 1).T.flatten() + state_batch_counter)
        atom_in_state_batch.append(torch.arange(num_atoms).repeat(num_states) + state_atom_batch_counter)
        state_ind_batch.append(torch.ones(num_states, dtype=torch.long) * len(state_ind_batch))
        state_to_molecule_batch.append(torch.ones(num_states * num_atoms, dtype=torch.long) * len(state_to_molecule_batch))
        coef_mask.append(data.coef_mask.repeat(num_states, 1, 1))
        state_num_atoms.extend([num_atoms] * num_states)
        molecule_num_states.append(num_states)
        state_batch_counter += num_states
        state_atom_batch_counter += num_atoms

    molecule_data = Data(
        atom_type=torch.cat(molecule_atom_type, dim=0),
        atom_pos=torch.cat(molecule_atom_pos, dim=0),
        batch=torch.cat(molecule_batch, dim=0),  # [total_atoms] atom -> molecule id (PyG-style batch index)
        occ=torch.cat(molecule_occ, dim=0),  # [total_states] occupied-band values, concatenated over molecules
        occ_batch=torch.cat(molecule_occ_batch, dim=0),  # [total_states] state -> molecule id
        num_atoms=torch.tensor(molecule_num_atoms),  # [num_molecules] number of atoms per molecule
        efield=torch.stack(efield, dim=0),  # [num_molecules, time] electric field per molecule
        coef_mask=torch.cat(molecule_coef_mask, dim=0),
        t=torch.tensor(ts),
    )

    state_data = Data(
        atom_type=torch.cat(state_atom_type, dim=0),
        atom_pos=torch.cat(state_atom_pos, dim=0),
        coef_0=torch.cat(coef_0, dim=1),
        delta_coef_cond=torch.cat(delta_coef_cond, dim=1),
        delta_coef_target=torch.cat(delta_coef_target, dim=1),
        state_phase_cond=torch.cat(state_phase_cond, dim=1),
        state_phase_target=torch.cat(state_phase_target, dim=1),
        batch=torch.cat(state_batch, dim=0),  # [total_state_atoms] atom-in-state node -> state id
        state_atom_batch=torch.cat(atom_in_state_batch, dim=0),  # [total_state_atoms] atom-in-state node -> atom id (in molecule_data)
        state_ind_batch=torch.cat(state_ind_batch, dim=0),  # [total_states] state id -> molecule id
        mol_batch=torch.cat(state_to_molecule_batch, dim=0),  # [total_state_atoms] atom-in-state node -> molecule id
        coef_mask=torch.cat(coef_mask, dim=0),
        num_atoms=torch.tensor(state_num_atoms),  # [total_states] number of atoms in each state graph
        num_states=torch.tensor(molecule_num_states),  # [num_molecules] number of occupied states per molecule
    )

    return {
        'state_data': state_data,
        'molecule_data': molecule_data,
    }
