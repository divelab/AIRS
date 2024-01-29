import msgpack
# import h5py
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

class ProteinDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        # Sort the data list by size
        lengths = [s.shape[0] for s in data_list]
        argsort = np.argsort(lengths)               # Sort by decreasing size
        self.data_list = [data_list[i] for i in argsort]
        # Store indices where the size changes
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    batch = {prop: batch_stack([mol[prop] for mol in batch])
             for prop in batch[0].keys()}

    atom_mask = batch['atom_mask']

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch


class ProteinDataLoader(DataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False):
        super().__init__(dataset, batch_size, shuffle=shuffle,
                             collate_fn=collate_fn, drop_last=drop_last)


class ProteinLatentTransform(object):
    def __init__(self, dataset_info, include_charges, include_idx, device, sequential, group_size=5):
        # self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
        self.device = device
        self.include_charges = include_charges
        self.include_idx = include_idx
        self.sequential = sequential
        self.group_size = group_size

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data['positions'] = data[:, 0:3]
        # atom_types = torch.from_numpy(data[:, 0].astype(int)[:, None])
        # one_hot = atom_types == self.atomic_number_list
        # new_data['one_hot'] = one_hot
        new_data['h_hidden'] = data[:, 3:]
        if self.include_charges:
            new_data['charges'] = torch.zeros(n, 1, device=self.device)
        else:
            new_data['charges'] = torch.zeros(0, device=self.device)
        if self.include_idx:
            if n % self.group_size:
                atom_n = n // self.group_size + 1  # number of indexes in the new structure
            else:
                atom_n = n // self.group_size
            new_data['idx'] = torch.arange(0, atom_n, 1 / self.group_size).int()[:n].view(n, 1).to(self.device)
        else:
            new_data['idx'] = torch.zeros(n, 1, device=self.device)

        new_data['atom_mask'] = torch.ones(n, device=self.device)

        return new_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='/mnt/data/shared/limei/protein/ProtFunct')
    parser.add_argument("--split", type=str, default='Train')
    args = parser.parse_args()