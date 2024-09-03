import os.path as osp
import numpy as np
import ase.io
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader


class Pyg3BPA(InMemoryDataset):
    def __init__(
        self,
        root="dataset/",
        name="train_300K",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.processed_folder = osp.join(root, name)

        super(Pyg3BPA, self).__init__(
            self.processed_folder, transform, pre_transform, pre_filter
        )

        if osp.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self) -> str:
        return "./dataset/3bpa/xyz_data/"

    @property
    def raw_file_names(self):
        return osp.join(self.name + ".xyz")

    @property
    def processed_file_names(self):
        return self.name + "_pyg.pt"

    def download(self):
        pass

    def process(self):
        print(osp.join(self.raw_dir, self.raw_file_names))
        mol_list = ase.io.read(osp.join(self.raw_dir, self.raw_file_names), index=":")
        print(len(mol_list))
        data_list = []
        for i in tqdm(range(len(mol_list))):
            mol = mol_list[i]
            R = mol.arrays.get("positions", None)
            z = mol.arrays.get("numbers", None)
            E = mol.info.get("energy", None)  # eV
            F = mol.arrays.get("forces", None)  # eV / Ang

            R = torch.tensor(R, dtype=torch.float32)
            z = torch.tensor(z, dtype=torch.int64)
            E = torch.tensor(E, dtype=torch.float32)
            F = torch.tensor(F, dtype=torch.float32)

            data = Data(pos=R, z=z, y=E, force=F)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
    
    def get_idx_split(self, data_size, valid_fraction, seed):
        train_size = data_size - int(valid_fraction * data_size)
        ids = list(range(data_size))
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:]
        )
        split_dict = {"train": train_idx, "valid": val_idx}
        return split_dict


if __name__ == "__main__":
    name = "train_300K"
    dataset = Pyg3BPA(root="./3bpa/pyg_data/", name=name)
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    print(dataset.data.y.shape)
    print(dataset.data.force.shape)
    print(dataset[0])
    if name[:5] == "train":
        split_idx = dataset.get_idx_split(
            len(dataset.data.y), valid_fraction=0.1, seed=3
        )
        print(dataset[split_idx["train"]])
        train_dataset, valid_dataset = (
            dataset[split_idx["train"]],
            dataset[split_idx["valid"]],
        )
        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        data = next(iter(train_loader))
        print(data)
    else:
        test_loader = DataLoader(dataset, batch_size=5, shuffle=True)
        data = next(iter(test_loader))
        print(data)

