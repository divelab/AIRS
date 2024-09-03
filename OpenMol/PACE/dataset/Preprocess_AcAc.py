import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_cluster import radius_graph


class PreprocessAcAc(InMemoryDataset):
    def __init__(
        self, pyg_dataset, cutoff, transform=None, pre_transform=None, pre_filter=None
    ):
        self.dataset = pyg_dataset
        self.cutoff = cutoff
        self.processed_folder = "/tmp/acac"
        self.average_energy = torch.stack([data.y for data in pyg_dataset]).mean()
        super(PreprocessAcAc, self).__init__(
            self.processed_folder, transform, pre_transform, pre_filter
        )
        self.process()

    @property
    def raw_dir(self) -> str:
        return "./dataset/acac/xyz_data/"

    @property
    def raw_file_names(self):
        return "None"

    @property
    def processed_file_names(self):
        return "None"

    def download(self):
        pass

    def process(self):
        z_set = set()
        for data in self.dataset:
            z_set.update(data.z.tolist())
        self.z_table = sorted(z_set)

        z_map = torch.zeros(9)
        for i, z in enumerate(self.z_table):
            z_map[z] = i
        z_map = z_map.reshape(-1, 1).float()

        atom_energy_list, force_list, neighbor_counts, z_counts, data_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for data in self.dataset:
            ### Compute scaling statistics & onehot node attrs
            onehot_full = torch.nn.functional.one_hot(data.z, num_classes=9).float()
            new_idx = torch.mm(onehot_full, z_map).reshape(-1).type(torch.int64)
            onehot = torch.nn.functional.one_hot(new_idx).float()

            atom_energy_list.append((data.y - self.average_energy) / len(data.z))
            force_list.append(data.force)

            ### Build edge
            edge_index = radius_graph(data.pos, r=self.cutoff, batch=None)
            neighbor_counts.append(edge_index.shape[-1])
            z_counts.append(len(data.z))

            new_data = Data(
                pos=data.pos,
                z=data.z,
                y=data.y,
                force=data.force,
                node_attrs=onehot,
                edge_index=edge_index,
            )
            data_list.append(new_data)

        atom_energy_list = torch.cat(atom_energy_list, dim=0)
        force_list = torch.cat(force_list, dim=0)

        self.mean = torch.mean(atom_energy_list).item()
        self.std = torch.sqrt(torch.mean(torch.square(force_list))).item()
        self.avg_num_neighbors = torch.sum(torch.tensor(neighbor_counts)) / torch.sum(
            torch.tensor(z_counts)
        )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)


if __name__ == "__main__":
    from PygAcAc import PygAcAc

    name = "test_MD_600K"
    pyg_dataset = PygAcAc(root="./acac/pyg_data/", name=name)
    split_idx = pyg_dataset.get_idx_split(
        len(pyg_dataset.data.y), valid_fraction=0.1, seed=3
    )
    train_dataset = pyg_dataset[split_idx["train"]]
    print(train_dataset)
    dataset = PreprocessAcAc(train_dataset, cutoff=5)
    print(dataset)
    print(dataset[0])
    print(dataset.mean, dataset.std)
    print(dataset.avg_num_neighbors)
