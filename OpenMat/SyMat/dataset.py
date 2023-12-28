import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import build_crystal_graph, build_crystal


class MatDataset(Dataset):
    def __init__(self, data_path, prop_name=None, graph_method='crystalnn'):
        super().__init__()
        
        self.graph_method = graph_method
        self.prop_name = prop_name

        if data_path[-3:] == '.pt':
            self.data_dict_list = torch.load(data_path)
        elif data_path[-4:] == '.csv':
            self.df = pd.read_csv(data_path)
            self._preprocess(data_path[:-4] + '.pt')
        else:
            raise NotImplementedError
    

    def _preprocess(self, preprocessed_file_path):
        data_dict_list = []

        for i in tqdm(range(len(self.df))):
            data_dict = self._get_mat_data(i)
            data_dict_list.append(data_dict)
        
        self.data_dict_list = data_dict_list
        torch.save(self.data_dict_list, preprocessed_file_path)
        

    def _get_mat_data(self, index):
        if hasattr(self, 'data_dict_list'):
            return self.data_dict_list[index]

        row = self.df.iloc[index]
        crystal_str = row['cif']
        crystal = build_crystal(crystal_str, niggli=True, primitive=False)
        graph_arrays = build_crystal_graph(crystal, self.graph_method)

        data_dict = {
            'mp_id': row['material_id'],
            'cif': crystal_str,
            'graph_arrays': graph_arrays
        }

        if self.prop_name in row.keys():
            data_dict[self.prop_name] = row[self.prop_name]
        
        return data_dict

    
    def __len__(self):
        if hasattr(self, 'data_dict_list'):
            return len(self.data_dict_list)
        return len(self.df)
    
    
    def __getitem__(self, index):
        data_dict = self._get_mat_data(index)
        prop = torch.tensor(data_dict[self.prop_name])
        (frac_coords, atom_types, scaled_lengths, lengths, angles, edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']
        data = Data(
            frac_coords=torch.Tensor(frac_coords).float(),
            atom_types=torch.LongTensor(atom_types),
            scaled_lengths=torch.Tensor(scaled_lengths).view(1, -1),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        return data
            