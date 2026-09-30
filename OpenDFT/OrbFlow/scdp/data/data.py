import numpy as np
import torch
from copy import deepcopy
from scdp.common.pyg import Data
from scdp.data.vnode import get_virtual_nodes
from scdp.data.utils import (
    read_pmg_pkl, read_vasp, read_cube, read_json, calculate_grid_pos, make_graph
)

class AtomicNumberTable:
    def __init__(self, zs):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)
    
def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    "adpated from: https://github.com/ACEsuit/mace/blob/main/mace"
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    adpted from: https://github.com/ACEsuit/mace/blob/main/mace
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)

class AtomicData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def build_graph_with_vnodes(
        cls,
        atom_types,
        atom_coords,
        cell,
        chg_density,
        origin,
        metadata: str,
        z_table: AtomicNumberTable,
        atom_cutoff: float = 5.0,
        disable_pbc: bool = False,
        vnode_method: str = "bond",
        vnode_factor: int = 2,
        vnode_res: float = 0.6,
        device: str = "cpu",
        max_neighbors: int = 24,
        struct=None,
        bond_length_scale: float = 1.0,
        grid_convention: str = "corner",
    ) -> "AtomicData":        
        n_atom = len(atom_coords)
        if vnode_method in ["voronoi", "bond", "both"]:
            coords_for_vnode = atom_coords
            if bond_length_scale != 1.0:
                coords_for_vnode = atom_coords * bond_length_scale
            virtual_nodes = get_virtual_nodes(
                coords_for_vnode, cell, not disable_pbc, 
                vnode_method, 
                vnode_factor,
                resolution=vnode_res,
                atom_types=atom_types, 
                struct=struct,
            )
            if virtual_nodes is not None and bond_length_scale != 1.0:
                virtual_nodes = virtual_nodes / bond_length_scale
        else:
            virtual_nodes = None
            
        if virtual_nodes is not None:
            virtual_nodes = torch.from_numpy(virtual_nodes).float()
            
            n_vnode = len(virtual_nodes)
            atom_types = torch.cat([atom_types, torch.zeros(len(virtual_nodes), dtype=torch.long)])
            atom_coords = torch.cat([atom_coords, virtual_nodes])
            is_vnode = torch.cat([torch.zeros(len(atom_coords) - len(virtual_nodes), dtype=torch.bool), 
                                torch.ones(len(virtual_nodes), dtype=torch.bool)])
        else:
            n_vnode = 0
            is_vnode = torch.zeros(n_atom, dtype=torch.bool)
            
        idx_src, idx_dst, shifts, unit_shifts = make_graph(
            cell.to(device),
            atom_coords.to(device),
            atom_coords.to(device),
            atom_cutoff,
            disable_pbc,
            disable_sc=True,
            max_neigh=max_neighbors
        )
        edge_index = torch.stack([idx_src, idx_dst], dim=0)
        
        indices = atomic_numbers_to_indices(atom_types, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        probe_coords = calculate_grid_pos(
            chg_density, cell, origin, grid_convention=grid_convention
        ).view(-1, 3)

        data_dict = dict(
            edge_index=edge_index,
            coords=atom_coords,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell.unsqueeze(0),
            atom_types=atom_types,
            node_attrs=one_hot,
            n_atom=n_atom,
            n_vnode=n_vnode,
            num_nodes=n_atom + n_vnode,
            probe_coords=probe_coords,
            n_probe=probe_coords.shape[0],
            chg_labels=chg_density.flatten(),
            grid_size=torch.tensor(chg_density.shape, dtype=torch.long).unsqueeze(0),
            is_vnode=is_vnode,
            metadata=metadata,
            vnode_method=vnode_method,
            build_method="vnode",
            grid_convention=grid_convention,
        )
        return cls(**data_dict)
    
    @classmethod
    def build_graph_with_all_probes(
        cls,
        atom_types,
        atom_coords,
        cell,
        chg_density,
        origin,
        metadata: str,
        z_table: AtomicNumberTable,
        atom_cutoff: float = 4.0,
        probe_cutoff: float = 4.0,
        disable_pbc: bool = False,
        device: str = "cpu",
        grid_convention: str = "corner",
    ) -> "AtomicData":
        """
        build the graph with all voxel grid (probe) coords.
        """ 
        idx_src, idx_dst, shifts, unit_shifts = make_graph(
            cell.to(device),
            atom_coords.to(device),
            atom_coords.to(device),
            atom_cutoff,
            disable_pbc,
            disable_sc=True,
        )
        edge_index = torch.stack([idx_src, idx_dst], dim=0)
        
        probe_coords = calculate_grid_pos(
            chg_density, cell, origin, grid_convention=grid_convention
        ).view(-1, 3)
        idx_atom, idx_probe, p_shifts, _ = make_graph(
            cell.to(device),
            atom_coords.to(device),
            probe_coords.to(device),
            probe_cutoff,
            disable_pbc,
        )
        
        indices = atomic_numbers_to_indices(atom_types, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        data_dict = dict(
            edge_index=edge_index,
            coords=atom_coords,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell.unsqueeze(0),
            atom_types=atom_types,
            node_attrs=one_hot,
            probe_coords=probe_coords,
            n_atom=atom_coords.shape[0],
            n_probe=probe_coords.shape[0],
            num_nodes=atom_coords.shape[0] + probe_coords.shape[0],
            idx_probe=idx_probe,
            idx_atom=idx_atom,
            p_shifts=p_shifts,
            chg_labels=chg_density,
            metadata=metadata,
            build_method="probe"
        )
        return cls(**data_dict)

    @classmethod
    def from_file(
        cls,
        fpath=None,
        fcontent=None,
        finfo=None,
        build_method: str = "vnode",
        z_table: AtomicNumberTable = None,
        atom_cutoff: float = 4.0,
        probe_cutoff: float = 4.0,
        disable_pbc: bool = False,
        vnode_method: str = "bond",
        vnode_factor: int = 2,
        vnode_res: float = 0.6,
        max_neighbors: int = 24,
        device: str = "cpu",
    ):
        assert build_method in ["probe", "vnode"], "<build_method> must be either <probe> or <vnode>."
        assert (fpath or (fcontent and finfo)) and not (fpath and (fcontent and finfo))
                
        if fpath:
            data = read_pmg_pkl(fpath)
            metadata = fpath.split("/")[-1].split(".")[0]
        else:
            if finfo.name.endswith((".cube", ".cube.gz", "cube.zz", ".cube.xz", "cube.lz4")):
                data = read_cube(fcontent)
            # its json here
            elif finfo.name.endswith((".json", ".json.gz", ".json.zz", ".json.xz", ".json.lz4")):
                data = read_json(fcontent)
            else:
                data = read_vasp(fcontent)
            metadata = finfo.name

        if build_method == "probe":
            return cls.build_graph_with_all_probes(
                *data[:-1],
                metadata=metadata,
                z_table=z_table,
                atom_cutoff=atom_cutoff,
                probe_cutoff=probe_cutoff,
                disable_pbc=disable_pbc,
                device=device,
            )
        elif build_method == "vnode":
            return cls.build_graph_with_vnodes(
                *data[:-1],
                metadata=metadata,
                z_table=z_table,
                atom_cutoff=atom_cutoff,
                disable_pbc=disable_pbc,
                vnode_method=vnode_method,
                vnode_factor=vnode_factor,
                vnode_res=vnode_res,
                device=device,
                max_neighbors=max_neighbors,
                struct=data[-1],
            )
    
    def sample_probe(self, n_probe: int = 200, use_block=False) -> Data:
        if self.build_method == "vnode":
            data = deepcopy(self)
            if use_block:
                grid_size = self.grid_size[0]
                side_len = grid_size * (n_probe / torch.prod(grid_size)).pow(1/3)
                start_x = torch.randint(0, grid_size[0] - side_len[0] + 1)
                start_y= torch.randint(0, grid_size[1] - side_len[1] + 1)
                start_z = torch.randint(0, grid_size[2] - side_len[2] + 1)
                data['chg_labels'] = self.chg_labels.view(
                    *grid_size)[
                        start_x:start_x+side_len[0], 
                        start_y:start_y+side_len[1], 
                        start_z:start_z+side_len[2]
                    ].flatten()
                data['probe_coords'] = self.probe_coords.view(
                    *grid_size, 3)[
                        start_x:start_x+side_len[0], 
                        start_y:start_y+side_len[1], 
                        start_z:start_z+side_len[2]
                    ].view(-1, 3)
                data['n_probe'] = torch.prod(side_len)
                data['sampled'] = True
            else:
                n_probe_total = self.n_probe
                sampled_probes = torch.randperm(n_probe_total)[:n_probe]
                data['chg_labels'] = self.chg_labels[sampled_probes]
                data['probe_coords'] = self.probe_coords[sampled_probes]
                data['n_probe'] = n_probe
                data['sampled'] = True
            return data
        
        elif self.build_method == "probe":
            n_probe_total = self.n_probe
            sampled_probes = torch.randperm(n_probe_total)[:n_probe]
            sampled_mask = torch.isin(self.idx_probe, sampled_probes)
            probe_one_hot = to_one_hot(
                torch.zeros(n_probe, dtype=torch.long)
                .to(self.node_attrs.device)
                .view(-1, 1),
                num_classes=self.node_attrs.shape[1],
            )

            node_attrs = torch.cat([self.node_attrs, probe_one_hot], dim=0)
            probe_coords = self.probe_coords[sampled_probes]
            coords = torch.cat([self.coords, probe_coords], dim=0)

            mapped_probe_idx = (
                self.idx_probe[sampled_mask].repeat(n_probe, 1).T == sampled_probes
            ).nonzero()[:, 1] + self.n_atom
            p_edges = torch.stack([self.idx_atom[sampled_mask], mapped_probe_idx])
            p_shifts = self.p_shifts[sampled_mask]
            edge_index = torch.cat([self.edge_index, p_edges], dim=1)
            shifts = torch.cat([self.shifts, p_shifts], dim=0)
            chg_labels = self.chg_labels.flatten()[sampled_probes]

            return Data(
                edge_index=edge_index,
                coords=coords,
                shifts=shifts,
                atom_types=torch.cat(
                    [self.atom_types, torch.zeros(n_probe, dtype=torch.long, device=self.atom_types.device)]),
                node_attrs=node_attrs,
                chg_labels=chg_labels,
                cell=self.cell,
                num_nodes=node_attrs.shape[0],
                n_atom=self.n_atom,
                n_probe=n_probe,
                n_node=self.n_atom + n_probe,
                is_probe=torch.cat(
                    [
                        torch.zeros(self.n_atom, dtype=torch.bool),
                        torch.ones(n_probe, dtype=torch.bool),
                    ]
                ),
                sampled=True,
            )
        else:
            raise NotImplementedError("Only <vnode> and <probe> build methods are supported.")