import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

import hienet._keys as KEY
from hienet._const import AtomGraphDataType


@compile_mode('script')
class ForceOutputFromEdge(nn.Module):
    """
    works when edge_vec.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_edge_vec: str = KEY.EDGE_VEC,
        data_key_edge_idx: str = KEY.EDGE_IDX,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_num_atoms: str = KEY.NUM_ATOMS,
    ):
        super().__init__()
        self.key_edge_vec = data_key_edge_vec
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_edge_idx = data_key_edge_idx
        self.key_num_atoms = data_key_num_atoms

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        tot_num = torch.sum(data[self.key_num_atoms])
        edge_idx = data[self.key_edge_idx]

        edge_vec_tensor = [data[self.key_edge_vec]]
        energy = [(data[self.key_energy]).sum()]

        dE_dr = torch.autograd.grad(
            energy, edge_vec_tensor, create_graph=self.training
        )[0]

        if dE_dr is not None:
            force = torch.zeros(tot_num, 3)
            force = scatter(dE_dr, edge_idx[0], dim=0, dim_size=tot_num)
            force -= scatter(dE_dr, edge_idx[1], dim=0, dim_size=tot_num)

            data[self.key_force] = force

        return data


@compile_mode('script')
class ForceOutput(nn.Module):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
    ):
        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = [data[self.key_pos]]
        energy = [(data[self.key_energy]).sum()]

        grad = torch.autograd.grad(
            energy,
            pos_tensor,
            create_graph=self.training,
        )[0]

        # without this 'if', type(grad) is 'Optional[Tensor]' which result in error
        if grad is not None:
            data[self.key_force] = torch.neg(grad)
        return data


@compile_mode('script')
class ForceStressOutput(nn.Module):
    def __init__(
        self,
        data_key_pos: str = KEY.POS,
        data_key_energy: str = KEY.SCALED_ENERGY,
        data_key_force: str = KEY.SCALED_FORCE,
        data_key_stress: str = KEY.SCALED_STRESS,
        data_key_cell_volume: str = KEY.CELL_VOLUME,
    ):

        super().__init__()
        self.key_pos = data_key_pos
        self.key_energy = data_key_energy
        self.key_force = data_key_force
        self.key_stress = data_key_stress
        self.key_cell_volume = data_key_cell_volume
        self._is_batch_data = True

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        pos_tensor = data[self.key_pos]
        energy = [(data[self.key_energy]).sum()]

        grad = torch.autograd.grad(
            energy,
            [pos_tensor, data['_strain']],
            create_graph=self.training,
        )

        # make grad is not Optional[Tensor]
        fgrad = grad[0]
        if fgrad is not None:
            data[self.key_force] = torch.neg(fgrad)

        sgrad = grad[1]
        volume = data[self.key_cell_volume]
        if sgrad is not None:
            if self._is_batch_data:
                stress = sgrad / volume.view(-1, 1, 1)
                stress = torch.neg(stress)
                voigt_stress = torch.vstack((
                    stress[:, 0, 0],
                    stress[:, 1, 1],
                    stress[:, 2, 2],
                    stress[:, 0, 1],
                    stress[:, 1, 2],
                    stress[:, 0, 2],
                ))
                data[self.key_stress] = voigt_stress.transpose(0, 1)
            else:
                stress = sgrad / volume
                stress = torch.neg(stress)
                voigt_stress = torch.stack((
                    stress[0, 0],
                    stress[1, 1],
                    stress[2, 2],
                    stress[0, 1],
                    stress[1, 2],
                    stress[0, 2],
                ))
                data[self.key_stress] = voigt_stress

        return data
