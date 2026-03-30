# based on https://github.com/atomicarchitects/equiformer_v2/blob/main/nets/equiformer_v2/equiformer_v2_oc20.py
import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch_scatter

from pyexpat.model import XML_CQUANT_OPT

from .ocpmodels.models.base import BaseModel
from .ocpmodels.scn.smearing import GaussianSmearing

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
    get_cond_normalization_layer
)
from .transformer_block_dm import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from .input_block import EdgeDegreeEmbedding, EfieldEmbedding
from .graph_expansion_helper import broadcast_edge_features
# from .activation import SeparableS2Activation

from e3nn import o3

class EquiformerV2_Eband_DM(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        use_pbc=False,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,

        num_layers=6, # 6,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',

        lmax_list=[2],  # [6],
        mmax_list=[2],  # [2],
        grid_resolution=None, #18,  #None, 

        # num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal',
        time_future=1,
        time_cond=1,

        efield_cond=True,
        add_noise=False,

        avg_num_nodes=9,
        avg_degree=8,
        use_dm_quad=False,

        use_dm_delta=True,
        pred_phase=False
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels

        self.efield_cond = efield_cond
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        # self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init

        self.time_cond = time_cond  # 20
        self.time_future = time_future
        self.add_noise = add_noise

        self._AVG_NUM_NODES  = avg_num_nodes
        self._AVG_DEGREE     = avg_degree 

        self.use_dm_quad = use_dm_quad
        self.use_dm_delta = use_dm_delta

        self.pred_phase = pred_phase

        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis, # 600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2
        # self.eband_edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels // 2] * 2

        # self.edge_channels_list = [int(self.distance_expansion.num_output) + self.sphere_channels] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation_l4 = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation_l4.append(SO3_Rotation(4))

        self.SO3_rotation_l2 = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation_l2.append(SO3_Rotation(2))

        self.global_SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.global_SO3_rotation.append(SO3_Rotation(4))

        # Initialize conversion between degree l and order m layouts
        # self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        self.mappingReduced_l4 = CoefficientMappingModule([4], [4])
        self.mappingReduced_l2 = CoefficientMappingModule([2], [2])

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid_l4 = ModuleListInfo('({}, {})'.format(4, 4))
        for l in range(4 + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(4 + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid_l4.append(SO3_m_grid)

        self.SO3_grid_l2 = ModuleListInfo('({}, {})'.format(2, 2))
        for l in range(2 + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(2 + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid_l2.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            [4], # self.lmax_list,
            [4], # self.mmax_list,
            self.global_SO3_rotation,
            self.mappingReduced_l4,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self._AVG_DEGREE
        )

        self.coef_proj = SO3_LinearV2((self.time_cond * 2 + 1) * 2, self.sphere_channels, lmax=2)
        # Phase
        if self.pred_phase:
            self.eband_phase_linear = nn.Linear(self.time_cond * 2, self.sphere_channels)

        if self.efield_cond:
            self.norm_type = self.norm_type + '_cond'
            # self.cond_channels_list = [(self.time_cond + self.time_future) * 10] + [self.sphere_channels] * 2
            self.cond_channels_list_l2 = [(self.time_cond + self.time_future) * 10, self.sphere_channels, self.sphere_channels * (2 + 1) * 2]
            self.cond_channels_list_l4 = [(self.time_cond + self.time_future) * 10, self.sphere_channels, self.sphere_channels * (4 + 1) * 2]
            # self.eband_cond_channels_list = [(self.time_cond + self.time_future) * 10, self.sphere_channels // 2, self.sphere_channels // 2 * (max(self.lmax_list) + 1) * 2]
            self.norm = get_cond_normalization_layer(self.norm_type, lmax=2, num_channels=self.sphere_channels, 
                                                     mappingReduced=self.mappingReduced_l2, cond_channels_list=self.cond_channels_list_l2)
        else:
            self.cond_channels_list_l2=None
            self.cond_channels_list_l4=None
            self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)

        # self.global_attn_blocks = nn.ModuleList()
        self.eband_attn_blocks_l4 = nn.ModuleList()
        self.eband_attn_blocks_l2 = nn.ModuleList()

        self.dm_dim = 1
        if self.use_dm_delta:
            self.dm_dim += 2 * self.time_cond

        if self.use_dm_quad:
            self.dm_dim += 2 * self.time_cond

        # self.ffn_blocks = nn.ModuleList()
        for _ in range(2):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                [4],
                [4],
                self.SO3_rotation_l4,
                self.mappingReduced_l4,
                self.SO3_grid_l4,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop,
                use_attn=True,
                use_ffn=True,
                use_cond=self.efield_cond,
                cond_channels_list=self.cond_channels_list_l4,
                use_dm_feat=True,
                dm_dim=self.dm_dim
            )
            self.eband_attn_blocks_l4.append(block)

        for _ in range(4):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                [2], # self.lmax_list,
                [2], #self.mmax_list,
                self.SO3_rotation_l2,
                self.mappingReduced_l2,
                self.SO3_grid_l2,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop,
                use_attn=True,
                use_ffn=True,
                use_cond=self.efield_cond,
                cond_channels_list=self.cond_channels_list_l2,
                use_dm_feat=False
            )
            self.eband_attn_blocks_l2.append(block)

        # Output blocks for energy and forces
       
        if self.pred_phase:
            self.phase_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels, 
                2 * self.time_future,
                [2], # self.lmax_list,
                [2], # self.mmax_list,
                self.SO3_grid_l2,  
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )
        else:
            self.coef_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                4 * self.time_future,
                [2], # self.lmax_list,
                [2], # self.mmax_list,
                self.SO3_rotation_l2, 
                self.mappingReduced_l2, 
                self.SO3_grid_l2, 
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
                use_dm_feat=False
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        # FullTensorProduct(2x0e+2x1o+1x2e x 2x0e+2x1o+1x2e -> 9x0e+12x1o+5x1e+4x2o+9x2e+4x3o+1x3e+1x4e | 45 paths | 0 weights)
        self.tp = o3.FullTensorProduct(
            irreps_in1='2x0e + 2x1e + 1x2e',
            irreps_in2='2x0e + 2x1e + 1x2e'
        )

        self.dm_proj_diag = o3.Linear(
            # o3.Irreps(f'{9 * 33}x0e+{17 * 33}x1e+{13 * 33}x2e+{5 * 33}x3e+{1 * 33}x4e'),
            o3.Irreps(f'{9 * self.dm_dim}x0e+{17 * self.dm_dim}x1e+{13 * self.dm_dim}x2e+{5 * self.dm_dim}x3e+{1 * self.dm_dim}x4e'),
            o3.Irreps(f'{self.sphere_channels}x0e+{self.sphere_channels}x1e+{self.sphere_channels}x2e+{self.sphere_channels}x3e+{self.sphere_channels}x4e'),
            biases=True
            # biases=False
        )

    def to_e3nn(self, x):
        # x: ..., n_spd, C
        x = x.transpose(-1, -2)
        x_0, x_1, x_2 = x.split([1, 3, 5], dim=-1)
        x = torch.cat([x_0.flatten(start_dim=-2), x_1.flatten(start_dim=-2), x_2.flatten(start_dim=-2)], dim=-1)
        return x

    def from_e3nn(self, x):
        # x: ..., n_spd * C
        num_channels = x.shape[-1] // 9
        x_0, x_1, x_2 = x.split([num_channels, num_channels * 3, num_channels * 5], dim=-1)
        x_0 = x_0.reshape(*x_0.shape[:-1], num_channels, 1)
        x_1 = x_1.reshape(*x_1.shape[:-1], num_channels, 3)
        x_2 = x_2.reshape(*x_2.shape[:-1], num_channels, 5)
        x = torch.cat([x_0, x_1, x_2], dim=-1)
        x = x.transpose(-1, -2)
        return x

    def concat_dms(self, xs):
        # xs: ncond, N, C
        xs = xs.transpose(0, 1) # N, ncond, C
        x_0, x_1, x_2, x_3, x_4 = xs.split([9, 17 * 3, 13 * 5, 5 * 7, 1 * 9], dim=-1)
        res = torch.cat([
            x_0.flatten(start_dim=1),
            x_1.flatten(start_dim=1),
            x_2.flatten(start_dim=1),
            x_3.flatten(start_dim=1),
            x_4.flatten(start_dim=1)], dim=-1)
        return res

    def forward(self, batch_data, max_state_samples=None):

        graph_data = batch_data['state_data']
        global_data = batch_data['molecule_data']

        delta_coef_cond = graph_data.delta_coef_cond  # nhist, N, 9, 2
        coef_0 = graph_data.coef_0.real  # 1, N, 9, 2
        if self.pred_phase:
            state_phase_cond = graph_data.state_phase_cond # nhist, num_bands

        if self.training and self.add_noise:
            eps = 0.01 * torch.randn_like(delta_coef_cond, dtype=torch.complex64) * batch_data['state_data'].coef_mask.unsqueeze(0)
            delta_coef_cond = delta_coef_cond + eps

        # cond
        if self.time_cond > 0:
            coef_cond = torch.cat([delta_coef_cond.real, delta_coef_cond.imag, coef_0], dim=0)  # nhist * 2 + 1, N, 9, 2

        perm_ind = [1, 2, 0]

        data = Data(pos=graph_data.atom_pos[:, perm_ind], atomic_numbers=graph_data.atom_type, natoms=graph_data.num_atoms, batch=graph_data.batch)
        global_data = Data(pos=global_data.atom_pos[:, perm_ind], atomic_numbers=global_data.atom_type, natoms=global_data.num_atoms, batch=global_data.batch)
        if self.efield_cond:
            efield = batch_data['molecule_data'].efield
        else:
            efield = None

        self.dtype = data.pos.dtype
        self.device = data.pos.device

        # Global Graph
        global_atomic_numbers = global_data.atomic_numbers.long()

        (
            global_edge_index,
            global_edge_distance,
            global_edge_distance_vec,
            global_cell_offsets,
            _,  # cell offset distances
            global_neighbors,
        ) = self.generate_graph(global_data)

        global_data.edge_index = global_edge_index
        eband_edge_data = broadcast_edge_features(
            global_data=global_data,
            batch_data=batch_data,
            max_state_samples=max_state_samples,
            device=self.device,
        )
        all_eband_edge_index = eband_edge_data["all_eband_edge_index"]
        all_eband_edge_batch = eband_edge_data["all_eband_edge_batch"]
        sampled_eband_edge_index = eband_edge_data["sampled_eband_edge_index"]
        sampled_eband_edge_batch = eband_edge_data["sampled_eband_edge_batch"]
        sampled_mask = eband_edge_data["sampled_mask"]
        sampled_graph_batch = eband_edge_data["sampled_graph_batch"]

        coef_cond = coef_cond[:, sampled_mask, :, :]
        # # sample eband phase
        if self.pred_phase:
            sampled_eband_inds = torch.unique(batch_data['state_data'].batch[sampled_mask]) # which eband inds are sampled
            state_phase_cond = state_phase_cond[:, sampled_eband_inds]
            state_phase_cond = state_phase_cond.transpose(0, 1) # num_sampled_bands, nhist
            state_phase_cond = torch.cat([state_phase_cond.real, state_phase_cond.imag], dim=1) # num_sampled_bands, 2 * nhist

        num_atoms = coef_cond.shape[1]

        # DM
        def calc_dm(c_0_left, c_0_right, delta_t_left, delta_t_right, occ_num, eband_pair_batch):
            """
            :param c_0_left, c_0_right: 1, num_pairs, 13
            :param delta_t_left, delta_t_right: nhist, num_pairs, 13
            :param occ_num: num_pairs
            :param eband_batch: num_pairs            
            :return dm: nhist, num_pairs_global, 169 (=13x13) x (nhist * num_dms + 1)
            """
            # 1, num_pairs, 169 (=13x13)
            dm_0 = self.tp(c_0_left, c_0_right)
            dms = [dm_0]

            if self.use_dm_delta:
                # nhist, num_pairs, 169
                dm_0_delta_real = self.tp(c_0_left, delta_t_right.real) \
                                + self.tp(delta_t_left.real, c_0_right)

                dm_0_delta_imag = - self.tp(c_0_left, delta_t_right.imag) \
                                + self.tp(delta_t_left.imag, c_0_right)

                dms.extend([dm_0_delta_real, dm_0_delta_imag])

            if self.use_dm_quad:
                dm_delta_real = self.tp(delta_t_left.real, delta_t_right.real) \
                                + self.tp(delta_t_left.imag, delta_t_right.imag)
                
                dm_delta_imag = - self.tp(delta_t_left.real, delta_t_right.imag) \
                                + self.tp(delta_t_left.imag, delta_t_right.real)

                # dms = torch.cat([dm_0, dm_0_delta_real, dm_0_delta_imag, dm_delta_real, dm_delta_imag], dim=0)
                dms.extend([dm_delta_real, dm_delta_imag])
            # else:
            #     dms = torch.cat([dm_0, dm_0_delta_real, dm_0_delta_imag], dim=0)
            dms = torch.cat(dms, dim=0)
            dm = self.concat_dms(dms) # num_pairs, 169x(nhist * num_dms + 1)
            # Reduce over energy bands
            # dm = torch_scatter.scatter_sum(dm * occ_num.float().unsqueeze(1), eband_pair_batch, dim=0) # num_pairs_global, 169x(nhist * num_dms + 1)
            dm = torch_scatter.scatter_sum(dm * occ_num.unsqueeze(1), eband_pair_batch, dim=0) # num_pairs_global, 169x(nhist * num_dms + 1)
            return dm

        # DM off diagonal
        coef_0_e3nn = self.to_e3nn(graph_data.coef_0.real)[..., :13] # 1, N, 13
        coef_0_e3nn_src = coef_0_e3nn[:, all_eband_edge_index[0], :]
        coef_0_e3nn_dst = coef_0_e3nn[:, all_eband_edge_index[1], :]
        delta_coef_cond_e3nn = self.to_e3nn(graph_data.delta_coef_cond)[..., :13] # nhist, N, 13
        delta_coef_cond_e3nn_src = delta_coef_cond_e3nn[:, all_eband_edge_index[0], :]
        delta_coef_cond_e3nn_dst = delta_coef_cond_e3nn[:, all_eband_edge_index[1], :]

        # DM diagonal blocks
        dm_diag = calc_dm(
            c_0_left=coef_0_e3nn,
            c_0_right=coef_0_e3nn,
            delta_t_left=delta_coef_cond_e3nn,
            delta_t_right=delta_coef_cond_e3nn,
            occ_num=batch_data['molecule_data'].occ[batch_data['state_data'].batch].float(),
            eband_pair_batch=batch_data['state_data'].state_atom_batch
        )

        dm_diag_feat = self.dm_proj_diag(dm_diag)
        dm_feat_split = dm_diag_feat.split([self.sphere_channels * d for d in [1, 3, 5, 7, 9]], dim=-1)
        dm_diag_feat = torch.cat([feat.reshape(feat.shape[0], self.sphere_channels, -1) for feat in dm_feat_split], dim=-1).transpose(1, 2)        

        eband_edge_mol_batch = batch_data['state_data'].batch[all_eband_edge_index[0]]

        dm_off_diag = calc_dm(
            c_0_left=coef_0_e3nn_src,
            c_0_right=coef_0_e3nn_dst,
            delta_t_left=delta_coef_cond_e3nn_src,
            delta_t_right=delta_coef_cond_e3nn_dst,
            occ_num=batch_data['molecule_data'].occ[eband_edge_mol_batch].float(),
            eband_pair_batch=all_eband_edge_batch
        )

        # Compute 3x3 rotation matrix per edge
        global_edge_rot_mat = self._init_edge_rot_mat(
            global_data, global_edge_index, global_edge_distance_vec
        )
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.global_SO3_rotation[i].set_wigner(global_edge_rot_mat)
            self.SO3_rotation_l4[i].set_wigner(global_edge_rot_mat[sampled_eband_edge_batch])
            self.SO3_rotation_l2[i].set_wigner(global_edge_rot_mat[sampled_eband_edge_batch])

        # Edge encoding (distance and atom edge)
        global_edge_distance = self.distance_expansion(global_edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = global_atomic_numbers[global_edge_index[0]]  # Source atom atomic number
            target_element = global_atomic_numbers[global_edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            global_edge_distance = torch.cat((global_edge_distance, source_embedding, target_embedding), dim=1)

        # Init per node representations using an atomic number based embedding
        x_global = SO3_Embedding(
            global_data.pos.shape[0],
            [4], # self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )
        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x_global.embedding[:, offset_res, :] = self.sphere_embedding(global_atomic_numbers)
            else:
                x_global.embedding[:, offset_res, :] = self.sphere_embedding(
                    global_atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            # offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
            offset_res = offset_res + int((4 + 1) ** 2)

        # Edge-degree embedding
        global_edge_degree = self.edge_degree_embedding(
            global_atomic_numbers,
            global_edge_distance,
            global_edge_index)
        x_global.embedding = x_global.embedding + global_edge_degree.embedding

        # Eband local embedding
        num_conds = self.time_cond * 2 + 1
        coef_emb = coef_cond.permute(1, 2, 3, 0).reshape(num_atoms, 9, 2 * num_conds)
        x = SO3_Embedding(
            num_atoms,
            [2], # self.lmax_list,
            num_conds * 2,
            self.device,
            self.dtype,
        )
        x.embedding[:, :9, :] = coef_emb
        x = self.coef_proj(x)
        # if self.training:
        #     x.embedding = x.embedding + x_global.embedding
        # else:

        x.embedding = torch.cat([x.embedding, torch.zeros(x.embedding.shape[0], 7 + 9, x.embedding.shape[2], dtype=x.embedding.dtype).to(self.device)], dim=1)
        x.set_lmax_mmax([4], [4])
        x.embedding = x.embedding + x_global.embedding[graph_data.state_atom_batch[sampled_mask]] + dm_diag_feat[batch_data['state_data'].state_atom_batch[sampled_mask]]

        # Add eband phase embedding
        if self.pred_phase:
            eband_phase_embedding = self.eband_phase_linear(state_phase_cond) # num_sampled_bands, C

            def sample_batch(batch, mask):
                _, contiguous_batch = torch.unique(batch[mask], return_inverse=True)
                return contiguous_batch
            
            x.embedding[:, :1, :] = x.embedding[:, :1, :] + eband_phase_embedding[sample_batch(batch_data['state_data'].batch, sampled_mask), None, :]

        # Network blocks
        for i in range(2):
            x = self.eband_attn_blocks_l4[i](
                x=x,                  # SO3_Embedding
                global_atomic_numbers=global_atomic_numbers,
                global_edge_distance=global_edge_distance,
                edge_index=sampled_eband_edge_index,
                global_edge_index=global_edge_index,
                batch=sampled_graph_batch,    # for GraphDropPath
                cond_input=efield,
                cond_batch=batch_data['state_data'].mol_batch[sampled_mask],
                eband_edge_batch=sampled_eband_edge_batch,
                dm_off_diag=dm_off_diag
            )

        x.embedding = x.embedding[:, :9, :]
        x.set_lmax_mmax([2], [2])

        for i in range(4):
            x = self.eband_attn_blocks_l2[i](
                x=x,                  # SO3_Embedding
                global_atomic_numbers=global_atomic_numbers,
                global_edge_distance=global_edge_distance,
                edge_index=sampled_eband_edge_index,
                global_edge_index=global_edge_index,
                batch=sampled_graph_batch,    # for GraphDropPath
                cond_input=efield,
                cond_batch=batch_data['state_data'].mol_batch[sampled_mask],
                eband_edge_batch=sampled_eband_edge_batch,
                dm_off_diag=None
            )

        # Final layer norm
        if self.efield_cond:
            x.embedding = self.norm(
                x.embedding, 
                cond_input=efield, 
                cond_batch=batch_data['state_data'].mol_batch[sampled_mask]
            )
        else:
            x.embedding = self.norm(x.embedding)


        if not self.pred_phase:
            coefs = self.coef_block(
                x=x,
                global_atomic_numbers=global_atomic_numbers,
                global_edge_distance=global_edge_distance,
                edge_index=sampled_eband_edge_index,
                global_edge_index=global_edge_index,
                eband_edge_batch=sampled_eband_edge_batch,
                dm_off_diag=None
            )

            out_coef_complex = coefs.embedding[:, :9, :].reshape(num_atoms, 9, 2, self.time_future, 2).permute(4, 3, 0, 1, 2)
            out_coef_complex = out_coef_complex[0].float() + out_coef_complex[1].float() * 1j  # time_future, N, 9, 2 
            out_coef_complex = out_coef_complex * batch_data['state_data'].coef_mask[sampled_mask].unsqueeze(0)

            pred = dict(
                delta_coef_t_norm=out_coef_complex, 
                sampled_mask=sampled_mask,
                )
        
        else:
            node_phase = self.phase_block(x).embedding
            node_phase = node_phase[:, 0, :]

            phase = torch_scatter.scatter_sum(node_phase, sample_batch(batch_data['state_data'].batch, sampled_mask), dim=0)
            sampled_eband_inds = torch.unique(batch_data['state_data'].batch[sampled_mask]) # which eband inds are sampled
            phase = phase / batch_data['state_data'].num_atoms[sampled_eband_inds, None]
            out_phase_complex = phase[:, :self.time_future].float() + phase[:, self.time_future:].float() * 1j
            out_phase_complex = out_phase_complex.transpose(0, 1) # time_future. num_ebands_sampled

            pred = dict(
                state_phase_t = out_phase_complex,
                sampled_mask=sampled_mask,
                )

        return pred


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
    
