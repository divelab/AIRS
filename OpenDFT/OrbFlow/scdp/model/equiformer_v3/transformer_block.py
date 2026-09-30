import torch
import math
import copy

from .activation import (
    SmoothLeakyReLU,
    check_activation_name,
    get_activation,
    has_scalars,
    add_dropout,
    prepare_activation_forward_param,
    LinearSwiGLU,
    LinearSquare,
    SwiGLU
)
from .layer_norm import (
    get_normalization_layer,
    RMSNorm
)
from .radial_function import RadialFunction
from .so2_ops import SO2Linear
from .so3 import (
    SO3Linear,
    SO3Grid
)
from .utils import reduce_edge
from .drop import (
    GraphDropPath,
    EquivariantDropout
)
from .softmax import GraphSoftmax


class EquivariantGraphAttention(torch.nn.Module):
    """
        EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels
            num_heads (int):            Number of attention heads
            attn_alpha_head (int):      Number of channels for alpha vector in each attention head
            attn_value_head (int):      Number of channels for value vector in each attention head
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid`

            max_num_elements (int):     Maximum number of atomic numbers
            edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                            The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
            use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
            
            activation (str):           Type of activation function.
                                        Please see `check_activation_name()` in `./activation.py`.
            
            use_attn_renorm (bool):     Whether to re-normalize attention weights
            use_add_merge (bool):       Default: False
                                        If True, use addition to merge the source/target node features instead of concat, 
                                        which can save 2x compute when rotating with Wigner-D matrices.
            use_rad_l_parametrization (bool):
                                        Default: True
                                        If True, all the m components within the same type-L vector will share the same
                                        weight from the radial function.
            softcap (float):            Default: None
                                        If not None, use soft capping to limit the range of attention logits to
                                        [- `softcap`, + `softcap`].
            eps (float):                Default: 1e-16
                                        Epsilon value used in the softmax operation
            
            alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
            attn_mask_rate (float):     Mask rate for neighbors considered in attention
            attn_weights_drop (float):  Dropout rate for attention weights
            value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
    """
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        eps=1e-16,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0
    ):
        super().__init__()

        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        
        self.so3_rotation = so3_rotation
        self.grid_resolution_list = grid_resolution_list

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        self.activation         = activation
        self.use_attn_renorm    = use_attn_renorm
        self.use_add_merge      = use_add_merge
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.softcap            = softcap
        self.eps                = eps

        # Radial function
        if self.use_atom_edge_embedding:
            self.source_embedding = torch.nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = torch.nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            torch.nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            torch.nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        
        if not self.use_add_merge:
            if self.use_rad_l_parametrization:
                self.edge_channels_list.append((self.num_in_channels * (self.lmax + 1) * 2))
            else:
                num_rad_out_channels = 0
                for m in range(self.mmax + 1):
                    num_rad_out_channels = num_rad_out_channels + (self.lmax + 1 - m)
                num_rad_out_channels = num_rad_out_channels * self.num_in_channels
                num_rad_out_channels = num_rad_out_channels * 2     # source and target
                self.edge_channels_list.append(num_rad_out_channels)
            self.rad_func = RadialFunction(
                self.edge_channels_list, 
                lmax=self.lmax, 
                mmax=(self.lmax if self.use_rad_l_parametrization else self.mmax),
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                use_expand=True
            )
        elif self.use_add_merge:
            assert self.use_rad_l_parametrization
            self.edge_channels_list.append((self.num_in_channels * (self.lmax + 1) * 2))
            self.rad_func = RadialFunction(
                self.edge_channels_list, 
                lmax=self.lmax, 
                mmax=self.lmax,
                use_rad_l_parametrization=True,
                use_expand=True
            )
            self.rad_func.net[-1].weight.data.mul_((1.0 / math.sqrt(2.0)))

        check_activation_name(self.activation)
        if ( ('swiglu' in self.activation) or ('square' in self.activation) ):
            self.num_hidden_channels = self.num_hidden_channels * 2
            self.use_swiglu = True
        else:
            self.use_swiglu = False

        extra_m0_out_channels = self.num_heads * self.attn_alpha_channels
        if has_scalars(self.activation):
            self.split_m0_channels_list = [extra_m0_out_channels]   # for `torch.split()`
            if 'sep-merge_gates2_swiglu' in self.activation:
                temp = self.num_hidden_channels + (self.num_hidden_channels // 2)
                extra_m0_out_channels = extra_m0_out_channels + temp
                self.split_m0_channels_list.append(temp)
            elif 'sep' in self.activation:
                extra_m0_out_channels = extra_m0_out_channels + self.num_hidden_channels
                self.split_m0_channels_list.append(self.num_hidden_channels)
            elif 'gate' in self.activation:
                temp = self.lmax * self.num_hidden_channels
                extra_m0_out_channels = extra_m0_out_channels + temp
                self.split_m0_channels_list.append(temp)
            else:
                raise ValueError

        self.so2_linear_1 = SO2Linear(
            ((2 * self.num_in_channels) if not self.use_add_merge else self.num_in_channels),
            self.num_hidden_channels,
            self.lmax,
            self.mmax,
            extra_m0_out_channels=extra_m0_out_channels     # for attention weights and activation
        )
        
        # Graph attention
        self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels) if self.use_attn_renorm else torch.nn.Identity()
        self.alpha_act = torch.nn.SiLU() if alpha_drop != 0.0 else SmoothLeakyReLU()
        self.alpha_dropout = torch.nn.Dropout(alpha_drop) if alpha_drop != 0.0 else torch.nn.Identity()
        self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
        std = 1.0 / math.sqrt(self.attn_alpha_channels)
        torch.nn.init.uniform_(self.alpha_dot, -std, std)
        self.attn_softmax = GraphSoftmax(
            eps=self.eps,
            exp_dropout=attn_mask_rate, 
            softcap=self.softcap
        )
        self.attn_weights_dropout = torch.nn.Dropout(attn_weights_drop) if attn_weights_drop != 0.0 else torch.nn.Identity()
        
        # S2/gate activation
        self.act = get_activation(
            act_name=self.activation,
            lmax=self.lmax,
            mmax=self.mmax,
            grid_resolution_list=self.grid_resolution_list,
            use_m_primary=True
        )
        # Value dropout
        if value_drop != 0.0:
            add_dropout(self.act, value_drop)

        self.so2_linear_2 = SO2Linear(
            (self.num_hidden_channels if not self.use_swiglu else (self.num_hidden_channels // 2)),
            self.num_heads * self.attn_value_channels,
            self.lmax,
            self.mmax,
            extra_m0_out_channels=None
        )
        if '-merge' in self.activation:
            # Since we add two type-0 vectors in merge activation, we divide the correpsonding weights by sqrt(2)
            temp = self.so2_linear_2.num_in_channels
            self.so2_linear_2.fc_m0.weight.data[0:temp, :].mul_(1.0 / math.sqrt(2.0))
            
        self.proj = SO3Linear(self.num_heads * self.attn_value_channels, self.num_out_channels, lmax=self.lmax)


    def forward(
        self,
        x,
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None
    ):
        num_nodes = x.shape[0]

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            #source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            #target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_atomic_numbers)
            target_embedding = self.target_embedding(target_atomic_numbers)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        # Radial function
        x_edge_weight = self.rad_func(x_edge)

        # Merge source/target node features
        x = x.to(x_edge_weight.dtype)
        x_source = torch.index_select(x, index=edge_index[0], dim=0)
        x_target = torch.index_select(x, index=edge_index[1], dim=0)
        if not self.use_add_merge:
            # Concat    
            x_message = torch.cat((x_source, x_target), dim=2)
            if self.use_rad_l_parametrization:
                x_message = x_message * x_edge_weight
                x_message = self.so3_rotation.rotate(x_message)
            else:
                x_message = self.so3_rotation.rotate(x_message)
                x_message = x_message * x_edge_weight
        elif self.use_add_merge:
            # Add
            x_edge_weight_source = x_edge_weight.narrow(2, 0, self.num_in_channels)
            x_edge_weight_target = x_edge_weight.narrow(2, self.num_in_channels, self.num_in_channels)
            x_source = x_source * x_edge_weight_source
            x_target = x_target * x_edge_weight_target
            x_message = x_source + x_target
            x_message = self.so3_rotation.rotate(x_message)
        
        x_message, x_m0_extra = self.so2_linear_1(x_message)

        # S2/gate activation
        if has_scalars(self.activation):
            x_alpha  = x_m0_extra.narrow(1, 0, self.split_m0_channels_list[0])
            x_scalar = x_m0_extra.narrow(1, self.split_m0_channels_list[0], self.split_m0_channels_list[1])
        else:
            x_alpha  = x_m0_extra
            x_scalar = None
        act_input_dict = prepare_activation_forward_param(
            act_name=self.activation, 
            inputs=x_message, 
            scalars=x_scalar
        )
        x_message = self.act(**act_input_dict)

        x_message = self.so2_linear_2(x_message)

        # Graph attention
        x_alpha = x_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
        x_alpha = self.alpha_norm(x_alpha)
        x_alpha = self.alpha_act(x_alpha)
        x_alpha = self.alpha_dropout(x_alpha)
        alpha = torch.einsum('bik, ik -> bi', x_alpha, self.alpha_dot)
        #alpha = torch_geometric.utils.softmax(alpha, edge_index[1], num_nodes=num_nodes)
        alpha = self.attn_softmax(alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=edge_envelope_weight)
        if edge_envelope_weight is not None:
            alpha = alpha * edge_envelope_weight
        alpha = alpha.view(alpha.shape[0], 1, self.num_heads, 1)
        alpha = self.attn_weights_dropout(alpha)
        if torch.is_autocast_enabled():
            alpha = alpha.to(torch.float16)

        # Attention weights * non-linear messages
        attn = x_message
        attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message = attn

        # Rotate back the irreps
        x_message = self.so3_rotation.rotate_inv(x_message)
        
        # Compute the sum of the incoming neighboring messages for each target node
        x_message = reduce_edge(
            inputs=x_message,
            edge_index=edge_index[1], 
            output_shape=(num_nodes, x_message.shape[1], x_message.shape[2])
        )

        # Project
        outputs = self.proj(x_message)

        return outputs


class FeedForwardNetwork(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            num_hidden_channels (int):  Number of hidden channels
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid`

            activation (str):           Type of activation function
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
            
            dropout (float):            Dropout rate for the hidden features.
    """
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        grid_resolution_list,
        activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        dropout=0.0
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.grid_resolution_list = grid_resolution_list
        self.activation = activation
        self.use_grid_mlp = use_grid_mlp
        
        check_activation_name(self.activation)
        if self.use_grid_mlp:
            assert 's2' in self.activation

        self.so3_linear_1 = SO3Linear(self.num_in_channels, self.num_hidden_channels, lmax=self.lmax)

        if self.use_grid_mlp:
            # Scalar path
            if 'sep' in self.activation:
                if 'swiglu' in self.activation:
                    self.scalar_mlp = torch.nn.Sequential(
                        LinearSwiGLU(self.num_in_channels, self.num_hidden_channels),
                        (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity())
                    )
                elif 'square' in self.activation:
                    self.scalar_mlp = torch.nn.Sequential(
                        LinearSquare(self.num_in_channels, self.num_hidden_channels),
                        (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity())
                    )
                else:
                    self.scalar_mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.num_in_channels, self.num_hidden_channels),
                        torch.nn.SiLU(),
                        (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity())
                    )
            else:
                self.scalar_mlp = None
            
            self.so3_grid = SO3Grid(
                lmax=self.lmax,
                mmax=self.lmax,
                resolution_list=self.grid_resolution_list,
                use_m_primary=False
            )

            # Grid MLP
            if 'swiglu' in self.activation:
                if 'gates2' not in self.activation:
                    self.grid_mlp = torch.nn.Sequential(
                        LinearSwiGLU(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                        (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()),
                        torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False)
                    )
                elif 'gates2' in self.activation:
                    self.grid_mlp = GatedSwiGLUGridMLP(self.num_in_channels, self.num_hidden_channels, dropout)
            elif 'square' in self.activation:
                self.grid_mlp = torch.nn.Sequential(
                    LinearSquare(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()),
                    torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False)
                )
            else:
                self.grid_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    torch.nn.SiLU(),
                    (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()),
                    torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    torch.nn.SiLU(),
                    (torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()),
                    torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False)
                )
        else:   # No grid MLP
            assert self.activation in ['gate', 's2', 'sep_s2', 'sep-merge_gates2_swiglu']
            if self.activation == 'gate':
                self.gating_linear = torch.nn.Linear(self.num_in_channels, (self.lmax * self.num_hidden_channels))
            elif self.activation == 'sep_s2':
                self.gating_linear = torch.nn.Linear(self.num_in_channels, self.num_hidden_channels)
            elif self.activation == 'sep-merge_gates2_swiglu':
                del self.so3_linear_1
                self.so3_linear_1 = SO3Linear(self.num_in_channels, 2 * self.num_hidden_channels, lmax=self.lmax)
                self.gating_linear = torch.nn.Linear(
                    self.num_in_channels,
                    (2 * self.num_hidden_channels + self.num_hidden_channels)
                )
            else:
                self.gating_linear = None
            self.act = get_activation(
                act_name=self.activation,
                lmax=self.lmax,
                mmax=self.lmax,
                grid_resolution_list=self.grid_resolution_list,
                use_m_primary=False
            )

            if dropout != 0.0:
                add_dropout(self.act, dropout)

        self.so3_linear_2 = SO3Linear(self.num_hidden_channels, self.num_out_channels, lmax=self.lmax)
        if '-merge' in self.activation:
            self.so3_linear_2.weight.data[0, :, :].mul_(1.0 / math.sqrt(2.0))


    def forward(self, inputs):
        # Scalar path
        gating_scalars = None
        if self.use_grid_mlp:
            if self.scalar_mlp is not None:
                gating_scalars = self.scalar_mlp(inputs.narrow(1, 0, 1))
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(inputs.narrow(1, 0, 1))

        # First SO(3) linear layer
        outputs = self.so3_linear_1(inputs)

        if self.use_grid_mlp:
            # Grid MLP
            output_grid = self.so3_grid.to_grid(outputs)
            if 'gates2' not in self.activation:
                if '_mem' not in self.activation:
                    output_grid = self.grid_mlp(output_grid)
                else:
                    output_grid = torch.utils.checkpoint.checkpoint(
                        self.grid_mlp,
                        output_grid
                    )
            elif 'gates2' in self.activation:
                if '_mem' not in self.activation:
                    output_grid = self.grid_mlp(output_grid, inputs.narrow(1, 0, 1))
                else:
                    output_grid = torch.utils.checkpoint.checkpoint(
                        self.grid_mlp,
                        output_grid,
                        inputs.narrow(1, 0, 1)
                    )
            outputs = self.so3_grid.from_grid(output_grid)
            
            if self.scalar_mlp is not None:
                if '-merge' not in self.activation:
                    # Concat the scalar MLP outputs and the (L > 0) part of the grid MLP outputs
                    outputs = torch.cat(
                        (gating_scalars, outputs.narrow(1, 1, outputs.shape[1] - 1)),
                        dim=1
                    )
                else:
                    # Add the scalar MLP outputs to the grid MLP outputs
                    outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + gating_scalars
        else:
            act_input_dict = prepare_activation_forward_param(
                act_name=self.activation, 
                inputs=outputs, 
                scalars=gating_scalars
            )
            outputs = self.act(**act_input_dict)

        # Second SO(3) linear layer
        outputs = self.so3_linear_2(outputs)

        return outputs


class GatedSwiGLUGridMLP(torch.nn.Module):
    def __init__(self, num_in_channels, num_hidden_channels, dropout):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.dropout = dropout
        
        self.gating_linear = torch.nn.Linear(self.num_in_channels, self.num_hidden_channels)
        self.gate_act = torch.nn.Sigmoid()
        self.grid_linear_1 = torch.nn.Linear(self.num_hidden_channels, 2 * self.num_hidden_channels, bias=False)
        self.grid_drop = (torch.nn.Dropout(self.dropout) if self.dropout > 0.0 else torch.nn.Identity())
        self.grid_linear_2 = torch.nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False)


    def forward(self, input_grid, scalars):
        gate_scalars = self.gating_linear(scalars)
        gate_scalars = self.gate_act(gate_scalars)
        output_grid = self.grid_linear_1(input_grid)
        output_grid_1, output_grid_2 = torch.chunk(output_grid, chunks=2, dim=-1)
        output_grid_1 = output_grid_1 * gate_scalars
        output_grid = output_grid_1 * output_grid_2
        output_grid = self.grid_drop(output_grid)
        output_grid = self.grid_linear_2(output_grid)
        return output_grid


class TransBlockV3(torch.nn.Module):
    """
        Args:
            num_in_channels (int):      Number of input channels
            attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
            num_heads (int):            Number of attention heads
            attn_alpha_head (int):      Number of channels for alpha vector in each attention head
            attn_value_head (int):      Number of channels for value vector in each attention head
            ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
            num_out_channels (int):     Number of output channels

            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)

            so3_rotation (SO3Rotation): Class to calculate Wigner-D matrices and rotate embeddings
            attn_grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in attention
            ffn_grid_resolution_list (list:int):      
                                        Grid resolution list in class `SO3Grid` in feedforward network
            
            max_num_elements (int):     Maximum number of atomic numbers
            edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                            The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
            use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
            
            attn_activation (str):      Type of activation function for equivariant graph attention
            use_attn_renorm (bool):     Whether to re-normalize attention weights
            use_add_merge (bool):       Default: False
                                        If True, use addition to merge the source/target node features instead of concat, 
                                        which can save 2x compute when rotating with Wigner-D matrices.
            use_rad_l_parametrization (bool):
                                        Default: True
                                        If True, all the m components within the same type-L vector will share the same
                                        weight from the radial function.
            softcap (float):            Default: None
                                        If not None, use soft capping to limit the range of attention logits to
                                        [- `softcap`, + `softcap`].
            attn_eps (float):           Default: 1e-16
                                        Epsilon value used in the softmax operation of attention

            ffn_activation (str):       Type of activation function for feedforward network
            use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
            
            norm_type (str):            Type of normalization layer

            alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
            attn_mask_rate (float):     Mask rate for neighbors considered in attention
            attn_weights_drop (float):  Dropout rate for attention weights
            value_drop (float):         Dropout rate for the hidden features in non-linear value vectors.
            drop_path_rate (float):     Drop path rate
            proj_drop (float):          Dropout rate for outputs of attention and FFN
            ffn_drop (float):           Dropout rate for the hidden features in FFN
    """
    def __init__(
        self,
        num_in_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        attn_grid_resolution_list,
        ffn_grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        attn_activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        ffn_activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        norm_type='sep_layer_norm',
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0
    ):
        super().__init__()

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.ga = EquivariantGraphAttention(
            num_in_channels=num_in_channels,
            num_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            num_out_channels=num_in_channels,
            lmax=lmax,
            mmax=mmax,
            so3_rotation=so3_rotation,
            grid_resolution_list=attn_grid_resolution_list,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            activation=attn_activation,
            use_attn_renorm=use_attn_renorm,
            use_add_merge=use_add_merge,
            use_rad_l_parametrization=use_rad_l_parametrization,
            softcap=softcap,
            eps=attn_eps,
            alpha_drop=alpha_drop,
            attn_mask_rate=attn_mask_rate,
            attn_weights_drop=attn_weights_drop,
            value_drop=value_drop
        )

        if 'rms_norm' in norm_type:
            if self.ga.alpha_norm is not None:
                del self.ga.alpha_norm
                self.ga.alpha_norm = RMSNorm(attn_alpha_channels)

        self.drop_path = GraphDropPath(drop_path_rate) #if drop_path_rate > 0.0 else None
        self.proj_drop = EquivariantDropout(lmax=lmax, mmax=lmax, drop_prob=proj_drop) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.ffn = FeedForwardNetwork(
            num_in_channels=num_in_channels,
            num_hidden_channels=ffn_hidden_channels,
            num_out_channels=num_out_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution_list=ffn_grid_resolution_list,
            activation=ffn_activation,
            use_grid_mlp=use_grid_mlp,
            dropout=ffn_drop,
        )

        if num_in_channels != num_out_channels:
            self.ffn_shortcut = SO3Linear(num_in_channels, num_out_channels, lmax=lmax)
        else:
            self.ffn_shortcut = None


    def forward(
        self,
        x,                          # torch.Tensor
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight=None,  # for smooth cutoff
        batch=None                  # for GraphDropPath
    ):
        outputs = x
        x_res = x

        outputs = self.norm_1(outputs)
        outputs = self.ga(
            outputs,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight
        )

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        outputs = outputs + x_res

        x_res = outputs
        outputs = self.norm_2(outputs)
        outputs = self.ffn(outputs)

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        if self.ffn_shortcut is not None:
            x_res = self.ffn_shortcut(x_res)

        outputs = outputs + x_res

        return outputs