import torch
import copy
from .so3 import SO3Grid


def check_activation_name(act_name):
    assert act_name in [
        'gate',
        's2',
        'sep_s2',
        's2_swiglu',
        's2_swiglu_mem',
        'sep_s2_swiglu',
        'sep-merge_s2_swiglu',
        'sep_s2_swiglu_mem',
        'sep-merge_s2_swiglu_mem',
        'sep_s2_square',
        'sep-merge_gates2_swiglu',
        'sep-merge_gates2_swiglu_mem'
    ]
    return 


def get_activation(act_name, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
    """
        use_m_primary (bool):   Default: False
                                Whether to change the layout of m components.
                                If `False`, the layout of m is (0), (-1, 0, +1), (-2, -1, 0, +1, +2), ...
                                If `True`, the layout of m is (0, 0, ...), (1, 1, ...), ...
                                The second one is used in SO(2) linear operations to avoid redundant 
                                matrix multiplications.
    """
    check_activation_name(act_name)
    if act_name == 'gate':
        act_class = GateActivation
    elif act_name == 's2':
        act_class = S2Activation
    elif act_name == 'sep_s2':
        act_class = SeparableS2Activation
    elif act_name == 's2_swiglu':
        act_class = S2Activation_SwiGLU
    elif act_name == 's2_swiglu_mem':
        act_class = S2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU
    elif act_name == 'sep-merge_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU_Merge
    elif act_name == 'sep_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep-merge_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_Merge_MemoryEfficient
    elif act_name == 'sep_s2_square':
        act_class = SeparableS2Activation_Square
    elif act_name == 'sep-merge_gates2_swiglu':
        act_class = SeparableGateS2Activation_SwiGLU_Merge
    elif act_name == 'sep-merge_gates2_swiglu_mem':
        act_class = SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient
    args = {
        'lmax': lmax, 
        'mmax': mmax,
        'use_m_primary': use_m_primary
    }
    if act_name != 'gate':
        args['grid_resolution_list'] = grid_resolution_list
    return act_class(**args)


def has_scalars(act_name):
    if act_name not in ['s2', 's2_swiglu', 's2_swiglu_mem']:
        return True
    return False


def add_dropout(act, drop):
    """
        1.  Add extra dropout to original activation functions
    """
    attribute_name_list = ['act', 'gate_act', 'scalar_act']
    for attr_name in attribute_name_list:
        if attr_name == 'gate_act' and isinstance(act, SeparableGateS2Activation_SwiGLU_Merge):
            continue # For `SeparableGateS2Activation_SwiGLU_Merge`, dropout is from `grid_drop`
        if hasattr(act, attr_name):
            temp = copy.deepcopy(getattr(act, attr_name))
            update_act_list = [
                temp,
                torch.nn.Dropout(drop)
            ]
            delattr(act, attr_name)
            setattr(act, attr_name, torch.nn.Sequential(*update_act_list))
    if hasattr(act, 'grid_drop'):
        delattr(act, 'grid_drop')
        setattr(act, 'grid_drop', torch.nn.Dropout(drop))
    return


def prepare_activation_forward_param(act_name, inputs, scalars):
    output_dict = {
        'inputs': inputs
    }
    if has_scalars(act_name):
        output_dict['scalars'] = scalars
    return output_dict


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope


    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, use_m_primary=False):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        if not self.use_m_primary:
            expand_index = torch.zeros([num_components]).long()
            start_idx = 0
            for l in range(1, self.lmax + 1):
                length = min((2 * l + 1), (2 * self.mmax + 1))
                expand_index[start_idx : (start_idx + length)] = (l - 1)
                start_idx = start_idx + length
        elif self.use_m_primary:
            expand_index = []
            for m in range(self.mmax + 1):
                if m == 0:
                    l_index = torch.arange(self.lmax)       # We do not have L = 0
                else:
                    l_index = torch.arange((m - 1), self.lmax)
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)            # +- m
            expand_index = torch.cat(expand_index, dim=0)
            expand_index = expand_index.long()
        self.register_buffer('expand_index', expand_index)

        self.scalar_act = torch.nn.SiLU()
        self.gate_act   = torch.nn.Sigmoid()


    def forward(self, inputs, scalars):
        '''
            `inputs`: shape  [N, (lmax + 1) ** 2, num_channels]
            `scalars`: shape [N, lmax * num_channels]
        '''
        gate_scalars = self.gate_act(scalars)
        gate_scalars = gate_scalars.reshape(gate_scalars.shape[0], self.lmax, -1)
        gate_scalars = torch.index_select(gate_scalars, dim=1, index=self.expand_index)

        # L = 0
        input_scalars = inputs.narrow(1, 0, 1)
        input_scalars = self.scalar_act(input_scalars)
        # L > 0
        input_vectors = inputs.narrow(1, 1, inputs.shape[1] - 1)
        input_vectors = input_vectors * gate_scalars

        output_tensors = torch.cat((input_scalars, input_vectors), dim=1)

        return output_tensors


    def extra_repr(self):
        return 'lmax={}, mmax={}, use_m_primary={}'.format(self.lmax, self.mmax, self.use_m_primary)


class S2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act = torch.nn.SiLU()
        

    def forward(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        outputs = self.so3_grid.from_grid(x_grid)
        return outputs


class SeparableS2Activation(S2Activation):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


def swiglu_torch(gate, up_states):
    gate = torch.nn.functional.silu(gate)
    outputs = gate * up_states
    return outputs


class SwiGLU(torch.nn.Module):
    '''
        1.  The module only contains the activation.
        2.  The number of output channels is the half of that of input channels.
    '''
    def __init__(self, backend='torch'):
        super(SwiGLU, self).__init__()
        assert backend in ['torch']
        self.backend = backend
        self.func = swiglu_torch


    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        outputs = self.func(x_1, x_2)
        return outputs


    def extra_repr(self):
        return 'backend={}'.format(self.backend)
    

class LinearSwiGLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, backend='torch'):
        super(LinearSwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = SwiGLU(backend)

    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.act(outputs)
        return outputs


class S2Activation_SwiGLU(S2Activation):
    '''
        1.  Assume we only have one resolution.
        2.  Use SwiGLU after projecting to grids.
    '''
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)
        del self.act
        self.act = SwiGLU(backend)


class S2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU):
    '''
        1.  Assume we only have one resolution.
        2.  Use SwiGLU after projecting to grids.
        3.  We use gradient checkpointing for this activation function.
    '''
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)
    

    def kernel(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        outputs = self.so3_grid.from_grid(inputs)
        return outputs
    
    
    def forward(self, inputs):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel, 
            inputs,
            use_reentrant=False
        )
        return outputs
    

class SeparableS2Activation_SwiGLU(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs
    

class SeparableS2Activation_SwiGLU_Merge(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)
        

    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = output_vectors #.clone()
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs
    

class SeparableS2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1
        )
        return outputs


class SeparableS2Activation_SwiGLU_Merge_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    
    def kernel(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        outputs = output_vectors #.clone()
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


    def forward(self, inputs, scalars):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel, 
            inputs,
            scalars,
            use_reentrant=False
        )
        return outputs


class Square(torch.nn.Module):
    '''
        The number of output channels is the half of that of input channels.
    '''
    def __init__(self):
        super(Square, self).__init__()


    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        outputs = x_1 * x_2
        return outputs


class LinearSquare(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LinearSquare, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = Square()

    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.act(outputs)
        return outputs


class SeparableS2Activation_Square(torch.nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act  = Square()
        
    
    def forward(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        outputs = torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)), 
            dim=1
        )
        return outputs


class SeparableGateS2Activation_SwiGLU_Merge(GateActivation):
    """
        1.  'Separable' means that we have two paths (type-0 vectors and type-L vectors (L >= 0)).
        2.  We divide type-0 vectors into two parts for 3. and 4.
        3.  We apply SwiGLU to the first part of type-0 path ('SwiGLU').
        4.  We apply Sigmoid to the second part of type-0 path ('Gate'), which is to be used in 5.
        5.  For the type-L path, we project to S2 grid signals ('S2') and divide into two parts of 
            equal sizes.
            We use the nonlinear weights in 4. to gate the first part of the S2 grid signals 
            ('GateS2Activation').
            Then, we perform elementwise multiplication, which is equivalent to the self tensor product for 
            many-body interactions. 
            After elementwise multiplication, we optionally apply dropout, which enables training in a
            non-equivariant manner.
            Finally, we project S2 grid signals back to equivariant features.
            Note that this is similar to SwiGLU (but without SiLU activation) in 3.
        6.  We merge the two paths mentioned in 1. by adding the type-0 parts ('Merge').
    """
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, use_m_primary)
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        del self.scalar_act
        self.scalar_act = SwiGLU(backend)
        del self.expand_index
        self.grid_drop = torch.nn.Identity()


    def forward(self, inputs, scalars):
        """
            `inputs`: shape  [N, (lmax + 1) ** 2, 2 * num_channels]
            `scalars`: shape [N, 2 * num_channels + num_channels] or
                             [N, 1, 2 * num_channels + num_channels]
        """
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        # 3. SwiGLU to type-0 path
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)    # [N, 1, num_channels]
        # 4. Sigmoid for gating
        gate_scalars = self.gate_act(gate_scalars)          # [N, 1, num_channels]
        # 5. Project to S2 grid signals, perform nonlinear gating, perform elementwise multiplication, 
        #    optionally perform dropout, and project back
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        #x_grid_1 = x_grid_1 * gate_scalars
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        # 6. Merge
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


class SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient(SeparableGateS2Activation_SwiGLU_Merge):
    """
        1.  Add gradient checkpointing to `SeparableGateS2Activation_SwiGLU_Merge`
    """
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)


    def kernel(self, inputs, scalars):
        """
            `inputs`: shape  [N, (lmax + 1) ** 2, 2 * num_channels]
            `scalars`: shape [N, 2 * num_channels + num_channels] or
                             [N, 1, 2 * num_channels + num_channels]
        """
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        # 3. SwiGLU to type-0 path
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)    # [N, 1, num_channels]
        # 4. Sigmoid for gating
        gate_scalars = self.gate_act(gate_scalars)          # [N, 1, num_channels]
        # 5. Project to S2 grid signals, perform nonlinear gating, perform elementwise multiplication, 
        #    optionally perform dropout, and project back
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        #x_grid_1 = x_grid_1 * gate_scalars
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        # 6. Merge
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


    def forward(self, inputs, scalars):
        outputs = torch.utils.checkpoint.checkpoint(
            self.kernel, 
            inputs,
            scalars,
            use_reentrant=False
        )
        return outputs