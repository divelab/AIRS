import torch
from CEL.networks.models.meta_model import ModelClass
from torch.autograd import Function
from itertools import chain
from copy import deepcopy
from tqdm import tqdm
from torch.func import stack_module_state
from torch.func import functional_call
from einops import rearrange, reduce, repeat, pack, unpack
import math

class InvariantFuncEncMISlow(torch.nn.Module, ModelClass):
    def __init__(
            self,
            y_channels: int,
            W_channels: int,
            dfunc_hidden_channels: int,
            func_embedding_channels: int,
            hyper_hidden_channels: int,
            hyper_mlp_depth: int,
            depth_enc: int,
            discrim_hidden_channels: int,
            discrim_mlp_depth: int,
            transformer_dim_ffn: int,
            transformer_num_heads: int,
            transformer_num_layers: int,
            lambda_e_detach: bool,
            num_envs: int,
            always_rebuild_forecaster: bool = False,
            vectorize_func: bool = False,
            deep_copy: bool = False,
    ):
        super().__init__()
        self.dfunc_hidden_channels = dfunc_hidden_channels
        self.hyper_hidden_channels = hyper_hidden_channels
        self.func_embedding_channels = func_embedding_channels
        self.hyper_mlp_depth = hyper_mlp_depth
        self.depth_enc = depth_enc
        self.discrim_hidden_channels = discrim_hidden_channels
        self.discrim_mlp_depth = discrim_mlp_depth
        self.transformer_dim_ffn = transformer_dim_ffn
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.lambda_e_detach = lambda_e_detach
        self.W_channels = W_channels
        self.num_envs = num_envs
        self.always_rebuild_forecaster = always_rebuild_forecaster
        self.vectorize_func = vectorize_func
        self.deep_copy = deep_copy
        self.derivative_func = torch.nn.Sequential(
            *([
                torch.nn.Linear(y_channels, dfunc_hidden_channels // 4),
                torch.nn.LayerNorm(dfunc_hidden_channels // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(dfunc_hidden_channels // 4, dfunc_hidden_channels // 2),
                torch.nn.LayerNorm(dfunc_hidden_channels // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(dfunc_hidden_channels // 2, dfunc_hidden_channels),
            ] + list(chain(*[[
                torch.nn.LayerNorm(dfunc_hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(dfunc_hidden_channels, dfunc_hidden_channels)
            ] for _ in range(depth_enc)]))
              + [torch.nn.LayerNorm(dfunc_hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(dfunc_hidden_channels, y_channels)])
        )
        # the parameters of the derivative function are all calculated instead of being stored and learned.
        set_requires_grad(self.derivative_func, False)

        # define vectorized multi-environment derivative function

        base_model = deepcopy(self.derivative_func)
        base_model = base_model.to('meta')

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vectorized_deri_func = torch.vmap(fmodel, in_dims=(0, 0, 0), out_dims=0)

        # define hypernetwork
        self.multi_deri_funcs = None
        self.params, self.buffers = None, None
        self.num_funcs = 1

        self.hyper_network = HyperNetwork(self.derivative_func, y_channels, func_embedding_channels,
                                          transformer_dim_ffn, transformer_num_heads, transformer_num_layers,
                                          hyper_hidden_channels, hyper_mlp_depth, W_channels, num_envs)

        self.E_discriminator = make_mlp(func_embedding_channels, num_envs, discrim_hidden_channels, discrim_mlp_depth)

    def forward(self, y, input_length, **kwargs):
        if self.training:
            y = repeat(y, "B T y_channels -> (repeat B) T y_channels", repeat=2)
        rebuild_forecaster = self.always_rebuild_forecaster
        if self.vectorize_func:
            if self.multi_deri_funcs is None or self.num_funcs != y.shape[0] or kwargs.get('forecast_ode') is not None:
                rebuild_forecaster = True
                self.num_funcs = y.shape[0]
                self.multi_deri_funcs = [deepcopy(self.derivative_func) for _ in range(y.shape[0])]
                params, self.buffers = stack_module_state(self.multi_deri_funcs)
                self.params_dict = {p_name: list(torch.chunk(p, y.shape[0], dim=0)) for p_name, p in params.items()}
            elif self.deep_copy:
                self.multi_deri_funcs = deepcopy(self.multi_deri_funcs)
                set_requires_grad(self.multi_deri_funcs[0], False)
            if self.training or rebuild_forecaster:
                # --- update derivative functions: 1. when training for each batch, 2. when the number of functions changes
                # 3. when forecast_ode is set: begin a new batch of inference ---
                # --- self.params_dict: a set of predicted parameterized functions, such as polynomial and trigonometric functions ---
                # NOTE: for the same ODE integration process, we only need to update the self.params_dict once
                self.params_dict, disentangled_params = self.hyper_network(y[:, :input_length], self.params_dict, rebuild_forecaster, vectorize_func=self.vectorize_func, always_rebuild=self.always_rebuild_forecaster, **kwargs)

            # For max I(F_C;Y), max I(F;Y)
            dydt = self.vectorized_deri_func({p_name: torch.cat(p) for p_name, p in self.params_dict.items()}, self.buffers, y)
        else:
            self.num_funcs = y.shape[0]
            dmodels, disentangled_params = self.hyper_network(y[:, :input_length], None, True, vectorize_func=self.vectorize_func,
                                                                       deriv_func=self.derivative_func, **kwargs)
            dydt = []
            for i, dmodel in enumerate(dmodels):
                dydt.append(dmodel(y[i]))
            dydt = torch.stack(dydt, 0)


        return dydt

class HyperNetwork(torch.nn.Module):
    def __init__(self, main_func, y_channels, func_embedding_channels,
                 transformer_dim_ffn, transformer_num_heads, transformer_num_layers,
                 hidden_channels, hyper_mlp_depth, W_channels, num_envs):
        super().__init__()
        params, buffers = stack_module_state([main_func])
        self.num_param_main_func = sum(p.numel() for p in params.values())

        # params_main_func = serialize_parameters(main_func)
        # num_param_main_func = params_main_func.numel()
        # an GRU RNN encoder
        # self.encoder = torch.nn.GRU(y_channels, hidden_channels, num_layers=1, batch_first=True)
        # transformer encoder
        projection = torch.nn.Linear(y_channels, func_embedding_channels)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=func_embedding_channels,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_dim_ffn,
            batch_first=True,
        )
        self.encoder = torch.nn.Sequential(projection,
                                           PositionalEncoding(func_embedding_channels),
                                           torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers))

        # self.disentangle_projection = torch.nn.Linear(func_embedding_channels, 3 * func_embedding_channels)
        self.fc_projection = make_mlp(func_embedding_channels, func_embedding_channels, hidden_channels, hyper_mlp_depth)
        self.fe_projection = make_mlp(func_embedding_channels, func_embedding_channels, hidden_channels, hyper_mlp_depth)

        self.hyper_mlp = make_mlp(func_embedding_channels, self.num_param_main_func, hidden_channels, hyper_mlp_depth)

        self.FE_mlp = make_mlp(func_embedding_channels, func_embedding_channels, hidden_channels, hyper_mlp_depth)

        # self.param_buffer = torch.zeros(num_envs, num_param_main_func)
        # If the main model's parameter is generated by a hypernetwork, then the VC-dimension of the main model is restricted by the hypernetwork.
        # Therefore, the main model parameters should not be generated from a low-dimensional space.
        # Consider that the main model/func has VC-dimension d, then the function of the function (hypernetwork) should have VC-dimension at least d.
        # Generally, it should be much larger than d. Maybe d x num_envs? Let's try it first.
        # self.frame_main_func = deepcopy(main_func) # Deep copy here or use the keyword deep_copy in the deserialize_parameters function

    def forward(self, y, params_dict, rebuild_forecaster, vectorize_func=True, deriv_func=None, always_rebuild=False, **kwargs):
        if self.training:
            single_y = rearrange(y, "(chunk B) T y_channels -> chunk B T y_channels", chunk=2)[0]
        else:
            single_y = y
        zs = self.encoder(single_y) # y: B x T x y_channels
        final_z = zs[:, 0, :] # use the first token as the special summerization
        # zs_env_sp, final_z_env_sp = self.encoder(y) # y: B x T x y_channels
        f_c = self.fc_projection(final_z.squeeze(0))
        f_e = self.fe_projection(final_z.squeeze(0))
        # f_c, f_e_theta, f_e_alpha = rearrange(disentangled_funcs, "B (inv_et_ea d_z) -> inv_et_ea B d_z", inv_et_ea=3)
        # f_e = self.FE_mlp(f_e_theta + f_e_alpha)
        if not self.training:
            assert rebuild_forecaster
            assert kwargs.get('forecast_ode') is not None
            if kwargs.get('forecast_ode') == 'inv':
                f_combined = f_c
            elif kwargs.get('forecast_ode') == 'combine':
                f_combined = f_c + f_e
        else:
            f_combined = rearrange([f_c, f_c + f_e], "fc_f B d_z -> (fc_f B) d_z")
        params_func = self.hyper_mlp(f_combined) # d_z -> d_\theta
        # params_inv, params_env, params_individual = rearrange(params_func, "B (inv_env_ind params) -> inv_env_ind B params", inv_env_ind=3)
        if vectorize_func:
            if rebuild_forecaster:
                if always_rebuild:
                    self.param_buffer = params_func
                else:
                    self.param_buffer = torch.zeros(y.shape[0], self.num_param_main_func, device=y.device)
                for i in range(y.shape[0]):
                    connect_buffer_to_params(self.param_buffer[i], params_dict, pointer=0, func_idx=i)
            if not always_rebuild:
                self.param_buffer.detach_().copy_(params_func)
            return params_dict, {'f_c': f_c, 'f_e': f_e}  # 'f_e^theta': f_e_theta, 'f_e^alpha': f_e_alpha,
        else:
            dmodels = []
            for i in range(y.shape[0]):
                dmodels.append(deserialize_parameters(deriv_func, params_func[i], deep_copy=True))
            return dmodels, {'f_c': f_c, 'f_e': f_e}




class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape [batch_size, seq_len, hidden_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

def serialize_parameters(model, requires_grad=True):
    params = []
    for param in model.parameters():
        # Flatten and append to the list
        params.append(param.view(-1))
    # Concatenate all parameters into a single vector
    serialized_vector = torch.cat(params)
    return serialized_vector

def connect_buffer_to_params(buffer, params, pointer, func_idx):
    for param_name in params.keys():
        num_param = params[param_name][0].numel()  # Number of elements in the parameter
        # params[param_name] should not be sliced, instead, it should be used as a single pointer
        params[param_name][func_idx] = buffer[pointer:pointer + num_param].view(params[param_name][func_idx].shape)
        # params[param_name][func_idx].copy_(buffer[pointer:pointer + num_param].view(params[param_name][func_idx].shape))
        # # Delete the original parameter
        # delattr(module, name)

        # # Assign the custom tensor to the original parameter address
        # setattr(module, name, buffer[pointer:pointer + num_param].reshape(param.shape))

        pointer += num_param

    # for child_name, child_module in module.named_children():
    #     pointer = connect_buffer_to_params(buffer, child_module, pointer)
    # return pointer

def deserialize_parameters(model, param_vector, deep_copy=False):
    if deep_copy:
        model = deepcopy(model)
    set_requires_grad(model, False)
    pointer = 0  # Initialize pointer to track the position in the parameter vector
    for param in model.parameters():
        num_param = param.numel()  # Number of elements in the parameter
        # Extract the part of the parameter vector and reshape it
        param_flat = param_vector[pointer:pointer + num_param]
        # param.data = param_flat.view(param.size())#.clone()
        param.copy_(param_flat.view(param.size()))#.clone()
        pointer += num_param
    return model

def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf

class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None

def make_mlp(input_dim, output_dim, hidden_channels, num_layers):
    return torch.nn.Sequential(
        *([
            torch.nn.Linear(input_dim, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
        ] + list(chain(*[[
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            ] for _ in range(num_layers - 1)]))
          + [
            torch.nn.Linear(hidden_channels, output_dim),
        ])
    )