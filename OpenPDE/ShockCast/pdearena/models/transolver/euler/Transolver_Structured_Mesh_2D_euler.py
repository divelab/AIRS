import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from pdearena.models.transolver.model.Embedding import timestep_embedding
from pdearena.models.transolver.model.Physics_Attention import Physics_Attention_Structured_Mesh_2D
from pdearena.configs.config import CFDConfig
from pdearena.utils.activations import SinusoidalEmbedding
from torch_geometric.data import Batch

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            cond_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.dt_embed = nn.Linear(1, hidden_dim * 2)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.adaLN_modulation_attn = MLP(cond_dim, hidden_dim * mlp_ratio, hidden_dim * 2, n_layers=0, res=False, act=act)
        self.adaLN_modulation_mlp = MLP(cond_dim, hidden_dim * mlp_ratio, hidden_dim * 2, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, emb):
        dt, emb = emb
        dt_emb_attn, dt_emb_mlp = self.dt_embed(dt).unsqueeze(1).chunk(2, dim=2)
        shift_msa, scale_msa = self.adaLN_modulation_attn(emb).chunk(2, dim=1)
        fx = dt_emb_attn * self.Attn(modulate(self.ln_1(fx), shift_msa, scale_msa)) + fx
        shift_mlp, scale_mlp = self.adaLN_modulation_mlp(emb).chunk(2, dim=1)
        fx = dt_emb_mlp * self.mlp(modulate(self.ln_2(fx), shift_mlp, scale_mlp)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

# {'H': 64,
#  'Time_Input': False,
#  'W': 64,
#  'dropout': 0.0,
#  'fun_dim': 10,
#  'mlp_ratio': 1,
#  'n_head': 8,
#  'n_hidden': 256,
#  'n_layers': 8,
#  'out_dim': 1,
#  'ref': 8,
#  'slice_num': 32,
#  'space_dim': 2,
#  'unified_pos': 1}
class Model(nn.Module):
    def __init__(self,
                 args: CFDConfig,
                 num_param_conditioning,
                 sinusoidal_embedding=False,
                 sine_spacing=1,
                 use_dt=True,
                #  space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=True,
                 act='gelu',
                 mlp_ratio=1,
                #  fun_dim=1,
                #  out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=True,
                #  H=85,
                #  W=85,
                 ):
        super(Model, self).__init__()

        self.use_checkpoint = args.use_gradient_checkpoint
        self.n_input_fields = args.n_input_fields
        self.n_output_fields = args.n_output_fields
        self.time_history = args.time_history
        self.time_future = args.time_future
        self.num_param_conditioning = num_param_conditioning
        self.use_dt = use_dt

        n_hidden = n_hidden
        if sinusoidal_embedding:
            embed_dim = n_hidden
            get_embed_fn = lambda: SinusoidalEmbedding(
                n_channels=n_hidden, 
                spacing=sine_spacing
            )
        else:
            embed_dim = 1
            get_embed_fn = lambda: nn.Identity()


        num_embs = self.num_param_conditioning
        if self.num_param_conditioning > 0:
            self.pde_embed = nn.ModuleList()
            for _ in range(self.num_param_conditioning):
                self.pde_embed.append(get_embed_fn())

        cond_embed_dim = embed_dim * num_embs



        self.__name__ = 'Transolver_2D'
        self.H = H = args.ny
        self.W = W = args.nx
        space_dim = args.n_spatial_dim
        fun_dim = self.time_history * self.n_input_fields
        out_dim = self.time_future * self.n_output_fields
        self.ref = ref
        self.unified_pos = unified_pos
        assert unified_pos
        if self.unified_pos:
            pos = self.get_grid()
            self.pos: torch.Tensor
            self.register_buffer('pos', pos)
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      cond_dim=cond_embed_dim,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      H=H,
                                                      W=W,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        # self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        # torch.cdist(grid[:, :, :, None, None, :], grid_ref[:, None, None, :, :, :]).flatten(-3).equal(pos)
        return pos

    def forward(self, batch: Batch):
        x = batch.x
        dt = batch.dt
        if self.num_param_conditioning > 0:
            z = batch.z
        else:
            z = None
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        x = x.flatten(-2)  # collapse H,W
        x = x.permute(0, 2, 1)
        emb = self.embed(
            z=z
        )
        dt = dt.view(-1, 1)
        emb = dt, emb

        # if self.unified_pos:
        #     x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        # if fx is not None:
        #     fx = torch.cat((x, fx), -1)
        #     fx = self.preprocess(fx)
        # else:
        #     fx = self.preprocess(x)
        #     fx = fx + self.placeholder[None, None, :]
        pos = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        x = torch.cat((x, pos), -1)
        x = self.preprocess(x)

        if dt is not None:
            Time_emb = timestep_embedding(dt, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb

        for block in self.blocks:
            x = block(x, emb)

        x = x.permute(0, 2, 1).unflatten(dim=-1, sizes=[orig_shape[-2], orig_shape[-1]])
        x = x.reshape(
            orig_shape[0], -1, self.n_output_fields, *orig_shape[3:]
        )

        return x

    def embed(
        self,
        z: torch.Tensor
    ):
        if self.num_param_conditioning > 0:
            embs = []
            assert z is not None
            z_list = z.chunk(self.num_param_conditioning, dim=1)
            for i, z in enumerate(z_list):
                z = z.flatten()
                embs.append(self.pde_embed[i](z))
            embs = [emb.view(len(emb), -1) for emb in embs]
            emb = torch.cat(embs, dim=1)
            return emb
        else:
            return None
