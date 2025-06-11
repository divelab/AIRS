import copy
from math import pi, log
from einops import rearrange, repeat
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm

from pdearena.utils.activations import SinusoidalEmbedding, ACTIVATION_REGISTRY
from pdearena.configs.config import Config


class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result

        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)


class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                                         1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FreqLinear(nn.Module):
    def __init__(self, in_channel, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channel + 4 * modes)
        weights = nn.Parameter(scale * torch.randn(4 * modes, in_channel, dtype=torch.float32))
        bias = nn.Parameter(torch.zeros(4 * modes, dtype=torch.float32))
        self.proj = nn.Linear(in_channel, 4 * modes)
        self.proj.weight = weights
        self.proj.bias = bias

    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x)
        h = h.reshape(B, self.modes, 2, 2)
        h = torch.view_as_complex(h).unsqueeze(1)
        hx, hy = h.chunk(2, dim=-1)
        hy = hy.transpose(-1, -2)
        return hx, hy
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, activation, n_modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, dropout, mode):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.n_modes = n_modes
        self.mode = mode
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 2 * out_dim),
            self.activation,
            FreqLinear(in_channel=2 * out_dim, modes=n_modes),
        )
        self.dt_emb = nn.Linear(1, out_dim)

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x, emb):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        dt, emb = emb
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x, emb)
        b = self.backcast_ff(x)
        dt_emb = self.dt_emb(dt)
        while len(dt_emb.shape) < len(b.shape):
            dt_emb = dt_emb.unsqueeze(1)
        return dt_emb * b, None

    def forward_fourier(self, x, emb):

        embx, emby = self.cond_emb(emb)
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        flip = M != N

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1) if flip else x_fty.new_zeros(B, I, N, M // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            x_fty = emby * x_fty[:, :, :, :self.n_modes]
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty,
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N) if flip else x_ftx.new_zeros(B, I, N // 2 + 1, M)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            x_ftx = embx * x_ftx[:, :, :self.n_modes, :]
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx,
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FNOFactorized2DBlock(nn.Module):
    def __init__(self, modes, width, input_dim=12, output_dim=12, cond_dim=1, activation='gelu',
                 dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight: bool = False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, mode='full'):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.drop = nn.Dropout(in_dropout)
        self.n_layers = n_layers

        self.forecast_ff = self.backcast_ff = None
        if share_fork:
            self.backcast_ff = FeedForward(
                width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param, gain=gain)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       cond_dim=cond_dim,
                                                       activation=activation,
                                                       n_modes=modes,
                                                       forecast_ff=self.forecast_ff,
                                                       backcast_ff=self.backcast_ff,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       dropout=dropout,
                                                       mode=mode))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def forward(self, x, emb):
        # x.shape == [n_batches, *dim_sizes, input_size]
        x = self.in_proj(x)
        x = self.drop(x)
        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, f = layer(x, emb)
            x = x + b
        return self.out(b)


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10 ** 6, std_epsilon=1e-8):
        super().__init__()
        self.register_buffer('max_accumulations', torch.tensor(max_accumulations))
        self.register_buffer('count', torch.tensor(0.0))
        self.register_buffer('n_accumulations', torch.tensor(0.0))
        self.register_buffer('sum', torch.full(size, 0.0))
        self.register_buffer('sum_squared', torch.full(size, 0.0))
        self.register_buffer('one', torch.tensor(1.0))
        self.register_buffer('std_epsilon', torch.full(size, std_epsilon))
        self.dim_sizes = None

    def _accumulate(self, x):
        x_count = x.shape[0]
        x_sum = x.sum(dim=0)
        x_sum_squared = (x ** 2).sum(dim=0)

        self.sum += x_sum
        self.sum_squared += x_sum_squared
        self.count += x_count
        self.n_accumulations += 1

    def _pool_dims(self, x):
        _, *dim_sizes, _ = x.shape
        self.dim_sizes = dim_sizes
        if self.dim_sizes:
            x = rearrange(x, 'b ... h -> (b ...) h')

        return x

    def _unpool_dims(self, x):
        if len(self.dim_sizes) == 1:
            x = rearrange(x, '(b m) h -> b m h', m=self.dim_sizes[0])
        elif len(self.dim_sizes) == 2:
            m, n = self.dim_sizes
            x = rearrange(x, '(b m n) h -> b m n h', m=m, n=n)
        return x

    def forward(self, x):
        x = self._pool_dims(x)
        # x.shape == [batch_size, latent_dim]

        if self.training and self.n_accumulations < self.max_accumulations:
            self._accumulate(x)

        x = (x - self.mean) / self.std
        x = self._unpool_dims(x)

        return x

    def inverse(self, x, channel=None):
        x = self._pool_dims(x)

        if channel is None:
            x = x * self.std + self.mean
        else:
            x = x * self.std[channel] + self.mean[channel]

        x = self._unpool_dims(x)

        return x

    @property
    def mean(self):
        safe_count = max(self.count, self.one)
        return self.sum / safe_count

    @property
    def std(self):
        safe_count = max(self.count, self.one)
        std = torch.sqrt(self.sum_squared / safe_count - self.mean ** 2)
        return torch.maximum(std, self.std_epsilon)


def fourier_encode(x, max_freq, num_bands=4, base=2):
    # Our data spans over a distance of 2. If there are 100 data points,
    # the sampling frequency (i.e. mu) is 100 / 2 = 50 Hz.
    # The Nyquist frequency is 25 Hz.
    x = x.unsqueeze(-1)
    # x.shape == [*dim_sizes, n_dims, 1]
    device, dtype, orig_x = x.device, x.dtype, x

    # max_freq is mu in the paper.
    # Create a range between (2^0 == 1) and (2^L == mu/2)
    scales = torch.logspace(0., log(max_freq / 2) / log(base),
                            num_bands, base=base, device=device, dtype=dtype)

    # Add leading dimensions
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    # scales.shape == [1, 1, 1, n_bands] for 2D images

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    # x.shape == [*dim_sizes, n_dims, n_bands * 2]

    # Interestingly enough, we also append the raw position
    x = torch.cat((x, orig_x), dim=-1)
    # x.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]
    return x


class FFNO(nn.Module):
    def __init__(self,
                 args: Config,
                 modes, width, dropout=0.0, in_dropout=0.0,
                 n_layers=4, share_weight=False,
                 share_fork=False, factor=2,
                 ff_weight_norm=False, n_ff_layers=2,
                 gain=1, layer_norm=False, mode='full',
                 num_freq_bands: int = 8,
                 freq_base: int = 2,
                 low: float = 0,
                 high: float = 1,
                 use_position: bool = True,
                 max_accumulations: float = 1000,
                 should_normalize: bool = False,
                 use_fourier_position: bool = False,
                 noise_std: float = 0.0,
                 sinusoidal_embedding: bool = True,
                 sine_spacing: float = 1e-3,
                 num_param_conditioning: int = 2,
                 diffusion_time_conditioning: bool = False,
                 use_dt: bool = True,
                 activation: str = 'gelu',
                 ):
        super().__init__()
        self.n_input_fields = args.n_input_fields
        self.n_output_fields = args.n_output_fields
        self.time_history = args.time_history
        self.time_future = args.time_future
        self.num_param_conditioning = num_param_conditioning
        self.diffusion_time_conditioning = diffusion_time_conditioning
        self.use_dt = use_dt

        if sinusoidal_embedding:
            embed_dim = width
            get_embed_fn = lambda: SinusoidalEmbedding(
                n_channels=width,
                spacing=sine_spacing
            )
        else:
            embed_dim = 1
            get_embed_fn = lambda: nn.Identity()

        if use_dt:
            num_embs = 1
            self.dt_embed = get_embed_fn()
        else:
            num_embs = 0

        if self.diffusion_time_conditioning:
            num_embs += 1
            self.diffusion_time_embed = get_embed_fn()

        if self.num_param_conditioning > 0:
            num_embs += self.num_param_conditioning
            self.pde_embed = nn.ModuleList()
            for _ in range(self.num_param_conditioning):
                self.pde_embed.append(get_embed_fn())

        cond_embed_dim = embed_dim * num_embs

        self.conv = FNOFactorized2DBlock(modes, width,
                                         input_dim=self.n_input_fields * self.time_history + (2 if use_position else 0),
                                         output_dim=args.n_output_fields * self.time_future,
                                         cond_dim=cond_embed_dim,
                                         activation=activation,
                                         dropout=dropout, in_dropout=in_dropout,
                                         n_layers=n_layers, share_weight=share_weight,
                                         share_fork=share_fork, factor=factor,
                                         ff_weight_norm=ff_weight_norm, n_ff_layers=n_ff_layers,
                                         gain=gain, layer_norm=layer_norm, mode=mode)
        self.use_fourier_position = use_fourier_position
        self.use_position = use_position
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base
        self.low = low
        self.high = high
        self.lr = None
        self.should_normalize = should_normalize
        self.normalizer = Normalizer([self.n_input_fields], max_accumulations)
        self.register_buffer('_float', torch.FloatTensor([0.1]))
        self.noise_std = noise_std

    def encode_positions(self, dim_sizes, low=-1, high=1, fourier=True):
        # dim_sizes is a list of dimensions in all positional/time dimensions
        # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]

        # A way to interpret `pos` is that we could append `pos` directly
        # to the raw inputs to attach the positional info to the raw features.
        def generate_grid(size):
            return torch.linspace(low, high, steps=size,
                                  device=self._float.device)

        grid_list = list(map(generate_grid, dim_sizes))
        pos = torch.stack(torch.meshgrid(*grid_list, indexing='ij'), dim=-1)
        # pos.shape == [*dim_sizes, n_dims]

        if not fourier:
            return pos

        # To get the fourier encodings, we will go one step further
        fourier_feats = fourier_encode(
            pos, self.k_max, self.num_freq_bands, base=self.freq_base)
        # fourier_feats.shape == [*dim_sizes, n_dims, n_bands * 2 + 1]

        fourier_feats = rearrange(fourier_feats, '... n d -> ... (n d)')
        # fourier_feats.shape == [*dim_sizes, pos_size]

        return fourier_feats

    def _build_features(self, x):
        orig_shape = x.shape
        if self.should_normalize:
            x = x.flatten(0, 1).permute(0, 2, 3, 1)
            x = self.normalizer(x)
            x = x.unflatten(0, [orig_shape[0], -1]).permute(0, 2, 3, 1, 4).flatten(3)
        else:
            x = x.reshape(orig_shape[0], -1, *orig_shape[3:]).permute(0, 2, 3, 1)
        B, *dim_sizes, C = x.shape

        if self.use_position:
            pos_feats = self.encode_positions(
                dim_sizes, self.low, self.high, self.use_fourier_position)
            # pos_feats.shape == [*dim_sizes, pos_size]

            # if self.should_normalize:
            pos_feats_stats = {"mean": pos_feats.flatten(0, 1).mean(dim=0).view(1, 1, -1),
                               "sd": pos_feats.flatten(0, 1).std(dim=0).view(1, 1, -1)}
            pos_feats = (pos_feats - pos_feats_stats["mean"]) / pos_feats_stats["sd"]

            pos_feats = repeat(pos_feats, '... -> b ...', b=B)
            # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

            x = torch.cat([x, pos_feats], dim=-1)
            # xx.shape == [batch_size, *dim_sizes, 3]

        # if self.training:
        #     x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x

    def embed(
            self,
            dt: torch.Tensor,
            diff_time: torch.Tensor,
            z: torch.Tensor
    ):
        if self.use_dt:
            embs = [self.dt_embed(dt)]
        else:
            embs = []

        if self.diffusion_time_conditioning:
            assert diff_time is not None
            embs.append(self.diffusion_time_embed(diff_time + 1))
        else:
            assert diff_time is None

        if self.num_param_conditioning > 0:
            assert z is not None
            z_list = z.chunk(self.num_param_conditioning, dim=1)
            for i, z in enumerate(z_list):
                z = z.flatten()
                assert z.shape == dt.shape
                embs.append(self.pde_embed[i](z))

        embs = [emb.view(len(emb), -1) for emb in embs]
        emb = torch.cat(embs, dim=1)
        return emb

    def forward(self, batch):
        x = batch.x
        dt = batch.dt
        if self.num_param_conditioning > 0:
            z = batch.z
        else:
            z = None
        diff_time = batch.diff_time if hasattr(batch, 'diff_time') else None

        orig_shape = x.shape
        x = self._build_features(x)
        emb = self.embed(
            dt=dt,
            diff_time=diff_time,
            z=z
        )
        emb = dt.unsqueeze(1), emb

        im = self.conv(x, emb)

        if self.should_normalize:
            im = self.normalizer.inverse(im, channel=list(range(orig_shape[2])))

        return im.permute(0, 3, 1, 2).reshape(orig_shape[0], -1, *orig_shape[3:]).unsqueeze(1)
