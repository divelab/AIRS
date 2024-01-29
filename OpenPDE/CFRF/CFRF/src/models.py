import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
from networks import UNet4layers
import torch.nn.functional as F
from utils import create_mask
import math
from complexNonLinearities import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class complexMLP(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, mid_channels=None, complex=True):
        super(complexMLP, self).__init__()
        dtype = torch.cfloat if complex else torch.float
        self.mlp1 = nn.Linear(in_channels, mid_channels, dtype=dtype)
        self.mlp2 = nn.Linear(mid_channels, out_channels, dtype=dtype)
        self.act = complexReLU() if complex else nn.ReLU()

        self.outx = int(out_channels**(0.5))

    def forward(self, x):
        b, t, nx, ny = x.shape
        x = torch.flatten(x, start_dim=-2)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = x.view(b, t, self.outx, self.outx)
        return x
    
# transformer
class ModeFormer(nn.Module):
    def __init__(self, num_layers=3, input_dim=3, hidden_dim=16, output_dim=3, dropout_rate=0,
                 residual=False, upsample='nearest', scale_factor=1, freq_center_size=5, head=1, enc_modes=False,
                 encoding=None, injection=None, projection=None, layer_norm=True, activation="CReLU", 
                 postLN=False, Hermitian=False, ring=False, freeze_size=None, complex_pos_enc=False):
        super(ModeFormer, self).__init__()

        self.residual = residual
        self.Hermitian = Hermitian
        self.ring = ring

        self.offset = freq_center_size
        self.freeze_size = freeze_size

        self.projection = torch.nn.Identity()
        if projection:
            assert projection in ["MLP", "linear", "conv3x3"], f"unrecognized projection {projection}"
            if projection in ["linear"]:
                self.projection = nn.Conv2d(output_dim, output_dim, kernel_size=1)
            elif projection == "conv3x3":
                self.projection = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
            else:
                self.projection = nn.Sequential(nn.Conv2d(output_dim, hidden_dim, kernel_size=1),
                                                nn.ReLU(),
                                                nn.Conv2d(hidden_dim, output_dim, kernel_size=1))

        self.net = Transformer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, head=head,
                               num_layer=num_layers, enc_modes=enc_modes, encoding=encoding, injection=injection,
                               layer_norm=layer_norm, dropout_rate=dropout_rate, activation=activation, postLN=postLN, 
                               Hermitian=Hermitian, complex_pos_enc=complex_pos_enc)


        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode=upsample)

    def forward(self, x_up):

        # upsample input using interpolation
        # if x_up.shape[-1] < 256:
        #     x_up = self.upsampler(x_up)

        # convert to frequency domain
        b, t, nx, ny = x_up.shape
        k_max = nx // 2
        w_h = torch.fft.fft2(x_up, dim=[-2, -1])
        if self.Hermitian:
            w_h_shift = torch.fft.fftshift(w_h, dim=-2)
        else:
            w_h_shift = torch.fft.fftshift(w_h)

        if self.ring:
            w_h_shift_freeze_center = w_h_shift[..., (k_max-self.freeze_size):(k_max+self.freeze_size), (k_max-self.freeze_size):(k_max+self.freeze_size)].clone()

        # extract the center frequency modes
        # if self.Hermitian:
        #     center_w_h_shift = w_h_shift[..., (k_max-self.offset):(k_max+self.offset), (k_max-self.offset):(k_max)]
        # else:
        
        center_w_h_shift = w_h_shift[..., (k_max-self.offset):(k_max+self.offset), (k_max-self.offset):(k_max+self.offset)]
        
        if self.Hermitian:
            center_w_h_shift_ori = center_w_h_shift.clone()[..., :(self.offset+1)]
        else:
            center_w_h_shift_ori = center_w_h_shift.clone()

        res = self.net(center_w_h_shift)
        if self.residual:
            center_w_h_shift_updated = res + center_w_h_shift_ori
        else:
            center_w_h_shift_updated = res

        # put updated modes back, and mask all outer modes
        w_h_shift = torch.zeros_like(w_h_shift).to(device)
        if self.Hermitian:
            w_h_shift[..., (k_max-self.offset):(k_max+self.offset), (k_max-self.offset):(k_max+1)] = center_w_h_shift_updated
        else:
            w_h_shift[..., (k_max-self.offset):(k_max+self.offset), (k_max-self.offset):(k_max+self.offset)] = center_w_h_shift_updated

        if self.ring:
            w_h_shift[..., (k_max-self.freeze_size):(k_max+self.freeze_size), (k_max-self.freeze_size):(k_max+self.freeze_size)] = w_h_shift_freeze_center

        # w_h_shift_mask = create_mask(w_h_shift, offset=(k_max-self.offset)).to(w_h_shift.device)

        # convert to spatial domain
        if self.Hermitian:
            w_h = torch.fft.ifftshift(w_h_shift, dim=-2)
        else:
            w_h = torch.fft.ifftshift(w_h_shift)

        w = torch.fft.irfft2(w_h[..., :, :k_max + 1], dim=[-2, -1], s=(nx, ny))

        w = self.projection(w)

        return w

class complex_embedding(torch.nn.Module):
    def __init__(self, dmodel, injection="add", theta=True):
        super().__init__()
        assert injection in ["cat", "add"], "Injection should be one of ['cat', 'add']"
        self.add = injection == "add"
        self.dmodel = dmodel // 2 # x and y coordinate
        if not self.add:
            self.dmodel = self.dmodel // 4
        self.theta = theta
        self.wordemb = torch.nn.Embedding(2, self.dmodel)
        self.frequencyemb = torch.nn.Embedding(2, self.dmodel)
        if theta:
            self.initialphaseemb = torch.nn.Embedding(2, self.dmodel)

    def getembedding(self, x):
        amplitude = self.wordemb(x)
        frequency = self.frequencyemb(x)
        sentlen = x.size(-1)
        posseq = torch.arange(1, sentlen + 1, 1.0, device=amplitude.device)
        posseq = posseq.view(1, -1, 1)
        posseq = posseq.repeat([x.size(0), 1, amplitude.size(-1)])
        encoutputphase = torch.mul(posseq, frequency)
        if self.theta:
            self.initialphaseemb.weight = torch.nn.Parameter(self.initialphaseemb.weight % (2 * math.pi))
            encoutputphase = encoutputphase + self.initialphaseemb(x)
        encoutputphase = amplitude * torch.exp(1.j * encoutputphase)
        return encoutputphase

    def forward(self, x):
        batchsize, channels, size_x, size_y = x.shape
        grid_x = self.getembedding(torch.zeros(1, size_x, dtype=torch.long, device=x.device)) # [1, size_x, d_model]
        grid_y = self.getembedding(torch.ones(1, size_y, dtype=torch.long, device=x.device))
        grid_x = grid_x.transpose(1, 2).reshape(1, self.dmodel, size_x, 1).repeat([batchsize, 1, 1, size_y]) # [B, dmodel, size_x, size_y]
        grid_y = grid_y.transpose(1, 2).reshape(1, self.dmodel, 1, size_y).repeat([batchsize, 1, size_x, 1])
        grid = torch.cat([grid_x, grid_y], dim=1)
        if self.add:
            assert grid.shape == x.shape, f"Shape of grid {grid.shape} should match shape of x {x.shape} when injection=='add'"
            return x + grid
        return torch.cat([x, grid], dim=1)

class Transformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, head=1, num_layer=2, enc_modes=False, encoding=None,
                 dropout_rate=0, injection=None, layer_norm=True, activation="CReLU", postLN=False, Hermitian=False, complex_pos_enc=False):
        super(Transformer, self).__init__()

        self.complex_pos_enc = complex_pos_enc
        self.Hermitian = Hermitian
        self.num_layer = num_layer
        self.d_model = hidden_dim
        in_hidden_dim = hidden_dim

        self.enc_modes = enc_modes
        if enc_modes:
            assert encoding is not None, "encoding cannot be None"
            assert injection is not None, "injection cannot be None"
            assert encoding in ["complex", "stack"], "encoding not recognized"
            assert injection in ["add", "cat"], "injection not recognized"
            self.encoding = encoding
            self.injection = injection
            if injection == "cat":
                assert hidden_dim % 4 == 0, "hidden dim must be divisible by 4"
                in_hidden_dim = 3 * hidden_dim // 4

        if self.complex_pos_enc:
            # torch.manual_seed(1)
            self.complex_emb = complex_embedding(self.d_model, injection='cat')
            # x = torch.zeros(16, 32, 4, 4)


        self.encoder_layers = torch.nn.ModuleList()
        for i in range(self.num_layer):
            encoder = EncoderLayer(hidden_dim, hidden_dim*2, head=head, layer_norm=layer_norm,
                                   dropout_rate=dropout_rate, activation=activation, postLN=postLN)
            self.encoder_layers.append(encoder)


        self.input_projection = nn.Conv2d(input_dim, in_hidden_dim, 1, dtype=torch.cfloat)

        self.output_projection = nn.Linear(hidden_dim, output_dim, dtype=torch.cfloat)

    def mode_enc(self, x):

        # import pdb; pdb.set_trace()
        batchsize, _, size_x, size_y = x.shape
        d_model = self.d_model
        assert d_model % 2 == 0, "number of channels should be even"
        if self.encoding == "stack":
            d_model = d_model // 2
        if self.injection == "cat":
            d_model = d_model // 4

        device = x.device

        position = torch.fft.fftshift(torch.fft.fftfreq(size_x, 1 / size_x, device=device)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(len(position), d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(1, 0)
        pe_x = pe.clone()
        pe_y = pe.clone()

        pe_x = pe_x.reshape(1, d_model, size_x, 1).repeat([batchsize, 1, 1, size_y])
        pe_y = pe_y.reshape(1, d_model, 1, size_y).repeat([batchsize, 1, size_x, 1])

        if self.encoding == "stack":
            penc = torch.cat([pe_x, pe_y], dim=1)
        else: # self.enc_modes == "complex":
            penc = pe_x + 1.j * pe_y

        if self.injection == "add":
            assert x.shape == penc.shape, "shape of input and positional encoding do not match"
            return x + penc

        # else self.injection == "cat"

        return torch.cat([x, penc], dim=1)


    def forward(self, x): # [B, c, nx, ny]

        b, c, nx, ny = x.shape

        x = self.input_projection(x)

        if self.enc_modes:
            if self.complex_pos_enc:
                x = self.complex_emb(x)
            else:
                x = self.mode_enc(x)

        if self.Hermitian:
            ny = (ny // 2 + 1)
            x = x[..., :ny]

        x = x.reshape(b, -1, nx * ny).permute(0, 2, 1)  # [B, nx*ny, c]

        for i in range(self.num_layer):
            x = self.encoder_layers[i](x)

        out = self.output_projection(x) # [B, nx*ny, c]
        out = out.view(b, nx, ny, c).permute(0, 3, 1, 2) # [B, c, nx, ny]

        return out 


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, ffn_size, head=1, dropout_rate=0, layer_norm=True, activation="CReLU", postLN=False):
        super(EncoderLayer, self).__init__()

        self.postLN = postLN

        self.self_attention_norm = complexNorm(hidden_size) if layer_norm else torch.nn.Identity()
        self.self_attention = SelfAttention(hidden_size, head=head)
        self.self_attention_dropout = complexDropout(dropout_rate)

        self.ffn_norm = complexNorm(hidden_size) if layer_norm else torch.nn.Identity()
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, activation)
        self.ffn_dropout = complexDropout(dropout_rate)

    def forward(self, x):
        if self.postLN:
            y = self.self_attention(x)
            y = self.self_attention_dropout(y)
            y = x + y
            x = self.self_attention_norm(y)

            y = self.ffn(x)
            y = self.ffn_dropout(y)
            y = x + y
            x = self.ffn_norm(y)
        else:
            y = self.self_attention_norm(x)
            y = self.self_attention(y)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y
        return x

class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, head=1):
        super(SelfAttention, self).__init__()
        assert hidden_dim % head == 0, "Hidden dim must be divisible by number of heads"
        self.head = head
        self.dim_in = hidden_dim
        self.dim_k = hidden_dim
        self.dim_v = hidden_dim
        self.linear_q = nn.Linear(self.dim_in, self.dim_k, dtype=torch.cfloat)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, dtype=torch.cfloat)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, dtype=torch.cfloat)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(self.dim_k) // head)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.head
        dk = self.dim_k // nh
        dv = self.dim_v // nh
        
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        # dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # dist = torch.softmax(dist.abs(), dim=-1).to(torch.cfloat)  # batch, n, n

        # att = torch.bmm(dist, v)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist.abs(), dim=-1).to(torch.cfloat)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        return att
    
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_size, ffn_size, non_linearity):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = torch.nn.Linear(hidden_size, ffn_size, dtype=torch.cfloat)
        # self.gelu = torch.nn.GELU()
        # self.relu = ReLU(inplace=True)
#         self.swish = Swish()
        self.layer2 = torch.nn.Linear(ffn_size, hidden_size, dtype=torch.cfloat)
        self.relu = get_activation(non_linearity)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.gelu(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    
class complexNorm(nn.Module):
    def __init__(self, num_features):
        super(complexNorm, self).__init__()
        self.cnorm = nn.LayerNorm(num_features)
        self.rnorm = nn.LayerNorm(num_features)

    def forward(self, x):
        return self.rnorm(x.real) + 1.j * self.cnorm(x.imag)

class complexDropout(nn.Module):
    def __init__(self, dropout_rate, split=False):
        super(complexDropout, self).__init__()
        self.cdrop = torch.nn.Dropout(dropout_rate)
        self.rdrop = torch.nn.Dropout(dropout_rate)
        self.split = split

    def forward(self, x):
        if self.split:
            return self.rdrop(x.real) + 1.j * self.cdrop(x.imag)
        return self.rdrop(torch.ones_like(x.real)) * x


class SRCNN(nn.Module):
    def __init__(self, num_layers=3, input_dim=3, hidden_dim=16, output_dim=3, 
                 residual=False, upsample='nearest', scale_factor=1):
        super(SRCNN, self).__init__()
        
        self.residual = residual
        self.input_dim = input_dim

        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, output_dim, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode=upsample)

    def forward(self, x):

        b, t, nx, ny = x.shape

        up_dim = nx * self.scale_factor

        # import pdb; pdb.set_trace()
        if self.input_dim == 1:
            x = x.view(b*t, nx, ny).unsqueeze(1)

        if nx < 256:
            x = self.upsampler(x)

        ori_x = x.clone()
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        if self.residual:
            x = x + ori_x

        if self.input_dim == 1:
            x = x.view(b, t, up_dim, up_dim)

        return x


class Wrapper(nn.Module):
    def __init__(self, params, field_dim=1, model_name=None):
        super(Wrapper, self).__init__()
        
        self.field_dim = field_dim

        if model_name == 'SRCNN':
            model = SRCNN
        elif model_name == 'FNO':
            from models_FNO import FNO2d
            model = FNO2d
        elif model_name == 'RDN':
            model = RDN
        elif model_name == 'ModeFormer':
            model = ModeFormer
        elif model_name == 'RCAN':
            from RCAN import RCAN 
            model = RCAN
        else:
            raise ValueError(f"Model {model_name} not recognized")

        self.model_list = nn.ModuleList()
        for i in range(self.field_dim):
            self.model_list.append(model(**params))

    def forward(self, x):
        
        # input dim: [B, T*D, s_x, s_y]

        out_list = []
        for i in range(self.field_dim):
            inputs = x[:, i::self.field_dim, :, :]
            out = self.model_list[i](inputs)
            out_list.append(out)
        
        # import pdb; pdb.set_trace()
        B, T, sx, sy = out_list[0].shape
        out = torch.stack(out_list, dim=2).reshape(B, T*self.field_dim, sx, sy)

        return out
    


    
# code of RDN https://github.com/yjn870/RDN-pytorch/tree/f2641c0817f9e1f72acd961e3ebf42c89778a054

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        # assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_layers=3, input_dim=3, hidden_dim=16, output_dim=3, 
                 residual=False, upsample='nearest', scale_factor=1):
        super(UNet, self).__init__()

        self.residual = residual

        self.net = UNet4layers(input_dim, output_dim)

        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode=upsample)
        

    def forward(self, x):

        b,t,nx,ny = x.shape
        if nx < 256:
            x = self.upsampler(x)

        out = self.net(x)
        
        if self.residual:
            out = out + x

        return out