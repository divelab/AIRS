import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class seq_256bp_encoder(nn.Module):
    def __init__(self, base_size=4, out_dim=128, conv_dim=256):
        super(seq_256bp_encoder, self).__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        # cropped_len = 46
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_size, out_channels=self.conv_dim, kernel_size=(1, 8), stride=1, padding='same'),
            nn.ELU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dim = [self.conv_dim, 128, 64, 64, 128]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels=conv_dim[i], out_channels=conv_dim[i + 1], kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(conv_dim[i + 1]),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ))
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels=conv_dim[i + 1], out_channels=conv_dim[i + 1], kernel_size=(1, 1)),
                nn.ELU(),
            ))

    def forward(self, enhancers_input):
        if enhancers_input.shape[2] == 1:
            x_enhancer = enhancers_input
        else:
            x_enhancer = enhancers_input.permute(0, 3, 1, 2).contiguous()  # batch_size x 4 x 78 x 12566
        x_enhancer = self.stem_conv(x_enhancer)
        #         print(x_enhancer.shape)
        for i in range(0, len(self.conv_tower), 2):
            x_enhancer = self.conv_tower[i](x_enhancer)
            x_enhancer = self.conv_tower[i + 1](x_enhancer) + x_enhancer
        return x_enhancer


class enhancer_predictor_256bp(nn.Module):
    def __init__(self):
        super(enhancer_predictor_256bp, self).__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, enhancer_seq):
        if len(enhancer_seq.shape) < 4:
            enhancer_seq = enhancer_seq.unsqueeze(2)
        seq_embed = self.encoder(enhancer_seq)
        epi_out = self.embedToAct(seq_embed)
        return epi_out.squeeze(-1)


class MHAttention_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.):
        super(MHAttention_encoderLayer, self).__init__()
        # self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, 4*d_model) might cause loading problem, this parameter is not neccessary
        # self.linear2 = nn.Linear(4*d_model, d_model) might cause loading problem, this parameter is not neccessary
        # self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    # self-attention block
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w

    def forward(self, x, enhancers_padding_mask=None, attn_mask=None):
        x2 = self.norm1(x)
        x2, attention_w = self._sa_block(x2, key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
        x = x2 + x
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x, attention_w


class MHAttention_encoderLayer_noLN(nn.Module):
    def __init__(self, d_model=2048, nhead=8, dim_feedforward=256, dropout=0.1, activation=F.relu):
        super(MHAttention_encoderLayer_noLN, self).__init__()
        self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)

    # self-attention block
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w

    def forward(self, x_pe, enhancers_padding_mask=None, attn_mask=None):
        xt, attention_w = self._sa_block(x_pe, enhancers_padding_mask, attn_mask=attn_mask)
        x_pe = x_pe + xt
        x_pe = x_pe + self._ff_block(x_pe)
        return x_pe, attention_w


class EPInformer_v2(nn.Module):
    def __init__(self, base_size=4, n_encoder=3, out_dim=128, head=4, pre_trained_encoder=None, n_enhancers=50,
                 useBN=True, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_v2, self).__init__()
        self.n_enhancers = n_enhancers
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformerV2.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(
                base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancers)
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformerV2.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(
                base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancers)
        self.n_encoder = n_encoder
        if useLN:
            self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        else:
            self.attn_encoder = get_clones(MHAttention_encoderLayer_noLN(d_model=out_dim, nhead=head), self.n_encoder)
        attn_mask = (~np.identity(self.n_enhancers + 1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim / 32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ELU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim / 32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        if self.useFeat:
            if self.usePromoterSignal:
                feat_n = 9
            else:
                feat_n = 8
            self.pToExpr = nn.Sequential(
                nn.Linear(self.out_dim + feat_n, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.pToExpr = nn.Sequential(
                nn.Linear(self.out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        self.add_pos_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.out_dim + n_extraFeat, out_channels=self.out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feat=None, extraFeat=None):
        # if enhancers_padding_mask is None:
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        #         print(enhancers_padding_mask)
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        if extraFeat is not None:
            pe_flatten_embed = self.add_pos_conv(
                torch.concat([pe_flatten_embed, extraFeat], axis=-1).permute(0, 2, 1)).permute(0, 2, 1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed,
                                                          enhancers_padding_mask=enhancers_padding_mask,
                                                          attn_mask=self.attn_mask.to(pe_flatten_embed.device))
            attn_list.append(attn.unsqueeze(0))
        p_embed = torch.flatten(pe_flatten_embed[:, 0, :], start_dim=1)
        if self.useFeat:
            p_embed = torch.cat([p_embed, rna_feat], dim=-1)
        p_expr = self.pToExpr(p_embed)
        return p_expr, torch.cat(attn_list)
