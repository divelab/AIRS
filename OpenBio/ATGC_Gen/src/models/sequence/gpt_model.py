"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        self.n_head = config.n_head

        self.config = config
        if config.condition == 'attn':
            self.key_cond = nn.Linear(config.n_embd, config.n_embd)
            self.query_cond = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, q, k, v, layer_past=None, condition=None, mask=True):
        B, T, C = q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(q).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.config.condition == 'attn':
            cond_rep = condition.unsqueeze(1).repeat_interleave(self.n_head, 1)  # B,nh,T,2
            k = torch.concat((k, cond_rep), dim=-1)
            q = torch.concat((q, cond_rep), dim=-1)
            # k_cond = self.key_cond(condition).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # q_cond = self.query_cond(condition).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # # att_cond = (q_cond @ k_cond.transpose(-2, -1)) * (1.0 / math.sqrt(k_cond.size(-1)))
            # # att += att_cond
            # k += k_cond
            # q += q_cond

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        if config.condition == 'cross':
            self.enc_dec_attn = MultiHeadAttention(config)
            self.ln3 = nn.LayerNorm(config.n_embd)
        self.config = config

    def forward(self, x, condition=None, mask=True):
        x = self.ln1(x)
        y, attn = self.attn(x, x, x, condition=condition, mask=mask)
        x = x + y
        if self.config.condition == 'cross':
            x = self.ln3(x)
            y1, attn = self.enc_dec_attn(x, condition, condition, condition=condition, mask=False)
            x = x + y1

        x = x + self.mlp(self.ln2(x))
        return x, attn


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        # if config.tokenizer_name == 'char':
        #     self.padding_token_id = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        # elif config.tokenizer_name == 'bpe':
        #     self.padding_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        # elif config.tokenizer_name == 'sp_bpe':
        #     self.padding_token_id = tokenizer.token_to_id("[PAD]")
        # else:
        #     NotImplementedError()

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # if config.condition == 'encoding':
        #     self.cond_emb = nn.Linear(config.cond_dim, 1)
        # elif config.condition == 'attn':
        #     self.cond_emb = nn.Linear(config.cond_dim, config.n_embd)  # cross attention - encoder
        # elif config.condition == 'concat':
        #     self.cond_emb = nn.Linear(config.n_embd + config.cond_dim, config.n_embd)
        if config.condition == 'front_discrete':
            self.cond_emb = nn.Embedding(config.num_classes + 1, config.n_embd)
        # elif config.condition == 'cross' or config.condition == 'front_continuous':
        #     self.cond_emb = nn.Linear(config.cond_dim, config.n_embd)
        #     self.cond_relu = nn.ReLU()
        #     self.cond_dropout = nn.Dropout(config.embd_pdrop)
        #     self.cond_norm = nn.LayerNorm(config.n_embd)
        elif config.condition == 'front_signal':
            self.cond_emb = nn.Linear(config.cond_dim * (config.block_size - 2), config.n_embd)
            self.cond_relu = nn.ReLU()
            self.cond_dropout = nn.Dropout(config.embd_pdrop)
        # elif config.condition == 'front_concat':
        #     # config.n_embd = config.n_embd + 2
        #     self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd - 2)
        #     self.cond_emb = nn.Linear(config.cond_dim * (config.block_size - 2), config.n_embd)
        #     self.cond_relu = nn.ReLU()
        #     self.cond_dropout = nn.Dropout(config.embd_pdrop)
        # elif config.condition == 'protein_cell':
        #     self.cell_emb = nn.Embedding(config.num_classes + 1, config.n_embd)
        #     self.protein_emb = nn.Linear(2560, config.n_embd)
        #     self.protein_relu = nn.ReLU()
        #     self.protein_dropout = nn.Dropout(config.embd_pdrop)
        elif config.condition == 'protein_embed':
            self.protein_emb = nn.Linear(2560, config.n_embd)
            self.cond_emb = nn.Embedding(config.num_classes + 1, config.n_embd)

        self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.isconditional = config.isconditional

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # def forward(self, input_ids, position_ids=None, inference_params=None, state=None):
    #     idx = input_ids[:,:,0].long()
    #     condition = input_ids[:,1:,1:3]

    def forward(self, idx, condition=None, condition2=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # +2 is for the special tokens
        if len(condition.shape) == 3:
            cond_len = condition.shape[1]
        elif len(condition.shape) == 1:
            cond_len = 1

        # forward the GPT model
        # if self.config.condition == 'no_cond':
        #     token_embeddings = self.tok_emb(idx)
        #     position_embeddings = self.pos_emb[:, :t, :]
        #     type_embeddings = self.type_emb(torch.ones((b,t), dtype=torch.long, device=idx.device))
        # elif self.config.condition == 'concat':
        #     token_embeddings = self.tok_emb(idx)
        #     concat_emb = torch.concat((token_embeddings, condition), dim=-1)
        #     token_embeddings = self.cond_emb(concat_emb)
        #
        #     position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        #     type_embeddings = self.type_emb(torch.ones((b,t), dtype=torch.long, device=idx.device))
        if self.config.condition == 'front_discrete':
            token_embeddings = self.tok_emb(idx)
            cond_embed = self.cond_emb(condition).unsqueeze(1)  # bs * 1 * dim
            token_embeddings = torch.concat((cond_embed, token_embeddings), dim=1)

            position_embeddings = self.pos_emb[:, :t+1, :]

            type_tensor = torch.zeros((b, t + 1), dtype=torch.long, device=idx.device)
            type_tensor[:, 1:] = 1
            type_embeddings = self.type_emb(type_tensor)
        # elif self.config.condition == 'front_continuous':
        #     token_embeddings = self.tok_emb(idx)
        #     conditions_emb = self.cond_emb(condition)  # bs * 1024 * dim
        #     conditions_emb = self.cond_relu(conditions_emb)
        #     conditions_emb = self.cond_dropout(conditions_emb)
        #     conditions_emb = self.cond_norm(conditions_emb)
        #     token_embeddings = torch.concat((conditions_emb, token_embeddings), dim=1)
        #
        #     position_embeddings = self.pos_emb[:, :t+cond_len, :]
        #
        #     type_tensor = torch.zeros((b, t + cond_len), dtype=torch.long, device=idx.device)
        #     type_tensor[:, cond_len:] = 1
        #     type_embeddings = self.type_emb(type_tensor)
        elif self.config.condition == 'front_signal':
            cond_len = 1
            token_embeddings = self.tok_emb(idx)
            conditions_emb = self.cond_dropout(self.cond_relu(self.cond_emb(condition.reshape(b, 1, -1))))  # bs * 1 * (1024*2)
            token_embeddings = torch.concat((conditions_emb, token_embeddings), dim=1)

            position_embeddings = self.pos_emb[:, :t+cond_len, :]

            type_tensor = torch.zeros((b, t + cond_len), dtype=torch.long, device=idx.device)
            type_tensor[:, cond_len:] = 1
            type_embeddings = self.type_emb(type_tensor)
        # elif self.config.condition == 'front_concat':
        #     cond_len = 1
        #     token_embeddings = self.tok_emb(idx)  # bs * 1025 * 766
        #     padded_condition = F.pad(condition, (0, 0, 1, 0))  # 16 * 1025 * 2
        #     token_embeddings = torch.concat((token_embeddings, padded_condition[:,:t,:]), dim=-1)  # bs * 1025 * 768
        #     conditions_emb = self.cond_dropout(self.cond_relu(self.cond_emb(condition.reshape(b, 1, -1))))  # bs * 1 * (1024*2)
        #     token_embeddings = torch.concat((conditions_emb, token_embeddings), dim=1)  # bs * 1026 * 768
        #
        #     position_embeddings = self.pos_emb[:, :t+cond_len, :]
        #
        #     type_tensor = torch.zeros((b, t + cond_len), dtype=torch.long, device=idx.device)
        #     type_tensor[:, cond_len:] = 1
        #     type_embeddings = self.type_emb(type_tensor)
        elif self.config.condition == 'protein_embed':
            cond_len = 1000
            token_embeddings = self.tok_emb(idx)  # bs * 501 * 768
            protein_embeddings = self.protein_emb(condition)  # bs * 1000 * 768
            # cell_embeddings = self.cond_emb(condition2).unsqueeze(1)  # bs * 1 * dim
            token_embeddings = torch.concat((protein_embeddings, token_embeddings), dim=1)
            # bs * 1501 * 768

            position_embeddings = self.pos_emb[:, :t+cond_len, :]

            type_tensor = torch.zeros((b, t + cond_len), dtype=torch.long, device=idx.device)
            type_tensor[:, cond_len:] = 1
            type_embeddings = self.type_emb(type_tensor)
        else:
            raise NotImplementedError()

        # elif self.config.condition == 'protein_cell':
        #     assert protein_embed is not None
        #     cell_embed = self.cell_emb(condition).unsqueeze(1)
        #     protein_embed = self.protein_dropout(self.protein_relu(self.protein_emb(protein_embed.reshape(b,1,-1))))
        #     cond_len = 2
        #
        #     token_embeddings = self.tok_emb(idx)
        #     token_embeddings = torch.concat((cell_embed, protein_embed, token_embeddings), dim=1)
        #     position_embeddings = self.pos_emb[:, :t+cond_len, :]
        #     type_tensor = torch.zeros((b, t + cond_len), dtype=torch.long, device=idx.device)
        #     type_tensor[:, cond_len:] = 1
        #     type_embeddings = self.type_emb(type_tensor)

        x = token_embeddings + position_embeddings + type_embeddings
        # if self.config.condition == 'encoding':
        #     cond_embeddings = self.cond_emb(condition)
        #     x += cond_embeddings
        x = self.drop(x)

        conditions_emb = None
        # if self.config.condition == 'attn':
        #     conditions_emb = condition
        # elif self.config.condition == 'cross':
        #     conditions_emb = self.cond_emb(condition)
        #     conditions_emb = self.cond_relu(conditions_emb)
        #     conditions_emb = self.cond_dropout(conditions_emb)
        #     conditions_emb = self.cond_norm(conditions_emb)

        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x, condition=conditions_emb)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        if (
                self.config.condition == 'front_discrete' or
                # self.config.condition == 'front_continuous' or
                self.config.condition == 'front_signal' or
                # self.config.condition == 'front_concat' or
                # self.config.condition == 'protein_cell' or
                self.config.condition == 'protein_embed'
        ):
            logits = logits[:, cond_len:, :]

        return logits
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=logits), None

        # # if we are given some desired targets also calculate the loss
        # loss = None
        # if targets is not None:
        #     mask = targets != self.padding_token_id
        #     # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
        #     loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
        #     loss = (loss * mask.view(-1)).sum() / mask.sum()
        #
        # return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)
