import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    cross_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
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
        if config.num_props:
            if config.scaffold_maxlen:
                num = int(bool(config.num_props)) + int(config.scaffold_maxlen)
            else:
                num = int(bool(config.num_props))
        else:
            num = 0
        # num = int(bool(config.num_props)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.scaffold)
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save
    
    
class CrossAttention(nn.Module):
    
    """
    add cross attention
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        # self.key = nn.Linear(config.n_embd, config.n_embd)
        # self.query = nn.Linear(config.n_embd, config.n_embd)
        # self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # cross attention
        self.cross_key = nn.Linear(config.n_embd, config.n_embd)
        self.cross_query = nn.Linear(config.n_embd, config.n_embd)
        self.cross_value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.cross_crop = nn.Dropout(config.cross_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if config.num_props:
            if config.scaffold_maxlen:
                num = int(bool(config.num_props)) + int(config.scaffold_maxlen)
            else:
                num = int(bool(config.num_props))
        else:
            num = 0
        # num = int(bool(config.num_props)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.scaffold)
        # num = 1
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
        #                              .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head
        
        if config.mode == 'concat':
            self.concat_proj = nn.Sequential(
                nn.Linear(config.n_embd + 512, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
                nn.LayerNorm(config.n_embd),
                nn.Dropout(config.cross_pdrop),
            )

        elif config.mode == 'cross':
            self.cross_attn_proj = nn.Sequential(
                nn.Linear(512, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
                nn.LayerNorm(config.n_embd),
                nn.Dropout(config.cross_pdrop),
            )
        else:
            raise ValueError('mode should be "concat" or "cross"')

    def forward(self, x, layer_past=None, encoder_embedding=None, encoder_mask=None, mode='concat'):
        B, T, C = x.size()

        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # attn_save = att
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side


        if encoder_embedding is not None and encoder_mask is not None:
            ## option 1: pool encoder_embedding to a single vector, then concat it to each position and use MLP to adjust dimension
            if mode == 'concat':
                # import pdb; pdb.set_trace()
                encoder_embedding = encoder_embedding.sum(axis=1)
                
                # assume encoder_embedding shape is (B, C), we expand it to (B, T, C)
                encoder_embedding_expanded = encoder_embedding.unsqueeze(1).expand(-1, T, -1)
                cross_att_in = torch.cat([x, encoder_embedding_expanded], dim=-1)
                # import pdb; pdb.set_trace()
                cross_att_in = self.concat_proj(cross_att_in)

                # cross attention key, query, value
                cross_k = self.cross_key(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_q = self.cross_query(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_v = self.cross_value(cross_att_in).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                
                # calculate cross attention weights
                cross_att = (cross_q @ cross_k.transpose(-2, -1)) * (1.0 / math.sqrt(cross_k.size(-1)))
                cross_att = F.softmax(cross_att, dim=-1)
                cross_att = self.attn_drop(cross_att)
                cross_y = cross_att @ cross_v
                cross_y = cross_y.transpose(1, 2).contiguous().view(B, T, C)
                # import pdb; pdb.set_trace()
            
            elif mode == 'cross':
                ## option 2: encoder_embedding is a sequence of vectors, then use encoder_mask to mask the padding
                # encoder_embedding shape is (B, S, C)，S is sequence length after padding
                # encoder_mask shape is (B, S)，valid position is 1，padding position is 0
                # import pdb; pdb.set_trace()
                B, S, _ = encoder_embedding.size()
                encoder_embedding = self.cross_attn_proj(encoder_embedding)
                # key, query, value
                cross_k = self.cross_key(encoder_embedding).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
                cross_q = self.cross_query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                cross_v = self.cross_value(encoder_embedding).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
                
                # calculate cross attention weights
                cross_att = (cross_q @ cross_k.transpose(-2, -1)) * (1.0 / math.sqrt(cross_k.size(-1)))
                # import pdb; pdb.set_trace()
                # apply encoder_mask to mask the padding position
                if encoder_mask is not None:
                    # expand mask to match attention score shape (B, 1, 1, S)
                    encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
                    # set the attention score of padding position to negative infinity
                    cross_att = cross_att.masked_fill(encoder_mask == 0, float('-inf'))
                
                # softmax 
                cross_att = F.softmax(cross_att, dim=-1)
                cross_att = self.attn_drop(cross_att)
                # import pdb; pdb.set_trace()
            
                cross_y = cross_att @ cross_v
                cross_y = cross_y.transpose(1, 2).contiguous().view(B, T, C)
                # import pdb; pdb.set_trace()
            
            else:
                raise ValueError('mode should be "concat" or "cross"')


        # output projection
        y = self.resid_drop(self.proj(cross_y))
        return y, None

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ESM_protein = False
        if config.ESM_protein:
            self.cross_attn = CrossAttention(config)
            self.ln_cross = nn.LayerNorm(config.n_embd)
            self.ESM_protein = True
        
        self.attn = CausalSelfAttention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, protein_padded_embedding=None, protein_embedding_mask=None, mode='concat'):

        y, attn = self.attn(self.ln1(x))
        x = x + y

        if self.ESM_protein:
            y, _ = self.cross_attn(self.ln_cross(x), 
                                    encoder_embedding=protein_padded_embedding, 
                                    encoder_mask=protein_embedding_mask,
                                    mode=mode)
            x = x + y

        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.padding_token_id = 0
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
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

    def forward(self, idx, targets=None, condition_split_id=None, prop=None, scaffold=None, protein_padded_embedding=None, protein_embedding_mask=None):
        # import pdb; pdb.set_trace()
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # if self.config.num_props:
        #     assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        # if self.config.num_props:
        #     type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
        #     if prop.ndim == 2:
        #         p = self.prop_nn(prop.unsqueeze(1))    # for single property
        #     else:
        #         p = self.prop_nn(prop)    # for multiproperty
        #     p += type_embd
        #     x = torch.cat([p, x], 1)

        # if self.config.scaffold:
        #     type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

        #     scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
        #     if self.config.lstm:
        #         scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
        #         # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.n_embd).mean(2)
        #         scaffold_embeds = scaffold_embeds.permute(1,0,2)   # mean(0, keepdim = True)
        #         # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.n_embd)[-1].permute(1,0,2)
        #         # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.n_embd)
        #     scaffold_embeds += type_embd
        #     x = torch.cat([scaffold_embeds, x], 1)

        # x = self.blocks(x)
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x, 
                            protein_padded_embedding=protein_padded_embedding,
                            protein_embedding_mask=protein_embedding_mask,
                            mode=self.config.mode)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # print(logits.shape)
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen) 
        else:
            num = 0

        logits = logits[:, num:, :]


        # if self.config.num_props or self.config.scaffold:

        #     num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)  #int(self.config.lstm_layers)   # int(self.config.scaffold)      # int(self.config.scaffold)
            

        # print(logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            mask = targets != self.padding_token_id
            if self.isconditional or self.config.rag:
                # mask[:, 0:condition_split_id] = False
                # Create a range tensor [0, 1, 2, ..., seq_len-1]
                range_tensor = torch.arange(t, device=mask.device).expand(b, -1)
                # Expand split_id to match the shape of range_tensor for broadcasting
                expanded_split_id = condition_split_id.unsqueeze(1).expand(-1, t)
                # Generate the update mask (True where range_tensor < expanded_split_id)
                cond_mask = range_tensor < expanded_split_id
                # Update the original mask (set to False where the condition is False)
                mask[cond_mask] = False
            # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            
            if self.config.rag and self.config.alpha > 0:
                # print(condition_split_id)
                mask = targets != self.padding_token_id
                # Create a range tensor [0, 1, 2, ..., seq_len-1]
                range_tensor = torch.arange(t, device=mask.device).expand(b, -1)
                # Expand split_id to match the shape of range_tensor for broadcasting
                expanded_split_id = condition_split_id.unsqueeze(1).expand(-1, t)
                # Generate the update mask (True where range_tensor < expanded_split_id)
                cond_mask = range_tensor >= expanded_split_id
                # Update the original mask (set to False where the condition is False)
                mask[cond_mask] = False

                retrieve_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                retrieve_loss = (retrieve_loss * mask.view(-1)).sum() / mask.sum()

                loss += retrieve_loss * self.config.alpha


        return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)
    
        