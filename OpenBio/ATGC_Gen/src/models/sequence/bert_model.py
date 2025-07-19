import torch
import torch.nn as nn
import logging

from .gpt_model import Block

logger = logging.getLogger(__name__)


class Bert(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.beta_params = (3, 9)

        # embedding layer
        self.config = config
        if config.dataset_name == 'promoter':
            self.cond_dim = 2
        elif config.dataset_name == 'flybrain_enhancer' or config.dataset_name == 'mel_enhancer':
            self.cond_dim = 1
            self.cond_emb = nn.Embedding(config.num_classes + 1, config.n_embd)
        else:
            raise NotImplementedError()
        self.embed_layer = nn.Linear(self.cond_dim + self.config.vocab_size, config.n_embd)

        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        #
        # if config.condition == 'front_discrete':
        #     self.cond_emb = nn.Embedding(config.num_classes + 1, config.n_embd)
        # elif config.condition == 'front_signal':
        #     self.cond_emb = nn.Linear(config.cond_dim * (config.block_size - 2), config.n_embd)
        #     self.cond_relu = nn.ReLU()
        #     self.cond_dropout = nn.Dropout(config.embd_pdrop)
        # self.type_emb = nn.Embedding(2, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.pred_vocab_size, bias=False)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.config.block_size

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

    def generate_mask(self, input_embeds, input_seqs):
        batch_size, seq_length, embed_dim = input_embeds.shape

        # Step 1: Sample mask rate dynamically
        beta_dist = torch.distributions.Beta(*self.beta_params)
        uniform_dist = torch.distributions.Uniform(0, 1)

        beta_mask_rate = beta_dist.sample((batch_size,)).to(input_embeds.device)  # (batch_size,)
        uniform_mask_rate = uniform_dist.sample((batch_size,)).to(input_embeds.device)  # (batch_size,)

        # Use Bernoulli to decide whether to sample from Beta or Uniform
        use_uniform = torch.bernoulli(torch.full((batch_size,), 0.2, device=input_embeds.device))  # (batch_size,)

        # Compute final mask rate
        mask_rate = beta_mask_rate * (1 - use_uniform) + uniform_mask_rate * use_uniform  # (batch_size,)

        # Step 2: Generate mask tensor
        mask_tensor = torch.rand(batch_size, seq_length, device=input_embeds.device) < mask_rate.unsqueeze(1)  # (batch_size, seq_length)

        # Step 3: Create masked embeddings
        masked_embeds = input_embeds.clone()
        mask_labels = input_seqs.clone()

        # Apply BERT-style masking rules
        mask_replace_prob = torch.rand(batch_size, seq_length, device=input_embeds.device)

        # 80%: Replace with [MASK] token embedding
        mask_80 = (mask_replace_prob < 0.8) & mask_tensor
        # masked_embeds[mask_80, :4] = 0  # Replace with [MASK] embedding
        expanded_mask = mask_80.unsqueeze(-1).expand(-1, -1, self.config.vocab_size)
        # temp = masked_embeds[..., :4]
        # temp[expanded_mask] = 0

        masked_embeds[..., :self.config.vocab_size].masked_fill_(expanded_mask, 0)

        # 10%: Replace with random token embedding
        mask_10 = (mask_replace_prob >= 0.8) & (mask_replace_prob < 0.9) & mask_tensor
        expanded_mask = mask_10.unsqueeze(-1).expand(-1, -1, self.config.vocab_size)
        rand_indices = torch.randint(0, self.config.vocab_size, (mask_10.sum(),))  # (mask_10.sum(),)
        one_hot_vectors = torch.eye(self.config.vocab_size).to(input_embeds.device)
        # masked_embeds[mask_10, :4] = one_hot_vectors[torch.randint(0, 4, (mask_10.sum(),))]
        masked_embeds[..., :self.config.vocab_size].masked_scatter_(expanded_mask, one_hot_vectors[rand_indices].flatten())

        # 10%: Keep original embeddings (No action needed)

        # Step 4: Mask out non-masked positions in labels
        mask_labels[~mask_tensor] = -1

        return masked_embeds, mask_labels

    def forward(self, input_embeds, input_seqs, condition=None, protein_embed=None):
        b, t, input_dim = input_embeds.shape
        assert input_embeds.shape[1] == input_seqs.shape[1]
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."
        if self.training:
            # mask tokens
            mask_input, label = self.generate_mask(input_embeds, input_seqs)
        else:
            # pure generation
            mask_input = input_embeds
            label = input_seqs
        token_emb = self.embed_layer(mask_input)
        # if condition is not None:
        #     cond_emb = self.cond_emb(condition)
        #     token_emb = token_emb + cond_emb.unsqueeze(1)
        #     # token_emb = torch.concat((cond_emb.unsqueeze(1), token_emb), dim=1)
        x = token_emb + self.pos_emb

        x = self.drop(x)

        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x, condition=None, mask=False)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        if logits.shape[1] != t:
            logits = logits[:, -t:, :]

        return logits, label


