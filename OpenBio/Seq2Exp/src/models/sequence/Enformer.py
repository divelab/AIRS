import torch
import torch.nn as nn
from enformer_pytorch import from_pretrained, seq_indices_to_one_hot
from typing import Optional, Union
from transformers import PretrainedConfig, PreTrainedModel
from enformer_pytorch import Enformer, seq_indices_to_one_hot


class EnformerConfig(PretrainedConfig):
    model_type = "Enformer"

    def __init__(
            self,
            use_pretrain: bool = True,
            freeze: bool = True,
            d_model: int = 128,
            rna_feat_dim: int = 9,
            useRNAFeat: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_pretrain = use_pretrain
        self.freeze = freeze
        self.d_model = d_model
        self.useRNAFeat = useRNAFeat
        self.rna_feat_dim = rna_feat_dim


class GeneEnformer(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.valid_len_enformer = 196_608
        if config.use_pretrain:
            self.enformer = from_pretrained('EleutherAI/enformer-official-rough')
            if config.freeze:
                for param in self.enformer.parameters():
                    param.requires_grad = False
        else:
            self.enformer = Enformer.from_hparams()

        # output
        self.pToExpr = nn.Sequential(
            nn.Linear(self.enformer.dim * 2 + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        seqs,
        signals=None,
        mask_regions=None,
        bio_mask=None,
        peak_mask=None,
        rna_feat=None,
    ):
        # split seq
        start = (seqs.shape[1] - self.valid_len_enformer) // 2
        end = start + self.valid_len_enformer
        seqs = seqs[:,start:end]
        seqs = seqs - 7

        # seq: one hot
        output, embeddings = self.enformer(seqs, return_embeddings=True)
        seq_embeddings = torch.mean(embeddings, dim=1)

        p_embed = torch.cat([seq_embeddings, rna_feat], dim=-1) if self.config.useRNAFeat else seq_embeddings
        logits = self.pToExpr(p_embed)

        logits = logits.float()
        return logits
