import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import re
import torch.distributions as dist
from collections import namedtuple

from transformers.modeling_outputs import MaskedLMOutput
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MixerModel

from caduceus.modeling_caduceus import CaduceusPreTrainedModel, Caduceus
from caduceus.configuration_caduceus import CaduceusConfig, ExtendedMambaConfig
from src.models.sequence.EPInformer import EPInformer_v2
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.models.sequence.long_conv_lm import LMBackbone


def entropy(mask, eps=1e-10):
    entropy = -mask * torch.log(mask + eps) - (1 - mask) * torch.log(1 - mask + eps)
    average_entropy = entropy.mean()
    return average_entropy


def smooth_max(tensor, window_size):
    """take max value of signals over window size"""
    tensor = tensor.unsqueeze(1)
    smoothed_tensor = F.max_pool1d(tensor, kernel_size=window_size, stride=1, padding=(window_size - 1) // 2)
    smoothed_tensor = smoothed_tensor.squeeze(1)
    return smoothed_tensor


def moving_average_cal(logits, mv_kernel=1, stride=1, padding_value=0.5, padding_mode='same'):
    # moving average
    kernel = torch.ones((logits.shape[-1], 1, mv_kernel)) / mv_kernel
    kernel = kernel.to(logits.device, dtype=logits.dtype)
    kernel.requires_grad = False

    logits = logits.permute(0, 2, 1)
    if padding_mode == 'same':
        right_pad = (mv_kernel - 1) // 2 if (mv_kernel - 1) % 2 == 0 else (mv_kernel - 1) // 2 + 1
        left_pad = (mv_kernel - 1) // 2
    elif padding_mode == 'no_pad':
        left_pad = right_pad = 0
    logits_padded = F.pad(logits, (left_pad, right_pad), mode='constant', value=padding_value)
    logits = F.conv1d(logits_padded, kernel, stride=stride, padding=0, groups=logits.shape[1])
    logits = logits.permute(0, 2, 1)

    return logits


def gumbel_softmax_threshold(logits, tau: float = 1, hard: bool = False, dim: int = -1, threshold=0.5, mv_kernel=1,
                             bio_mask=None, bio_mask_weight=0.0, counter_zero=True, merge_mask=False, subseq_size=1000,
                             node_merge_mask=False, node_merge_range=1, is_training=False):
    if node_merge_mask:
        logits = moving_average_cal(logits, mv_kernel=node_merge_range, stride=node_merge_range, padding_mode='no_pad')

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if node_merge_mask:
        y_soft = torch.repeat_interleave(y_soft, repeats=node_merge_range, dim=1)

    # moving average
    y_soft = moving_average_cal(y_soft, mv_kernel=mv_kernel, stride=1, padding_value=0.5, padding_mode='same')

    if bio_mask is not None and bio_mask_weight != 0.0:
        bio_mask_true = bio_mask[..., 0]
        bio_mask_complement = torch.zeros_like(bio_mask_true) if counter_zero else 1 - bio_mask_true
        bio_mask = torch.concat((bio_mask_complement.unsqueeze(-1), bio_mask_true.unsqueeze(-1)), dim=-1)
        y_soft = (1 - bio_mask_weight) * y_soft + bio_mask_weight * bio_mask

    # make sure the middle 2k part is 1
    promo_mask = torch.zeros_like(y_soft[:,:,1])
    start = (promo_mask.shape[1] - 2000) // 2
    promo_mask[:, start:start + 2000] = 1.1
    y1 = torch.max(y_soft[:, :, 1], promo_mask)
    y_soft = torch.concat((y_soft[:, :, 0:1], y1.unsqueeze(-1)), dim=-1)

    if hard:
        # Straight through.
        max_vals, index = y_soft.max(dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        if not is_training:
            mask = max_vals > threshold
            y_hard = y_hard * mask.to(dtype=y_soft.dtype)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    # if merge_mask:
    #     ret_repeat = torch.repeat_interleave(ret, repeats=subseq_size, dim=1)
    #     tensor1 = F.pad(ret_repeat, (0, 0, subseq_size, 0))  # (bs, 1000, dim)
    #     tensor2 = F.pad(ret_repeat, (0, 0, 0, subseq_size))  # (bs, 1000, dim)
    #     ret = torch.max(tensor1, tensor2)
    return ret


def Beta_fn(a, b):
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))


def kldivergence_kuma(distribution, prior_alpha, prior_beta):
    distribution.a = distribution.concentration1
    distribution.b = distribution.concentration0
    # prior_alpha = torch.tensor([1.0], device=distribution.a.device)
    # prior_beta = torch.tensor([4.0], device=distribution.a.device)
    kl = 1. / (1 + distribution.a * distribution.b) * Beta_fn(distribution.a.reciprocal(), distribution.b)
    kl += 1. / (2 + distribution.a * distribution.b) * Beta_fn(2.0 * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (3 + distribution.a * distribution.b) * Beta_fn(3. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (4 + distribution.a * distribution.b) * Beta_fn(4. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (5 + distribution.a * distribution.b) * Beta_fn(5. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (6 + distribution.a * distribution.b) * Beta_fn(6. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (7 + distribution.a * distribution.b) * Beta_fn(7. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (8 + distribution.a * distribution.b) * Beta_fn(8. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (9 + distribution.a * distribution.b) * Beta_fn(9. * distribution.a.reciprocal(), distribution.b)
    kl += 1. / (10 + distribution.a * distribution.b) * Beta_fn(10. * distribution.a.reciprocal(), distribution.b)
    kl *= (prior_beta - 1) * distribution.b

    # use another taylor approx for Digamma function
    psi_b_taylor_approx = torch.log(distribution.b) - 1. / (2 * distribution.b) - 1. / (12 * distribution.b ** 2)
    kl += (distribution.a - prior_alpha) / distribution.a * (
                -0.57721 - psi_b_taylor_approx - 1 / distribution.b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl += torch.log(distribution.a * distribution.b) + torch.log(Beta_fn(prior_alpha, prior_beta))

    # final term
    kl += -(distribution.b - 1) / distribution.b

    return kl


class GeneExpHyena(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        interact='',
        use_bio_mask=False,
        base_size=4,
        signal_size=3,
        center_len=2000,
        rna_feat_dim=9,
        useRNAFeat=True,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.center_len = center_len
        self.useRNAFeat = useRNAFeat

        self.model = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
            **kwargs,
        )

        self.pToExpr = nn.Sequential(
            nn.Linear(d_model + rna_feat_dim if useRNAFeat else d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        hidden_states = self.model(
            seqs, position_ids=None, inference_params=None
        )

        if self.center_len:
            start_index = (hidden_states.shape[1] - self.center_len) // 2
            end_index = start_index + self.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits


class GeneExpMamba(nn.Module):
    def __init__(
            self,
            config: ExtendedMambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        # if config.interact == 'concat':
        #     input_dim = config.base_size + config.signal_size
        # elif config.interact == 'no_signal':
        #     input_dim = config.base_size
        # self.input_layer = nn.Linear(input_dim, config.d_model)

        self.config = config
        self.model = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            **{"device": device, "dtype": dtype},
        )

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        # if self.config.interact == 'concat':
        #     if self.config.signal_size == 2:
        #         signals = signals[..., :2]
        #     inputs_embeds = torch.concat((seqs, signals), dim=-1)
        # elif self.config.interact == 'no_signal':
        #     inputs_embeds = seqs
        #
        # inputs_embeds = self.input_layer(inputs_embeds)

        hidden_states = self.model(input_ids=seqs)

        if self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits


class GeneExpBiMamba(CaduceusPreTrainedModel):
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        if config.pretrained_model:
            # self.pre_model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name, trust_remote_code=True)
            state_dict = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name, trust_remote_code=True).state_dict()
            state_dict = {key.replace("caduceus.", ""): value for key, value in state_dict.items()}
            self.pre_model = Caduceus(config, **{'ignore_embed_layer': True})
            missing_keys, unexpected_keys = self.pre_model.load_state_dict(state_dict, strict=False)
            assert len(missing_keys) == 0

            if config.pretrained_freeze:
                for param in self.pre_model.parameters():
                    param.requires_grad = False
            # input
            # match1 = re.search(r'd_model-(\d+)', config.pretrained_model_name)
            # pretrain_dim = int(match1.group(1))
            # config.d_model = pretrain_dim
            # if config.interact == 'concat':
            #     input_layer_dim1 = pretrain_dim + config.signal_size
            # elif config.interact == 'no_signal':
            #     input_layer_dim1 = pretrain_dim
            # else:
            #     raise NotImplementedError()
            # self.input_layer = nn.Linear(input_layer_dim1, config.d_model)
            # model
            # downstream_config = copy.deepcopy(config)
            # downstream_config.n_layer = config.down_n_layer
            # self.caduceus = Caduceus(downstream_config, **kwargs)

        else:
            self.caduceus = Caduceus(config, **{'ignore_embed_layer': True})

        if config.interact == 'concat':
            input_dim = config.base_size + config.signal_size
        elif config.interact == 'no_signal':
            input_dim = config.base_size
        self.input_layer = nn.Linear(input_dim, config.d_model)

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        if self.config.interact == 'concat':
            if self.config.signal_size == 2:
                signals = signals[..., :2]
            inputs_embeds = torch.concat((seqs, signals), dim=-1)
        elif self.config.interact == 'no_signal':
            inputs_embeds = seqs

        if self.config.use_bio_mask:
            bio_mask_epinformer = bio_mask[..., 1]
            inputs_embeds = inputs_embeds * bio_mask_epinformer.unsqueeze(-1)

        inputs_embeds = self.input_layer(inputs_embeds)

        if self.config.pretrained_model:
            output_dict = self.pre_model(input_ids=None, inputs_embeds=inputs_embeds, output_hidden_states=True)
            last_hidden_states = output_dict.hidden_states[-1]  # dim=256
            if self.config.use_bio_mask:
                bio_mask_epinformer = bio_mask[..., 1]
                outputs = last_hidden_states * bio_mask_epinformer.unsqueeze(-1)
            else:
                outputs = last_hidden_states
        else:
            """HF-compatible forward method."""
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.caduceus(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs
        if self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits


class ModelMask(CaduceusPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.promo_len = config.center_len
        assert not (config.test_top and config.test_soft)
        assert not (config.use_bio_mask and config.use_peak_mask)

        # input layer
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model)
        if config.interact == 'concat':
            self.signal_input_layer = nn.Linear(config.signal_size, config.d_model)

        # generator
        gen_config = copy.deepcopy(config)
        gen_config.n_layer = config.gen_n_layer
        self.generator = Caduceus(gen_config, **{'ignore_embed_layer': True})

        # prior input weight - signal
        prior_weight = [float(x) for x in config.prior_weight.split(',')]
        prior_weight = [x / sum(prior_weight) for x in prior_weight]
        prior_weight = torch.log(torch.tensor(prior_weight, dtype=torch.float32))
        self.prior_weights = nn.Parameter(prior_weight, requires_grad=config.dist_param_grad)

        # signal beta
        signal_beta = [float(x) for x in config.prior_beta.split(',')]
        signal_beta = torch.tensor(signal_beta, dtype=torch.float32)
        self.signal_betas = nn.Parameter(signal_beta, requires_grad=config.dist_param_grad)

        # mask output
        self.mask_output = nn.Linear(config.d_model, 2)
        # Initialize weights and apply final processing
        self.post_init()

    def couple_post_dist(self):
        seq_alpha, seq_beta = self.post_distributions.concentration1, self.post_distributions.concentration0

        weights = F.softmax(self.prior_weights, dim=0)
        sig_alpha_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        sig_beta_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        for idx, prior_dist in enumerate(self.prior_dists):
            sig_alpha, sig_beta = prior_dist.concentration1, prior_dist.concentration0
            sig_alpha_total = sig_alpha_total + sig_alpha * weights[idx]
            sig_beta_total = sig_beta_total + sig_beta * weights[idx]

        # merge two distribution
        if self.config.only_x_sig:
            x_alpha = sig_alpha_total
            x_beta = sig_beta_total
        else:
            x_alpha = seq_alpha + sig_alpha_total  # make value valid, so no -1
            x_beta = seq_beta + sig_beta_total
        x_alpha = x_alpha * self.config.z_scale
        x_beta = x_beta * self.config.z_scale
        self.z_distribution = dist.Beta(x_alpha, x_beta)

    def posterior_dist(self, logit, eps=1e-8):
        alpha_logits = logit[..., 1]  # alpha, dist to 1
        beta_logits = logit[..., 0]  # beta, dist to 0

        alpha = F.softplus(alpha_logits) + self.config.beta_min
        beta = F.softplus(beta_logits) + self.config.beta_min

        if self.config.post_dist == 'kuma':
            self.post_distributions = dist.kumaraswamy.Kumaraswamy(alpha, beta)
        elif self.config.post_dist == 'beta':
            self.post_distributions = dist.Beta(alpha + eps, beta + eps)

    def prior_dist(self, signals, eps=1e-8, peak_mask=None):
        if self.config.prior_signal == 'h3k27ac':
            prior_signal = signals[...,0].unsqueeze(-1)
        elif self.config.prior_signal == 'DHS':
            prior_signal = signals[...,1].unsqueeze(-1)
        elif self.config.prior_signal == 'hic':
            prior_signal = signals[...,2].unsqueeze(-1)
        elif self.config.prior_signal == 'all':
            prior_signal = signals
        signal_dim = prior_signal.shape[-1]
        self.prior_dists = []
        for i in range(signal_dim):
            cur_alpha = (prior_signal[...,i] + self.config.beta_min) * self.config.prior_scale_factor
            if self.config.max_pool_size > 0:
                smooth_max(cur_alpha, window_size=self.config.max_pool_size)
            # # merge peak mask
            # if self.config.merge_peak_mask:
            #     cur_alpha = torch.where(peak_mask < 0.1, 0.0, cur_alpha)
            distribution = dist.Beta(cur_alpha + eps, (self.signal_betas[i] + self.config.beta_min) * self.config.prior_scale_factor)
            self.prior_dists.append(distribution)

        # add the includelist, regard as a new signal
        if self.config.use_include_list and (not self.config.mask_region_hard):
            include_alpha = self.includelist.to(signals.dtype) * self.config.include_alpha
            include_dist = dist.Beta(include_alpha + eps, eps)
            self.prior_dists.append(include_dist)

    def forward(
        self,
        seqs,
        signals,
        mask_regions=None,
        bio_mask=None,
        peak_mask=None,
        rna_feat=None,
    ):
        if self.config.merge_peak_mask:
            expanded_mask = peak_mask.unsqueeze(-1)
            signals = signals * expanded_mask

        self.includelist, self.blacklist = mask_regions[...,0], mask_regions[...,1]
        bs, seq_len, _ = seqs.shape
        seq_input_embeds = self.seq_input_layer(seqs)
        inputs_embeds = seq_input_embeds
        if self.config.interact == 'concat':
            signals = signals[..., :self.config.signal_size]
            signal_input_embeds = self.signal_input_layer(signals)
            if self.config.gen_signal:
                inputs_embeds = seq_input_embeds + signal_input_embeds

        assert not (self.config.use_bio_mask and self.config.use_peak_mask)
        # generator
        outputs = self.generator(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=False,
        )
        logits = self.mask_output(outputs)
        # calculate the mask distribution by seqs & signals
        self.posterior_dist(logits)
        self.prior_dist(signals, peak_mask=peak_mask)

        if self.config.decouple_x:
            self.couple_post_dist()
            z_dist = self.z_distribution
        else:
            z_dist = self.post_distributions

        if self.config.post_sample:
            soft_mask = z_dist.mean
            if self.config.test_top:
                top_num = int(seq_len * self.config.test_top_percent)
                _, indices = torch.topk(soft_mask, top_num, dim=-1)
                hard_mask = torch.zeros_like(soft_mask)
                batch_indices = torch.arange(bs).unsqueeze(-1).expand_as(indices)
                hard_mask[batch_indices, indices] = 1.0
                mask = hard_mask
            elif self.config.test_soft:
                mask = soft_mask
            else:
                if self.config.pool_mask != 0:
                    soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                hard_mask = (soft_mask >= self.config.sample_threshold).float()
                mask = hard_mask
        else:
            mask = F.gumbel_softmax(logits, tau=self.config.gumbel_temp, hard=True, dim=-1)[:,:,1]
        # includelist and blacklist
        mask[self.blacklist] = 0.0
        if self.config.use_include_list and self.config.mask_region_hard:
            mask[self.includelist] = 1.0

        # middle promoter length = 1
        start = (seq_len - self.promo_len) // 2
        mask[:, start:start + self.promo_len] = 1.0
        return mask


class GeneBiMambaMIRNP(CaduceusPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.promo_len = config.center_len
        assert not (config.test_top and config.test_soft)
        assert not (config.use_bio_mask and config.use_peak_mask)
        assert (config.test_soft + config.test_hard + config.test_top) <= 1

        # input layer
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model)
        if config.interact == 'concat':
            self.signal_input_layer = nn.Linear(config.signal_size, config.d_model)

        # position encoding
        if config.pos_enc:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_range, config.d_model))

        if config.use_mask_model:
            self.mask_model = ModelMask(config, **kwargs)
            checkpoint = torch.load(config.mask_model)
            new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            load = self.mask_model.load_state_dict(new_state_dict, strict=False)
            assert len(load.missing_keys) == 0
            for param in self.mask_model.parameters():
                param.requires_grad = False

        elif (not self.config.use_bio_mask) and (not self.config.use_peak_mask):
            # generator
            gen_config = copy.deepcopy(config)
            gen_config.n_layer = config.gen_n_layer
            self.generator = Caduceus(gen_config, **{'ignore_embed_layer': True})

            # prior input weight - signal
            prior_weight = [float(x) for x in config.prior_weight.split(',')]
            prior_weight = [x / sum(prior_weight) for x in prior_weight]
            prior_weight = torch.log(torch.tensor(prior_weight, dtype=torch.float32))
            self.prior_weights = nn.Parameter(prior_weight, requires_grad=config.dist_param_grad)

            # signal beta
            signal_beta = [float(x) for x in config.prior_beta.split(',')]
            signal_beta = torch.tensor(signal_beta, dtype=torch.float32)
            self.signal_betas = nn.Parameter(signal_beta, requires_grad=config.dist_param_grad)

            # mask output
            self.mask_output = nn.Linear(config.d_model, 2)

            # remove grad
            if config.only_x_sig:
                for param in self.generator.parameters():
                    param.requires_grad = False
                for param in self.mask_output.parameters():
                    param.requires_grad = False

        # encoder
        if config.enc_prd_ps:
            # parameter sharing between encoder and predictor
            self.encoder = self.generator
        else:
            enc_config = copy.deepcopy(config)
            enc_config.n_layer = config.enc_n_layer
            self.encoder = Caduceus(enc_config, **{'ignore_embed_layer': True})

        # output
        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        # marginal distribution p(z)
        marginal_mean = torch.tensor([config.marginal_mean])
        self.marginal_beta = (1 - marginal_mean) / marginal_mean
        self.marginal_alpha = torch.tensor([1.0])

        # Initialize weights and apply final processing
        self.post_init()

    def couple_post_dist(self):
        seq_alpha, seq_beta = self.post_distributions.concentration1, self.post_distributions.concentration0

        weights = F.softmax(self.prior_weights, dim=0)
        sig_alpha_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        sig_beta_total = torch.tensor(0.0, dtype=seq_alpha.dtype, device=seq_alpha.device)
        for idx, prior_dist in enumerate(self.prior_dists):
            sig_alpha, sig_beta = prior_dist.concentration1, prior_dist.concentration0
            sig_alpha_total = sig_alpha_total + sig_alpha * weights[idx]
            sig_beta_total = sig_beta_total + sig_beta * weights[idx]

        # merge two distribution
        if self.config.only_x_sig:
            x_alpha = sig_alpha_total
            x_beta = sig_beta_total
        else:
            x_alpha = seq_alpha + sig_alpha_total  # make value valid, so no -1
            x_beta = seq_beta + sig_beta_total
        x_alpha = x_alpha * self.config.z_scale
        x_beta = x_beta * self.config.z_scale
        self.z_distribution = dist.Beta(x_alpha, x_beta)

    def posterior_dist(self, logit, eps=1e-8):
        alpha_logits = logit[..., 1]  # alpha, dist to 1
        beta_logits = logit[..., 0]  # beta, dist to 0

        alpha = F.softplus(alpha_logits) + self.config.beta_min
        beta = F.softplus(beta_logits) + self.config.beta_min

        if self.config.post_dist == 'kuma':
            self.post_distributions = dist.kumaraswamy.Kumaraswamy(alpha, beta)
        elif self.config.post_dist == 'beta':
            self.post_distributions = dist.Beta(alpha + eps, beta + eps)

    def prior_dist(self, signals, eps=1e-8, peak_mask=None):
        if self.config.prior_signal == 'h3k27ac':
            prior_signal = signals[...,0].unsqueeze(-1)
        elif self.config.prior_signal == 'DHS':
            prior_signal = signals[...,1].unsqueeze(-1)
        elif self.config.prior_signal == 'hic':
            prior_signal = signals[...,2].unsqueeze(-1)
        elif self.config.prior_signal == 'all':
            prior_signal = signals
        signal_dim = prior_signal.shape[-1]
        self.prior_dists = []
        for i in range(signal_dim):
            cur_alpha = (prior_signal[...,i] + self.config.beta_min) * self.config.prior_scale_factor
            if self.config.max_pool_size > 0:
                smooth_max(cur_alpha, window_size=self.config.max_pool_size)
            # # merge peak mask
            # if self.config.merge_peak_mask:
            #     cur_alpha = torch.where(peak_mask < 0.1, 0.0, cur_alpha)
            distribution = dist.Beta(cur_alpha + eps, (self.signal_betas[i] + self.config.beta_min) * self.config.prior_scale_factor)
            self.prior_dists.append(distribution)

        # add the includelist, regard as a new signal
        if self.config.use_include_list and (not self.config.mask_region_hard):
            include_alpha = self.includelist.to(signals.dtype) * self.config.include_alpha
            include_dist = dist.Beta(include_alpha + eps, eps)
            self.prior_dists.append(include_dist)

    def kl_divergence(self):
        if self.config.decouple_x:
            marginal_alpha = self.marginal_alpha * self.config.marginal_scale
            marginal_beta = self.marginal_beta * self.config.marginal_scale
            marginal_alpha = marginal_alpha.to(self.prior_weights.device)
            marginal_beta = marginal_beta.to(self.prior_weights.device)
            marginal_z_dist = dist.Beta(marginal_alpha, marginal_beta)
            kl_loss = dist.kl_divergence(self.z_distribution, marginal_z_dist)
            kl_loss = torch.mean(torch.mean(kl_loss, dim=1))
            return kl_loss

        weights = F.softmax(self.prior_weights, dim=0)

        post_dist = self.post_distributions
        prior_dists = self.prior_dists
        kl_loss_total = torch.tensor(0.0, dtype=self.prior_weights.dtype, device=self.prior_weights.device)

        for idx, prior_dist in enumerate(prior_dists):
            if self.config.post_dist == 'beta':
                kl_loss = dist.kl_divergence(post_dist, prior_dist)
            elif self.config.post_dist == 'kuma':
                kl_loss = kldivergence_kuma(post_dist, prior_dist.concentration1, prior_dist.concentration0)
            else:
                raise NotImplementedError()
            kl_loss = torch.mean(torch.mean(kl_loss, dim=1))
            kl_loss_total = kl_loss_total + weights[idx] * kl_loss
        return kl_loss_total

    def aux_loss(self, mask):
        kl_loss = self.kl_divergence() if self.config.aux_loss_kl else None
        l_padded_mask = torch.cat([mask[:,0].unsqueeze(1), mask], dim=1)
        r_padded_mask = torch.cat([mask, mask[:,-1].unsqueeze(1)], dim=1)
        continuity_cost = torch.mean(torch.mean(torch.abs(l_padded_mask - r_padded_mask), dim=1)) if self.config.aux_loss_con else None
        aux_loss = {
            'kl_loss': kl_loss,
            'continuity_loss': continuity_cost,
        }
        return aux_loss

    def forward(
        self,
        seqs,
        signals,
        mask_regions=None,
        bio_mask=None,
        peak_mask=None,
        rna_feat=None,
    ):
        if self.config.merge_peak_mask:
            expanded_mask = peak_mask.unsqueeze(-1)
            signals = signals * expanded_mask

        self.includelist, self.blacklist = mask_regions[...,0], mask_regions[...,1]
        bs, seq_len, _ = seqs.shape
        seq_input_embeds = self.seq_input_layer(seqs)
        if self.config.pos_enc:
            # add positional embedding
            seq_input_embeds = seq_input_embeds + self.pos_emb
        inputs_embeds_enc = inputs_embeds = seq_input_embeds
        if self.config.interact == 'concat':
            signals = signals[..., :self.config.signal_size]
            signal_input_embeds = self.signal_input_layer(signals)
            inputs_embeds_enc = seq_input_embeds + signal_input_embeds
            if self.config.gen_signal or self.config.enc_prd_ps:
                inputs_embeds = seq_input_embeds + signal_input_embeds

        if self.config.use_bio_mask:
            mask = bio_mask[...,1]
            aux_infor = {
                'mask': mask,
            }
        elif self.config.use_peak_mask:
            mask = peak_mask
            aux_infor = {
                'mask': mask,
            }
        elif self.config.use_mask_model:
            mask = self.mask_model(seqs=seqs, signals=signals, mask_regions=mask_regions, bio_mask=bio_mask, peak_mask=peak_mask, rna_feat=rna_feat)
            top_num = int(self.config.top_mask_percent * seq_len)
            topk_indices = torch.topk(mask, top_num, dim=1).indices
            binary_mask = torch.zeros_like(mask)
            binary_mask.scatter_(1, topk_indices, 1)
            mask = binary_mask
            aux_infor = {
                'mask': mask,
            }
        else:
            # generator
            outputs = self.generator(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=False,
                return_dict=False,
            )
            logits = self.mask_output(outputs)
            # calculate the mask distribution by seqs & signals
            self.posterior_dist(logits)
            self.prior_dist(signals, peak_mask=peak_mask)

            if self.config.decouple_x:
                self.couple_post_dist()
                z_dist = self.z_distribution
            else:
                z_dist = self.post_distributions

            if self.config.post_sample:
                if not self.training:
                    soft_mask = z_dist.mean
                    if self.config.test_top:
                        top_num = int(seq_len * self.config.test_top_percent)
                        _, indices = torch.topk(soft_mask, top_num, dim=-1)
                        hard_mask = torch.zeros_like(soft_mask)
                        batch_indices = torch.arange(bs).unsqueeze(-1).expand_as(indices)
                        if self.config.test_top_soft:
                            hard_mask[batch_indices, indices] = soft_mask[batch_indices, indices]
                        else:
                            hard_mask[batch_indices, indices] = 1.0
                        mask = hard_mask
                    elif self.config.test_soft:
                        mask = soft_mask
                    elif self.config.test_hard:
                        if self.config.pool_mask != 0:
                            soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                        hard_mask = (soft_mask >= self.config.sample_threshold).float()
                        mask = hard_mask
                else:
                    soft_mask = z_dist.rsample()
                    if self.config.pool_mask != 0:
                        soft_mask = smooth_max(soft_mask, self.config.pool_mask)
                    if self.config.post_hard_dist:
                        hard_mask = (soft_mask >= self.config.sample_threshold).float()
                        mask = hard_mask - soft_mask.detach() + soft_mask  # differentiable hard mask
                    else:
                        mask = soft_mask.clone()
            else:
                mask = F.gumbel_softmax(logits, tau=self.config.gumbel_temp, hard=True, dim=-1)[:,:,1]
            # includelist and blacklist
            mask[self.blacklist] = 0.0
            if self.config.use_include_list and self.config.mask_region_hard:
                mask[self.includelist] = 1.0

            # middle promoter length = 1
            start = (seq_len - self.promo_len) // 2
            mask[:, start:start + self.promo_len] = 1.0
            # get aux loss
            aux_infor = self.aux_loss(mask=mask)
            aux_infor['mask'] = mask

        # encoder
        if self.config.pos_enc:
            valid_counts = mask.sum(dim=1)
            max_valid_len = valid_counts.max()
            padded_inputs_embeds_enc = torch.full((bs, int(max_valid_len), self.config.d_model), 0,
                                                  dtype=inputs_embeds_enc.dtype, device=inputs_embeds_enc.device)
            range_tensor = torch.arange(int(max_valid_len), device=inputs_embeds_enc.device).unsqueeze(0)
            valid_positions = range_tensor < valid_counts.unsqueeze(1)
            padded_inputs_embeds_enc[valid_positions] = inputs_embeds_enc[mask == 1]
            inputs_embeds_enc = padded_inputs_embeds_enc
        else:
            inputs_embeds_enc = inputs_embeds_enc * mask.unsqueeze(-1)
        outputs_enc = self.encoder(
            input_ids=None,
            inputs_embeds=inputs_embeds_enc,
            output_hidden_states=False,
            return_dict=False,
        )

        # output
        hidden_states = outputs_enc
        if self.config.pos_enc and self.config.center_len:
            valid_counts_left = mask[:,:seq_len//2].sum(dim=1) - self.promo_len // 2
            promo_indices = valid_counts_left.unsqueeze(1) + torch.arange(self.promo_len,
                                                                          device=outputs_enc.device).unsqueeze(0)
            batch_indices = promo_indices.unsqueeze(-1).expand(-1, -1, outputs_enc.shape[-1]).to(torch.int64)
            hidden_states = outputs_enc.gather(1, batch_indices)

        elif self.config.center_len:
            start_index = (hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            hidden_states = hidden_states[:, start_index: end_index, :]

        hidden_states = torch.mean(hidden_states, dim=1)

        p_embed = torch.cat([hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else hidden_states
        logits = self.pToExpr(p_embed)

        logits = logits.float()

        return logits, aux_infor, mask
