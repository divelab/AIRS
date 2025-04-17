"""Caduceus config for Hugging Face.

"""

from typing import Optional, Union
from dataclasses import dataclass
from transformers import PretrainedConfig

from mamba_ssm.models.config_mamba import MambaConfig


@dataclass
class ExtendedMambaConfig(MambaConfig):
    interact: str = ""
    use_bio_mask: bool = False
    base_size: int = 4
    signal_size: int = 3
    center_len: int = 2000
    rna_feat_dim: int = 9
    useRNAFeat: bool = True
    seq_range: int = 0


class CaduceusConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""
    model_type = "caduceus"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 2560,
            n_layer: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,
            seq_range: int = 0,
            # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
            norm_epsilon: float = 1e-5,

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # Caduceus-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            rcps: bool = False,
            complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead

            # gene expression
            interact: str = 'concat',
            base_size: int = 4,
            signal_size: int = 2,
            center_len: int = 0,
            rna_feat_dim: int = 9,
            useRNAFeat: bool = True,

            # RNP encoder (EPInformer)
            encoder: Optional[dict] = None,

            # bimamba RNP
            gen_n_layer: int = 1,
            enc_n_layer: int = 1,
            down_n_layer: int = 1,
            pretrained_mamba: bool = False,
            pretrained_model_name: str = '',
            pretrained_freeze: bool = False,
            gumbel_hard_train: bool = False,

            # extract rationale loss
            aux_loss_select: bool = True,
            aux_loss_con: bool = True,
            aux_loss_kl: bool = True,

            # use epinformer, bio mask
            use_bio_mask: bool = False,
            use_peak_mask: bool = False,

            # gumbel params RNP
            mask_threshold: float = 0.0,
            mv_size: int = 1,
            bio_mask_weight: float = 0.0,
            counter_zero: bool = True,
            gumbel_temp: float = 0.0,

            # mask
            # merge_mask: bool = False,
            subseq_size: int = 1000,
            node_merge_mask: bool = False,
            node_merge_range: int = 100,
            merge_peak_mask: bool = True,
            max_pool_size: int = 0,

            # incorporate signals
            signal_incor: str = '',
            signal_incor_hidden: int = 0,
            seq_gumbel_temp: float = 1.0,
            seq_gumbel_merge: int = 500,

            # two stage mask
            seq_mask: bool = True,
            signal_mask: bool = True,

            # MI RNP
            prior_signal: str = '',
            prior_beta: str = '',
            prior_weight: str = '',
            gen_signal: bool = True,
            prior_scale_factor: float = 1.0,
            post_sample: bool = False,
            sample_threshold: float = 0.5,
            post_dist: str = '',
            post_hard_dist: bool = True,

            # decouple x
            decouple_x: bool = False,
            only_x_sig: bool = False,
            mask_region_hard: bool = False,
            include_alpha: float = 0.0,
            use_include_list: bool = False,

            # marginal dist of z
            marginal_mean: float = 0.0,
            marginal_scale: float = 0.0,

            dist_param_grad: bool = False,

            # learned mask
            use_mask_model: bool = False,
            mask_model: str = '',

            # ps btw encoder & predictor
            enc_prd_ps: bool = False,

            # pos encoding
            pos_enc: bool = False,

            # pool mask
            pool_mask: int = 0,
            merge_mask: int = 0,

            # beta min
            beta_min: float = 0.0,

            # z dist scale
            z_scale: float = 1.0,

            # test top %
            test_top: bool = False,
            test_top_percent: float = 0.0,
            test_top_soft: bool = False,
            test_soft: bool = False,
            test_hard: bool = False,
            top_mask_percent: float = 0.0,

            use_gumbel: bool = True,
            beta_step: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
        self.interact = interact
        self.base_size = base_size
        self.signal_size = signal_size
        self.center_len = center_len
        self.rna_feat_dim = rna_feat_dim
        self.useRNAFeat = useRNAFeat
        self.encoder = encoder
        self.gen_n_layer = gen_n_layer
        self.enc_n_layer = enc_n_layer
        self.down_n_layer = down_n_layer
        self.aux_loss_select = aux_loss_select
        self.aux_loss_con = aux_loss_con
        self.pretrained_mamba = pretrained_mamba
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_freeze = pretrained_freeze
        self.gumbel_hard_train = gumbel_hard_train
        self.use_bio_mask = use_bio_mask
        self.mask_threshold = mask_threshold
        self.mv_size = mv_size
        self.bio_mask_weight = bio_mask_weight
        self.counter_zero = counter_zero
        self.merge_mask = merge_mask
        self.subseq_size = subseq_size
        self.signal_incor = signal_incor
        self.signal_incor_hidden = signal_incor_hidden
        self.seq_gumbel_temp = seq_gumbel_temp
        self.seq_gumbel_merge = seq_gumbel_merge
        self.node_merge_mask = node_merge_mask
        self.node_merge_range = node_merge_range
        self.seq_mask = seq_mask
        self.signal_mask = signal_mask
        self.prior_signal = prior_signal
        self.prior_beta = prior_beta
        self.prior_weight = prior_weight
        self.gumbel_temp = gumbel_temp
        self.aux_loss_kl = aux_loss_kl
        self.gen_signal = gen_signal
        self.prior_scale_factor = prior_scale_factor
        self.post_sample = post_sample
        self.sample_threshold = sample_threshold
        self.post_dist = post_dist
        self.post_hard_dist = post_hard_dist
        self.decouple_x = decouple_x
        self.only_x_sig = only_x_sig
        self.mask_region_hard = mask_region_hard
        self.include_alpha = include_alpha
        self.marginal_mean = marginal_mean
        self.marginal_scale = marginal_scale
        self.use_peak_mask = use_peak_mask
        self.merge_peak_mask = merge_peak_mask
        self.use_include_list = use_include_list
        self.max_pool_size = max_pool_size
        self.dist_param_grad = dist_param_grad
        self.use_mask_model = use_mask_model
        self.mask_model = mask_model
        self.enc_prd_ps = enc_prd_ps
        self.pos_enc = pos_enc
        self.seq_range = seq_range
        self.pool_mask = pool_mask
        self.merge_mask = merge_mask
        self.beta_min = beta_min
        self.z_scale = z_scale
        self.test_top = test_top
        self.test_top_percent = test_top_percent
        self.test_soft = test_soft
        self.test_hard = test_hard
        self.top_mask_percent = top_mask_percent
        self.use_gumbel = use_gumbel
        self.beta_step = beta_step
        self.test_top_soft = test_top_soft
