# Inspired by https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/metrics/perplexity.py
# But we compute the perplexity correctly: exp(average(nll)), not average(exp(nll))
# Also adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/perplexity.py
# But we pass in the loss to avoid recomputation

from typing import Any, Dict, Optional

import yaml
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import re
import pandas as pd
from src.tasks.sei import Sei, NonStrandSpecficEmbed
from src.tasks.enhancer_models import CNNModel
from src.tasks.utils import (upgrade_state_dict, convert_batch_one_hot, get_wasserstein_dist,
                             calculate_weighted_category_diversity)

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

try:
    from apex.transformer import parallel_state
except ImportError:
    parallel_state = None


class EnhancerClassifier:
    def __init__(self, classifier_hparams, classifier_path, num_classes):
        with open(classifier_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.classifier_model = CNNModel(hparams['args'], alphabet_size=4,
                                         num_cls=num_classes, classifier=True)
        self.classifier_model.load_state_dict(
            upgrade_state_dict(
                torch.load(classifier_path,
                           map_location=torch.device("cpu"))['state_dict'],
                prefixes=['model.']
            )
        )
        self.classifier_model.eval()
        self.classifier_model.to(torch.device("cpu"))
        for param in self.classifier_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_embed(self, input_tensor, device):
        self.classifier_model = self.classifier_model.to(device)
        input_tensor = input_tensor[:, 1:-1]
        ori_one_hot = convert_batch_one_hot(input_tensor)
        ori_argmax = torch.argmax(ori_one_hot, dim=-1)
        _, ori_embeddings = self.classifier_model(ori_argmax, t=None, return_embedding=True)
        return ori_embeddings


classifier_fb = EnhancerClassifier(
    classifier_hparams="./workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml",
    classifier_path="./workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt",
    num_classes=81,
)

classifier_mel = EnhancerClassifier(
    classifier_hparams="./workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml",
    classifier_path="./workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt",
    num_classes=47,
)


class HyenaDNA:
    def __init__(self,
                 pretrained_model_name='hyenadna-tiny-1k-seqlen',
                 next_token=True,
                 ignore_index=4,
                 ):
        from src.models.sequence.hyenaDNA import HyenaDNAPreTrainedModel

        '''
        this selects which backbone to use, and grabs weights/ config from HF
        4 options:
          'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
          'hyenadna-small-32k-seqlen'
          'hyenadna-medium-160k-seqlen'  # inference only on colab
          'hyenadna-medium-450k-seqlen'  # inference only on colab
          'hyenadna-large-1m-seqlen'  # inference only on colab
        '''
        max_lengths = {
            'hyenadna-tiny-1k-seqlen': 1024,
            'hyenadna-small-32k-seqlen': 32768,
            'hyenadna-medium-160k-seqlen': 160000,
            'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
            'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
        }

        self.max_length = max_lengths[pretrained_model_name]  # auto selects
        self.ignore_index = ignore_index

        # data settings:
        use_padding = True
        rc_aug = False  # reverse complement augmentation
        add_eos = False  # add end of sentence token

        # we need these for the decoder head, if using
        use_head = False
        n_classes = 2  # not used for embeddings only

        # you can override with your own backbone config here if you want,
        # otherwise we'll load the HF one in None
        backbone_cfg = None

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                     'hyenadna-small-32k-seqlen',
                                     'hyenadna-medium-160k-seqlen',
                                     'hyenadna-medium-450k-seqlen',
                                     'hyenadna-large-1m-seqlen']:
            # use the pretrained Huggingface wrapper instead
            self.model = HyenaDNAPreTrainedModel.from_pretrained(
                './checkpoints',
                pretrained_model_name,
                download=True,
                config=backbone_cfg,
                # device=device,
                use_head=use_head,
                n_classes=n_classes,
                next_token=next_token,
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def get_score(self, tok_seq, device):
        self.model = self.model.to(device)

        input_ids = tok_seq[:, :-1].contiguous()
        targets = tok_seq[:, 1:].contiguous()

        with torch.no_grad():
            logits = self.model(input_ids)

        # TODO: remove last token (EOS/SEP token), since huge loss
        mask = targets != self.ignore_index
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
        return (loss * mask.view(-1)), mask


hyena_pretrained_model = HyenaDNA()


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.
    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("total_log_probs", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

        self.loss_fn = CrossEntropyLoss()

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.
        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        count = target.numel()
        if loss is None:
            loss = self.loss_fn(preds, target)
        self.total_log_probs += loss.double() * count
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.
        Returns:
           Perplexity
        """
        return torch.exp(self.total_log_probs / self.count)


class NumTokens(Metric):
    """Keep track of how many tokens we've seen.
    """
    # TODO: how do we prevent the reset between the epochs? The reset happens on the 1st batch
    # of the next epoch.
    # Right now the hack is that we override reset(), which would mess up the forward method.
    # We then override forward to do the right thing.

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum",
                       persistent=True)  # We want the count to be saved to state-dict
        if parallel_state is not None and not parallel_state.is_unitialized():
            self.tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            self.tensor_parallel_world_size = 1

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:  # type: ignore
        self.count += target.numel() // self.tensor_parallel_world_size

    def compute(self) -> Tensor:
        return self.count

    def reset(self):
        count = self.count
        super().reset()
        self.count = count

    # Adapted from https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/metric.py
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """forward computation using single call to `update` to calculate the metric value on the current batch and
        accumulate global state.
        This can be done when the global metric state is a sinple reduction of batch states.
        """
        self.update(*args, **kwargs)
        return self.compute()


class SeiInitial:
    def __init__(self):
        self.sei = NonStrandSpecficEmbed(Sei(4096, 21907))
        self.sei.load_state_dict(upgrade_state_dict(
            torch.load('./data/promoter_design/best.sei.model.pth.tar', map_location='cpu')['state_dict'],
            prefixes=['module.']))
        self.sei.to(torch.device('cpu'))
        self.seifeatures = pd.read_csv('./data/promoter_design/target.sei.names', sep='|', header=None)
        self.sei.eval()
        for param in self.sei.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_sei_profile(self, seq_id, device, return_embed=False):
        self.sei = self.sei.to(device)
        seq_one_hot = convert_batch_one_hot(seq_id)
        B, L, K = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=device) * 0.25], 2) # batchsize x 4 x 4,096
        sei_out, sei_embed = self.sei(sei_inp, return_embed=return_embed) # batchsize x 21,907
        # sei_out = sei_out.cpu().detach().numpy()
        sei_out = sei_out[:, self.seifeatures[1].str.strip().values == 'H3K4me3'] # batchsize x 2,350
        predh3k4me3 = sei_out.mean(dim=1) # batchsize

        if return_embed:
            # sei_embed = sei_embed.cpu().detach().numpy()
            return predh3k4me3, sei_embed
        return predh3k4me3, None

    @torch.no_grad()
    def get_sei_profile_any(self, seq_id, protein_name, device, return_embed=False, cell_type=None):
        self.sei = self.sei.to(device)
        seq_one_hot = convert_batch_one_hot(seq_id)
        B, L, K = seq_one_hot.shape
        sei_inp = torch.cat([torch.ones((B, 4, 1536), device=device) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536), device=device) * 0.25], 2)
        sei_out, sei_embed = self.sei(sei_inp, return_embed=False) # batchsize x 21,907
        # sei_out = sei_out.cpu().detach().numpy()

        ans_list = []
        for each_b in range(B):
            require = ((self.seifeatures[0].str.strip().values == cell_type[each_b]) &
                       (self.seifeatures[1].str.strip().values == protein_name[each_b]) &
                       (self.seifeatures[2].str.strip().values == 'ENCODE'))
            ans = sei_out[each_b, require]
            ans = ans.mean(axis=0)
            ans_list.append(ans)

        return torch.stack(ans_list), None


sei_class = SeiInitial()


class SEIPromoter(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    mse: Tensor
    count: Tensor

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("mse", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:
        sei_profile, sei_embed = sei_class.get_sei_profile(target, self.device, False)
        sei_profile_pred, sei_embed_pred = sei_class.get_sei_profile(preds, self.device, False)

        self.mse += ((sei_profile - sei_profile_pred) ** 2).double().sum()
        self.count += len(target)  # number of samples

    def compute(self) -> Tensor:
        return self.mse / self.count


class Fluency(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    total_loss: Tensor
    count: Tensor

    def __init__(self,  **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.add_state("total_loss", default=torch.tensor(0.0, dtype=torch.float64),
                       dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        # self.pretrained_model = HyenaDNA()

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None) -> None:
        gen_loss, gen_mask = hyena_pretrained_model.get_score(preds, self.device)  # (bs*seq), bs * seq

        self.total_loss += gen_loss.double().sum()
        self.count += gen_mask.sum()  # number of tokens
        # print(f"current loss {gen_loss.double().sum()}, current count {gen_mask.sum()}, total_loss {self.total_loss}, total_count {self.count}")

    def compute(self) -> Tensor:
        return torch.exp(self.total_loss / self.count)


class FBD_fb(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Dict[str, Any]):
        kwargs['compute_on_cpu'] = True
        super().__init__(**kwargs)
        self.classifier_hparams = "../../../workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml"
        self.classifier_path = "../../../workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt"
        self.num_classes = 81
        self.init_enhancer_classifier()

        # self.ori_embed, self.gen_embed = [], []
        self.add_state("ori_embed", default=[], dist_reduce_fx="cat")
        self.add_state("gen_embed", default=[], dist_reduce_fx="cat")

    def init_enhancer_classifier(self):
        with open(self.classifier_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.classifier_model = CNNModel(hparams['args'], alphabet_size=4,
                                         num_cls=self.num_classes, classifier=True)
        self.classifier_model.load_state_dict(
            upgrade_state_dict(
                torch.load(self.classifier_path,
                           map_location=torch.device(self.device))['state_dict'],
                prefixes=['model.']
            )
        )
        self.classifier_model.eval()
        self.classifier_model = self.classifier_model.to(self.device)
        for param in self.classifier_model.parameters():
            param.requires_grad = False

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None, signal: Optional[Tensor] = None) -> None:
        preds = preds[:, 1:-1]
        target = target[:, 1:-1]
        with torch.no_grad():
            ori_one_hot = convert_batch_one_hot(target)
            ori_argmax = torch.argmax(ori_one_hot, dim=-1)
            _, ori_embeddings = self.classifier_model(ori_argmax, t=None, return_embedding=True)

            generated_one_hot = convert_batch_one_hot(preds)
            generated_argmax = torch.argmax(generated_one_hot, dim=-1)
            _, generated_embeddings = self.classifier_model(generated_argmax, t=None, return_embedding=True)

            self.ori_embed.append(ori_embeddings.detach())
            self.gen_embed.append(generated_embeddings.detach())

    def compute(self) -> Tensor:
        ori_embed = dim_zero_cat(self.ori_embed)
        gen_embed = dim_zero_cat(self.gen_embed)
        # ori_embed = torch.concat(self.ori_embed, dim=0)
        # gen_embed = torch.concat(self.gen_embed, dim=0)

        ori_embed_flat = ori_embed.view(-1, 128)
        gen_embed_flat = gen_embed.view(-1, 128)

        # TODO: make it np, not torch tensor
        fbd = get_wasserstein_dist(gen_embed_flat, ori_embed_flat)
        return fbd


class FBD_mel(FBD_fb):
    def __init__(self, **kwargs: Dict[str, Any]):
        kwargs['compute_on_cpu'] = True
        super().__init__(**kwargs)
        self.classifier_hparams = "../../../workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml"
        self.classifier_path = "../../../workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt"
        self.num_classes = 47
        self.init_enhancer_classifier()

        self.add_state("ori_embed", default=[], dist_reduce_fx="cat")
        self.add_state("gen_embed", default=[], dist_reduce_fx="cat")


class Diversity(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs: Dict[str, Any]):
        kwargs['compute_on_cpu'] = True
        super().__init__(**kwargs)
        self.add_state("list_seq", default=[], dist_reduce_fx="cat")
        self.add_state("list_key", default=[], dist_reduce_fx="cat")

        # self.class_generated_seq = {}

    def update(self, preds: Tensor, target: Tensor, loss: Optional[Tensor] = None,
               signal: Optional[Tensor] = None) -> None:

        self.list_seq.append(preds.detach())
        self.list_key.append(signal.detach())

        # for i in range(len(preds)):
        #     key_ = signal[i].item()
        #     value_ = preds[i].detach().cpu()
        #
        #     if key_ in self.class_generated_seq:
        #         self.class_generated_seq[key_].append(value_)
        #     else:
        #         self.class_generated_seq[key_] = [value_]

    def compute(self) -> Tensor:
        list_seq = dim_zero_cat(self.list_seq)
        list_key = dim_zero_cat(self.list_key)

        result_dict = {}
        for i in range(len(list_seq)):
            k = list_key[i].item()
            v = list_seq[i]
            if k in result_dict:
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]

        diversity_score = calculate_weighted_category_diversity(result_dict)
        return diversity_score


torchmetric_fns = {
    "perplexity": Perplexity,
    "num_tokens": NumTokens,

    "sei_promoter": SEIPromoter,
    "fluency": Fluency,

    "fbd_fb": FBD_fb,
    "fbd_mel": FBD_mel,
    "diversity": Diversity,
}
