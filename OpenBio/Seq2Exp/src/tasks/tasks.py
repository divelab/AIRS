import inspect
from typing import List

import torch.nn as nn
from einops import rearrange

import src.models.nn.utils as U
import src.tasks.metrics as M
import torchmetrics as tm
from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from src.tasks.custom_torchmetrics import torchmetric_fns as tm_mine
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection


class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)  # partially fix param
        self.loss = U.discard_kwargs(self.loss)  # ignore unused param in function, although pass it
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False
                )
            elif name in ['MultilabelAUROC', 'MultilabelAveragePrecision']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_labels=self.dataset.d_output
                )
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k
                )
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x, w = encoder(x, **z)
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


class AdaptiveLMTask(BaseTask):
    def __init__(
            self,
            div_val,
            cutoffs: List[int],
            tie_weights: bool,
            tie_projs: List[bool],
            init_scale=1.0,
            bias_scale=0.0,
            dropemb=0.0,
            dropsoft=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


class PEInterTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        input_PE, input_feat, input_dist, y_expr, eid = batch
        pred_expr, _ = model(input_PE, input_feat, input_dist)
        return pred_expr, y_expr, {}


class GeneExpTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        seqs, bio_mask, peak_mask, rna_feat, signals, mask_regions, y_expr, eid = batch
        pred_expr = model(seqs=seqs, signals=signals, mask_regions=mask_regions, bio_mask=bio_mask,
                          peak_mask=peak_mask, rna_feat=rna_feat)
        return pred_expr, y_expr, {}


class ExtractRationale(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        seqs, bio_mask, peak_mask, rna_feat, signals, mask_regions, y_expr, eid = batch
        pred_expr, aux_infor, mask = model(seqs=seqs, signals=signals, mask_regions=mask_regions, bio_mask=bio_mask,
                                           peak_mask=peak_mask, rna_feat=rna_feat)
        return pred_expr, y_expr, aux_infor


registry = {
    'base': BaseTask,
    'pe_inter': PEInterTask,
    'gene_expression': GeneExpTask,
    'extract_rationale': ExtractRationale,
}
