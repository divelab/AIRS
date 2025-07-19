"""Main training entry point for pre-training and downstream fine-tuning.

"""

import json
import os
import random
import time
import math
from functools import wraps
from typing import Callable, List, Sequence

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from scipy.stats import ks_2samp
import numpy as np
from collections import defaultdict

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks
from src.tasks.custom_torchmetrics import hyena_pretrained_model, sei_class, classifier_fb, classifier_mel
from src.tasks.utils import (get_wasserstein_dist, calculate_weighted_category_diversity, percent_identity,
                             kmer_statistics, percent_identity_group)
from src.tasks.utils import valid_vocab, convert_batch_one_hot


# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))

log = src.utils.train.get_logger(__name__)


def cosine_schedule(t: torch.Tensor):
    # t is a tensor of size (batch_size,) with values between 0 and 1. This is the
    # schedule used in the MaskGIT paper
    return torch.cos(t * math.pi * 0.5)


def top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def sample_from_logits(logits, top_k, temp=1.0, greedy=False, return_prob=False, next_true_token=None):
    logits = top_k_logits(logits, top_k)
    probabilities = F.softmax(logits / temp, dim=-1)
    if next_true_token is not None:
        next_token = next_true_token
    else:
        if greedy:
            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probabilities, 1, replacement=True)
    next_token_prob = probabilities.gather(dim=-1, index=next_token)
    if return_prob:
        return next_token, next_token_prob
    return next_token, None


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level: access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        # To be set in `setup`
        self.encoder, self.decoder, self.model = None, None, None
        self.task, self.loss, self.loss_val = None, None, None
        self.metrics, self.train_torchmetrics, self.val_torchmetrics, self.test_torchmetrics = None, None, None, None
        self.setup()

        self._state = None
        self.val_loader_names, self.test_loader_names = None, None

        self.reset_torch_metrics()

    def reset_torch_metrics(self):
        if "promoter" in self.dataset.dataset_name:
            self.sei_mse_val = torch.tensor(0.0, dtype=torch.float64)
            self.sample_count_val = torch.tensor(0, dtype=torch.int64)
            self.fluency_loss_val = torch.tensor(0.0, dtype=torch.float64)
            self.count_val = torch.tensor(0, dtype=torch.int64)
            # Predictive distribution shift
            self.sei_activity_original_val, self.sei_activity_pred_val = [], []
            # percent identity
            self.percent_identity_max_val = torch.tensor(0.0, dtype=torch.float64)
            self.percent_identity_sum_val = torch.tensor(0.0, dtype=torch.float64)
            # k-mer spectrum shift
            # record whole generated sequence
            self.original_seq_val, self.generated_seq_val = [], []

            self.sei_mse_test = torch.tensor(0.0, dtype=torch.float64)
            self.sample_count_test = torch.tensor(0, dtype=torch.int64)
            self.fluency_loss_test = torch.tensor(0.0, dtype=torch.float64)
            self.count_test = torch.tensor(0, dtype=torch.int64)
            # Predictive distribution shift
            self.sei_activity_original_test, self.sei_activity_pred_test = [], []
            # percent identity
            self.percent_identity_max_test = torch.tensor(0.0, dtype=torch.float64)
            self.percent_identity_sum_test = torch.tensor(0.0, dtype=torch.float64)
            # k-mer spectrum shift
            # record whole generated sequence
            self.original_seq_test, self.generated_seq_test = [], []

        elif "enhancer" in self.dataset.dataset_name:
            self.fbd_ori_embed_val, self.fbd_ori_embed_test = [], []
            self.fbd_gen_embed_val, self.fbd_gen_embed_test = [], []
            self.fluency_loss_val, self.fluency_loss_test = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.count_val, self.count_test = torch.tensor(0, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)
            self.original_list_seq_val, self.original_list_seq_test = [], []
            self.diversity_list_seq_val, self.diversity_list_seq_test = [], []
            self.diversity_list_signal_val, self.diversity_list_signal_test = [], []

            self.uncond_fbd_gen_embed_val, self.uncond_fbd_gen_embed_test = [], []
            self.uncond_fluency_loss_val, self.uncond_fluency_loss_test = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.uncond_count_val, self.uncond_count_test = torch.tensor(0, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)
            self.uncond_diversity_list_seq_val, self.uncond_diversity_list_seq_test = [], []
            self.uncond_diversity_list_signal_val, self.uncond_diversity_list_signal_test = [], []

        elif "chipseq" in self.dataset.dataset_name:
            # sei
            self.sei_val, self.sei_mse_val = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.sample_count_val = torch.tensor(0, dtype=torch.int64)
            self.uncond_sei_val, self.uncond_sei_mse_val = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.real_sei_val = torch.tensor(0.0, dtype=torch.float64)
            self.rand_sei_val, self.rand_sei_mse_val = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)

            self.sei_test, self.sei_mse_test = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.sample_count_test = torch.tensor(0, dtype=torch.int64)
            self.uncond_sei_test, self.uncond_sei_mse_test = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
            self.real_sei_test = torch.tensor(0.0, dtype=torch.float64)
            self.rand_sei_test, self.rand_sei_mse_test = torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)

            # fluency
            self.fluency_loss_val = torch.tensor(0.0, dtype=torch.float64)
            self.count_val = torch.tensor(0, dtype=torch.int64)
            self.uncond_fluency_loss_val = torch.tensor(0.0, dtype=torch.float64)
            self.uncond_count_val = torch.tensor(0, dtype=torch.int64)
            self.real_fluency_loss_val = torch.tensor(0.0, dtype=torch.float64)
            self.real_count_val = torch.tensor(0, dtype=torch.int64)

            self.fluency_loss_test = torch.tensor(0.0, dtype=torch.float64)
            self.count_test = torch.tensor(0, dtype=torch.int64)
            self.uncond_fluency_loss_test = torch.tensor(0.0, dtype=torch.float64)
            self.uncond_count_test = torch.tensor(0, dtype=torch.int64)
            self.real_fluency_loss_test = torch.tensor(0.0, dtype=torch.float64)
            self.real_count_test = torch.tensor(0, dtype=torch.int64)

            # diversity
            self.original_list_seq_val, self.original_list_seq_test = [], []
            self.diversity_list_seq_val, self.diversity_list_seq_test = [], []
            self.uncond_diversity_list_seq_val, self.uncond_diversity_list_seq_test = [], []

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more
        # memory than the others.
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        if config_path is not None:
            with open(config_path) as f:
                model_config_from_file = json.load(f)
            self.hparams.model.update(model_config_from_file)
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})
        # TODO: Hacky way to get complement_map for Caduceus models; need to find a more elegant implementation
        if "caduceus" in self.hparams.model.get("_name_"):
            OmegaConf.update(
                self.hparams.model.config, "complement_map", self.dataset.tokenizer.complement_map, force_add=True
            )
        # Instantiate the config class if using hydra's _target_ paradigm for the config
        if self.hparams.model.get("config", None) is not None and self.hparams.model.config.get("_target_", None) is not None:
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # if self.hparams.train.get("compile_model", False):
        #     self.model = torch.compile(self.model, dynamic=False)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules, so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)

        log.info("Custom load_state_dict function is running.")

        # strict==True will require all modules to match
        # strict==False can allow encoder/decoder to be loaded from scratch too
        # state_dict = {k: v for k, v in state_dict.items() if "torchmetrics" not in k}
        return super().load_state_dict(state_dict, strict=strict)

    # def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
    #     state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    #     filtered_state_dict = {k: v for k, v in state_dict.items() if "torchmetrics" not in k}
    #     return filtered_state_dict

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
                (n := self.hparams.train.state.n_context) is None
                or isinstance(n, int)
                and n >= 0
        )
        assert (
                (n := self.hparams.train.state.n_context_eval) is None
                or isinstance(n, int)
                and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, training=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if training else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def device_gather(self, tensor_gather):
        if self.trainer.world_size > 1:
            tensor_gather = self.all_gather(tensor_gather)
            tensor_gather = tensor_gather.reshape(-1, *tensor_gather.shape[2:])
        return tensor_gather

    # def on_fit_start(self) -> None:
    #     if "chipseq" in self.dataset.dataset_name:
    #         self.trainer.validate(self)

    def _shared_step(self, batch, batch_idx, prefix="train"):
        """Shared step logic between training, validation, and test"""
        # prefix: train, val, test
        self._process_state(batch, batch_idx, training=(prefix == "train"))
        x, y, w = self.forward(batch)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        self.log_dict(
            metrics,
            # on_step=log_on_step,
            # on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        if prefix != 'train':
            if self.hparams.model._name_ == 'bert_lm':
                # bert model
                mapping_tensor = torch.tensor([valid_vocab['A'], valid_vocab['C'], valid_vocab['G'],
                                               valid_vocab['T'], valid_vocab['N']], device=self.device)

                input_embeds, condition, input_seqs = batch
                generated_id = self.iterative_gen(batch)

                generated_id = mapping_tensor[generated_id]
                original_id = mapping_tensor[input_seqs]

                if "enhancer" in self.dataset.dataset_name:
                    empty_class = self.model.config.num_classes
                    bs = original_id.shape[0]
                    # empty_list = [empty_class for _ in range(bs)]
                    # empty_cond = torch.tensor(empty_list, dtype=condition.dtype, device=condition.device)
                    uncond_signal = torch.full((bs,), empty_class, dtype=torch.long).to(self.device)
                    input_embeds_no_cond = input_embeds.clone()
                    input_embeds_no_cond[:,:,-1] = uncond_signal.unsqueeze(-1)

                    new_batch = (input_embeds_no_cond, condition, input_seqs)
                    uncond_generated_id = self.iterative_gen(new_batch)
                    uncond_generated_id = mapping_tensor[uncond_generated_id]

                    cond_signal = input_embeds[:,0,-1].clone().long()
                    # cond_signal = condition

            elif self.hparams.model._name_ == 'transformer_lm':
                # generate sequences
                # data_, target_, _ = batch
                data_, target_, _ = batch['data'], batch['target'], batch['condition']
                original_id = torch.cat([data_, target_[:, -1:]], dim=1)
                batch_size = target_.shape[0]

                # generate sequence on original conditions
                generated_id = self.autoregressive_gen(batch_size, batch)
                assert generated_id.shape == original_id.shape

                if "enhancer" in self.dataset.dataset_name:
                    empty_class = self.model.config.num_classes
                    uncond_signal = torch.full((batch_size,), empty_class, dtype=torch.long).to(self.device)
                    # uncond_batch = (batch[0], batch[1], uncond_signal)
                    uncond_batch = {
                        "data": batch['data'],
                        "target": batch['target'],
                        "condition": uncond_signal,
                    }
                    uncond_generated_id = self.autoregressive_gen(batch_size, uncond_batch)
                    assert uncond_generated_id.shape == original_id.shape
                    uncond_generated_id = self.device_gather(uncond_generated_id)
                    uncond_signal = self.device_gather(uncond_signal)
                    cond_signal = self.device_gather(batch["condition"])
                elif "chipseq" in self.dataset.dataset_name:
                    # un-conditional generation
                    uncond_protein = torch.zeros_like(batch["condition"])
                    # uncond_batch = (batch[0], batch[1], uncond_protein)
                    uncond_batch = {
                        "data": batch['data'],
                        "target": batch['target'],
                        "condition": uncond_protein,
                        "cell_type": torch.zeros_like(batch['cell_type']) + self.model.config.num_classes,
                    }
                    uncond_generated_id = self.autoregressive_gen(batch_size, uncond_batch)
                    assert uncond_generated_id.shape == original_id.shape

                    # random sequence
                    bs, seq_len = original_id.shape
                    pad_token_id = self.dataset.tokenizer.pad_token_id
                    # rand_lengths = torch.randint(50, 1000, (batch_size,), device=self.device)
                    excluded_values = torch.tensor([0, 1, 4], device=self.device)
                    rand_lengths = (~torch.isin(original_id, excluded_values)).sum(dim=1)

                    sequences = [torch.cat([torch.tensor([0], device=self.device),
                                            torch.randint(7, 11, (l - 2,), device=self.device),
                                            torch.tensor([1], device=self.device)]) for l in rand_lengths]
                    # 7 - 10: ACGT
                    rand_id = torch.full((bs, seq_len), pad_token_id, dtype=torch.long, device=self.device)
                    for i, seq in enumerate(sequences):
                        rand_id[i, :len(seq)] = seq

            else:
                raise NotImplementedError()

            generated_id = self.device_gather(generated_id)
            original_id = self.device_gather(original_id)

            if "promoter" in self.dataset.dataset_name:
                sei_profile, sei_embed = sei_class.get_sei_profile(original_id, self.device, False)
                sei_profile_pred, sei_embed_pred = sei_class.get_sei_profile(generated_id, self.device, False)
                setattr(self, f"sei_mse_{prefix}", getattr(self, f"sei_mse_{prefix}") +
                        ((sei_profile - sei_profile_pred) ** 2).double().sum().cpu())
                setattr(self, f"sample_count_{prefix}", getattr(self, f"sample_count_{prefix}") + len(original_id))

                # fluency
                gen_loss, gen_mask = hyena_pretrained_model.get_score(generated_id, self.device)
                setattr(self, f"fluency_loss_{prefix}",
                        getattr(self, f"fluency_loss_{prefix}") + gen_loss.double().sum().cpu())
                setattr(self, f"count_{prefix}", getattr(self, f"count_{prefix}") + gen_mask.sum().cpu())

                # Predictive distribution shift
                sei_activity_original = getattr(self, f"sei_activity_original_{prefix}")
                sei_activity_pred = getattr(self, f"sei_activity_pred_{prefix}")
                sei_activity_original.append(sei_profile.double().cpu())
                sei_activity_pred.append(sei_profile_pred.double().cpu())

                # percent identity
                percent_identity_elements = percent_identity(generated_id, original_id)
                setattr(self, f"percent_identity_max_{prefix}",
                        torch.max(getattr(self, f"percent_identity_max_{prefix}"), percent_identity_elements.double().max().cpu())
                        )
                setattr(self, f"percent_identity_sum_{prefix}",
                        getattr(self, f"percent_identity_sum_{prefix}") + percent_identity_elements.double().sum().cpu())

                # k-mer spectrum shift, record whole sequence
                original_seq = getattr(self, f"original_seq_{prefix}")
                generated_seq = getattr(self, f"generated_seq_{prefix}")
                original_seq.append(original_id.cpu())
                generated_seq.append(generated_id.cpu())
            elif "enhancer" in self.dataset.dataset_name:
                # fluency
                gen_loss, gen_mask = hyena_pretrained_model.get_score(generated_id, self.device)
                setattr(self, f"fluency_loss_{prefix}",
                        getattr(self, f"fluency_loss_{prefix}") + gen_loss.double().sum().cpu())
                setattr(self, f"count_{prefix}", getattr(self, f"count_{prefix}") + gen_mask.sum().cpu())

                uncond_gen_loss, uncond_gen_mask = hyena_pretrained_model.get_score(uncond_generated_id, self.device)
                setattr(self, f"uncond_fluency_loss_{prefix}",
                        getattr(self, f"uncond_fluency_loss_{prefix}") + uncond_gen_loss.double().sum().cpu())
                setattr(self, f"uncond_count_{prefix}", getattr(self, f"uncond_count_{prefix}") + uncond_gen_mask.sum().cpu())

                classifier_model = classifier_fb if "flybrain" in self.dataset.dataset_name else classifier_mel
                fbd_gen_embed = getattr(self, f"fbd_gen_embed_{prefix}")
                fbd_ori_embed = getattr(self, f"fbd_ori_embed_{prefix}")
                uncond_fbd_gen_embed = getattr(self, f"uncond_fbd_gen_embed_{prefix}")
                fbd_gen_embed.append(classifier_model.get_embed(generated_id, self.device).detach().cpu())
                fbd_ori_embed.append(classifier_model.get_embed(original_id, self.device).detach().cpu())
                uncond_fbd_gen_embed.append(classifier_model.get_embed(uncond_generated_id, self.device).detach().cpu())

                original_list_seq = getattr(self, f"original_list_seq_{prefix}")
                diversity_list_seq = getattr(self, f"diversity_list_seq_{prefix}")
                diversity_list_signal = getattr(self, f"diversity_list_signal_{prefix}")
                uncond_diversity_list_seq = getattr(self, f"uncond_diversity_list_seq_{prefix}")
                uncond_diversity_list_signal = getattr(self, f"uncond_diversity_list_signal_{prefix}")
                original_list_seq.append(original_id.detach().cpu())
                diversity_list_seq.append(generated_id.detach().cpu())
                diversity_list_signal.append(cond_signal.detach().cpu())
                uncond_diversity_list_seq.append(uncond_generated_id.detach().cpu())
                uncond_diversity_list_signal.append(uncond_signal.detach().cpu())
            elif "chipseq" in self.dataset.dataset_name:
                # fluency
                gen_loss, gen_mask = hyena_pretrained_model.get_score(generated_id, self.device)
                setattr(self, f"fluency_loss_{prefix}",
                        getattr(self, f"fluency_loss_{prefix}") + gen_loss.double().sum().cpu())
                setattr(self, f"count_{prefix}", getattr(self, f"count_{prefix}") + gen_mask.sum().cpu())

                uncond_gen_loss, uncond_gen_mask = hyena_pretrained_model.get_score(uncond_generated_id, self.device)
                setattr(self, f"uncond_fluency_loss_{prefix}",
                        getattr(self, f"uncond_fluency_loss_{prefix}") + uncond_gen_loss.double().sum().cpu())
                setattr(self, f"uncond_count_{prefix}", getattr(self, f"uncond_count_{prefix}") + uncond_gen_mask.sum().cpu())

                real_gen_loss, real_gen_mask = hyena_pretrained_model.get_score(original_id, self.device)
                setattr(self, f"real_fluency_loss_{prefix}",
                        getattr(self, f"real_fluency_loss_{prefix}") + real_gen_loss.double().sum().cpu())
                setattr(self, f"real_count_{prefix}", getattr(self, f"real_count_{prefix}") + real_gen_mask.sum().cpu())

                # sei
                # sei_profile, sei_embed = sei_class.get_sei_profile(original_id, self.device, False)
                # sei_profile_uncond, sei_embed_uncond = sei_class.get_sei_profile(uncond_generated_id, self.device, False)
                # sei_profile_pred, sei_embed_pred = sei_class.get_sei_profile(generated_id, self.device, False)

                sei_profile, sei_embed = sei_class.get_sei_profile_any(original_id, w['protein_name'], self.device, False, cell_type=[self.dataset.dataset_train.idx_to_cell[idx.item()] for idx in batch['cell_type']])
                sei_profile_uncond, sei_embed_uncond = sei_class.get_sei_profile_any(uncond_generated_id, w['protein_name'], self.device, False, cell_type=[self.dataset.dataset_train.idx_to_cell[idx.item()] for idx in batch['cell_type']])
                sei_profile_pred, sei_embed_pred = sei_class.get_sei_profile_any(generated_id, w['protein_name'], self.device, False, cell_type=[self.dataset.dataset_train.idx_to_cell[idx.item()] for idx in batch['cell_type']])
                sei_profile_rand, sei_embed_rand = sei_class.get_sei_profile_any(rand_id, w['protein_name'], self.device, False, cell_type=[self.dataset.dataset_train.idx_to_cell[idx.item()] for idx in batch['cell_type']])
                setattr(self, f"sei_{prefix}", getattr(self, f"sei_{prefix}") + sei_profile_pred.double().sum().cpu())
                setattr(self, f"uncond_sei_{prefix}", getattr(self, f"uncond_sei_{prefix}") + sei_profile_uncond.double().sum().cpu())
                setattr(self, f"rand_sei_{prefix}", getattr(self, f"rand_sei_{prefix}") + sei_profile_rand.double().sum().cpu())
                setattr(self, f"real_sei_{prefix}", getattr(self, f"real_sei_{prefix}") + sei_profile.double().sum().cpu())
                setattr(self, f"sample_count_{prefix}", getattr(self, f"sample_count_{prefix}") + len(original_id))

                setattr(self, f"sei_mse_{prefix}", getattr(self, f"sei_mse_{prefix}") +
                        ((sei_profile - sei_profile_pred) ** 2).double().sum().cpu())
                setattr(self, f"uncond_sei_mse_{prefix}", getattr(self, f"uncond_sei_mse_{prefix}") +
                        ((sei_profile - sei_profile_uncond) ** 2).double().sum().cpu())
                setattr(self, f"rand_sei_mse_{prefix}", getattr(self, f"rand_sei_mse_{prefix}") +
                        ((sei_profile - sei_profile_rand) ** 2).double().sum().cpu())

                # diversity
                original_list_seq = getattr(self, f"original_list_seq_{prefix}")
                diversity_list_seq = getattr(self, f"diversity_list_seq_{prefix}")
                uncond_diversity_list_seq = getattr(self, f"uncond_diversity_list_seq_{prefix}")

                original_list_seq.append(original_id.detach().cpu())
                diversity_list_seq.append(generated_id.detach().cpu())
                uncond_diversity_list_seq.append(uncond_generated_id.detach().cpu())

        return loss

    def iterative_gen(self, batch):
        # masked language model generation
        input_embeds, condition, input_seqs = batch
        input_embeds[:, :, :4] = 0
        bs, seq_len = input_seqs.shape

        total_step = self.hparams.model.config.generation_step
        if isinstance(total_step, float):
            total_step = int(total_step * seq_len)
        elif total_step == 'full':
            total_step = seq_len

        for step in range(total_step):
            sampling_mask = (input_embeds[:, :, :4] == 0).all(dim=-1)  # current mask token, bs*seq
            sampling_mask_per_sample = sampling_mask[0]

            # Calculate number of tokens to sample
            still_masked = torch.sum(sampling_mask_per_sample).int()
            perc_masked_after_this_step = cosine_schedule(
                torch.tensor((step + 1) / total_step)
            )
            num_tokens_masked_after_this_step = (perc_masked_after_this_step * seq_len + 0.1).int()
            num_to_sample = still_masked - num_tokens_masked_after_this_step

            # model forward
            new_batch = (input_embeds, condition, input_seqs)
            cur_logits, label, _ = self.forward(new_batch)

            # get unmask position
            entropy = torch.distributions.Categorical(logits=cur_logits).entropy()  # bs*seq len
            entropy = entropy.masked_fill(
                ~sampling_mask, torch.finfo(entropy.dtype).max
            )
            entropies, indices = entropy.topk(num_to_sample, dim=-1, largest=False)

            is_top_k = torch.zeros((bs, seq_len), dtype=torch.bool, device=input_embeds.device).scatter(
                1, indices, True
            )
            where_to_sample = sampling_mask & is_top_k

            # remove N, only keep first 4 dimension, convert to one-hot encoding
            cur_logits = cur_logits.reshape(bs * seq_len, -1)
            select_indices, _ = sample_from_logits(cur_logits, 5, temp=self.hparams.train.temp,
                               greedy=self.hparams.model.config.greedy_gen, return_prob=False)
            select_indices = select_indices.squeeze(-1)
            select_indices = select_indices.reshape(bs, seq_len)

            mask_N = (select_indices == 4)  # N token
            select_indices[mask_N] = 0  # alter later

            one_hot_output = F.one_hot(select_indices, num_classes=4).to(dtype=cur_logits.dtype, device=cur_logits.device)
            one_hot_output[mask_N.unsqueeze(-1).expand_as(one_hot_output)] = 0
            cur_logits = one_hot_output

            # update embedding
            new_embeds = torch.concat((cur_logits, input_embeds[:, :, 4:]), dim=-1)
            input_embeds = torch.where(
                where_to_sample.unsqueeze(-1), new_embeds, input_embeds
            )

        # mapping_tensor = torch.tensor([valid_vocab['A'], valid_vocab['C'], valid_vocab['G'], valid_vocab['T']])
        # 4-dim one hot to 5 choice: ACGTN
        argmax_indices = torch.argmax(input_embeds[:, :, :4], dim=-1)
        is_all_zero = (input_embeds[:, :, :4].sum(dim=-1) == 0)
        generated_id = torch.where(is_all_zero, torch.full_like(argmax_indices, 4), argmax_indices)

        return generated_id

    def autoregressive_gen(self, batch_size, batch):
        cls_id = self.dataset.tokenizer.cls_token_id
        generated_id = torch.LongTensor(batch_size, 1).fill_(cls_id).to(self.device)

        for cur_len in range(self.dataset.max_length + 1):
            # new_batch = (generated_id, None, batch["condition"])
            new_batch = {
                "data": generated_id,
                "target": None,
                "condition": batch["condition"],
                "cell_type": batch['cell_type'] if 'cell_type' in batch else None,
            }
            logits, _, _ = self.forward(new_batch)
            logits = logits[:, -1, :]
            pred_id, pred_prob = sample_from_logits(logits, 4, temp=self.hparams.train.temp, return_prob=True)
            generated_id = torch.cat([generated_id, pred_id], dim=-1)

        sep_id = self.dataset.tokenizer.sep_token_id
        eos_mask = generated_id == sep_id
        cumulative_eos_mask = eos_mask.cumsum(dim=1) > 0
        generated_id[cumulative_eos_mask] = sep_id
        return generated_id

    def eval_promo_metrics(self, prefix):
        sei_mse = getattr(self, f"sei_mse_{prefix}")
        fluency_loss = getattr(self, f"fluency_loss_{prefix}")
        count = getattr(self, f"count_{prefix}")
        sei_activity_pred = getattr(self, f"sei_activity_pred_{prefix}")
        sei_activity_original = getattr(self, f"sei_activity_original_{prefix}")
        percent_identity_max = getattr(self, f"percent_identity_max_{prefix}")
        percent_identity_sum = getattr(self, f"percent_identity_sum_{prefix}")
        sample_count = getattr(self, f"sample_count_{prefix}")
        original_seq = getattr(self, f"original_seq_{prefix}")
        generated_seq = getattr(self, f"generated_seq_{prefix}")

        avg_sei_mse = sei_mse / sample_count

        # fluency
        avg_fluency = torch.exp(fluency_loss / count)

        # ks
        sei_activity_pred = torch.cat(sei_activity_pred, dim=0).numpy()
        sei_activity_original = torch.cat(sei_activity_original, dim=0).numpy()
        ks_statistic, p_value = ks_2samp(sei_activity_pred, sei_activity_original)
        pred_dist_shift = torch.tensor(ks_statistic)

        # percent identity
        percent_identity_avg = percent_identity_sum / sample_count

        # k-mer spectrum shift
        original_seq = torch.cat(original_seq, dim=0)
        generated_seq = torch.cat(generated_seq, dim=0)
        kld, jsd = kmer_statistics(original_seq, generated_seq)  # kl divergence, js divergence

        metrics = {
            f'{prefix}/sei_promoter': avg_sei_mse,
            f'{prefix}/fluency': avg_fluency,
            f'{prefix}/pred_dist_shift': pred_dist_shift,
            f'{prefix}/percent_identity_max': percent_identity_max,
            f'{prefix}/percent_identity_avg': percent_identity_avg,
            f'{prefix}/JS_distance': jsd,
        }
        return metrics

    def eval_enhan_metrics(self, prefix):
        fluency_loss = getattr(self, f"fluency_loss_{prefix}")
        count = getattr(self, f"count_{prefix}")
        uncond_fluency_loss = getattr(self, f"uncond_fluency_loss_{prefix}")
        uncond_count = getattr(self, f"uncond_count_{prefix}")
        fbd_gen_embed = getattr(self, f"fbd_gen_embed_{prefix}")
        fbd_ori_embed = getattr(self, f"fbd_ori_embed_{prefix}")
        uncond_fbd_gen_embed = getattr(self, f"uncond_fbd_gen_embed_{prefix}")
        diversity_list_seq = getattr(self, f"diversity_list_seq_{prefix}")
        diversity_list_signal = getattr(self, f"diversity_list_signal_{prefix}")
        uncond_diversity_list_seq = getattr(self, f"uncond_diversity_list_seq_{prefix}")
        uncond_diversity_list_signal = getattr(self, f"uncond_diversity_list_signal_{prefix}")
        original_list_seq = getattr(self, f"original_list_seq_{prefix}")

        # fluency
        avg_fluency = torch.exp(fluency_loss / count)
        uncond_avg_fluency = torch.exp(uncond_fluency_loss / uncond_count)

        # frechet distance
        fbd_gen_embed = torch.cat(fbd_gen_embed, dim=0)
        fbd_ori_embed = torch.cat(fbd_ori_embed, dim=0)
        uncond_fbd_gen_embed = torch.cat(uncond_fbd_gen_embed, dim=0)
        fbd_gen_embed_flat = fbd_gen_embed.view(-1, 128).float().numpy()
        fbd_ori_embed_flat = fbd_ori_embed.view(-1, 128).float().numpy()
        uncond_fbd_gen_embed_flat = uncond_fbd_gen_embed.view(-1, 128).float().numpy()
        fbd = get_wasserstein_dist(fbd_gen_embed_flat, fbd_ori_embed_flat)
        uncond_fbd = get_wasserstein_dist(uncond_fbd_gen_embed_flat, fbd_ori_embed_flat)
        fbd = torch.tensor(fbd, dtype=torch.float32)
        uncond_fbd = torch.tensor(uncond_fbd, dtype=torch.float32)

        diversity_list_seq = torch.cat(diversity_list_seq, dim=0)
        diversity_list_signal = torch.cat(diversity_list_signal, dim=0)
        result_dict = {}
        for i in range(len(diversity_list_seq)):
            k = diversity_list_signal[i].item()
            v = diversity_list_seq[i]
            if k in result_dict:
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]
        diversity_score = calculate_weighted_category_diversity(result_dict)

        uncond_diversity_list_seq = torch.cat(uncond_diversity_list_seq, dim=0)
        uncond_diversity_list_signal = torch.cat(uncond_diversity_list_signal, dim=0)
        uncond_result_dict = {}
        for i in range(len(uncond_diversity_list_seq)):
            k = uncond_diversity_list_signal[i].item()
            v = uncond_diversity_list_seq[i]
            if k in uncond_result_dict:
                uncond_result_dict[k].append(v)
            else:
                uncond_result_dict[k] = [v]
        uncond_diversity_score = calculate_weighted_category_diversity(uncond_result_dict)

        # percent identity
        pi_max_go, pi_avg_go, pi_max_gg, pi_avg_gg = percent_identity_group(diversity_list_seq,
                                                                            torch.cat(original_list_seq, dim=0),
                                                                            diversity_list_signal)

        metrics = {
            f'{prefix}/fbd': fbd,
            f'{prefix}/uncond_fbd': uncond_fbd,
            f'{prefix}/diversity': diversity_score,
            f'{prefix}/uncond_diversity': uncond_diversity_score,
            f'{prefix}/fluency': avg_fluency,
            f'{prefix}/uncond_fluency': uncond_avg_fluency,
            f'{prefix}/percent_identity_max_gen_obs': pi_max_go,
            f'{prefix}/percent_identity_avg_gen_obs': pi_avg_go,
            f'{prefix}/percent_identity_max_gen_gen': pi_max_gg,
            f'{prefix}/percent_identity_avg_gen_gen': pi_avg_gg,
        }
        return metrics

    def eval_chipseq_metrics(self, prefix):
        sei = getattr(self, f"sei_{prefix}")
        uncond_sei = getattr(self, f"uncond_sei_{prefix}")
        real_sei = getattr(self, f"real_sei_{prefix}")
        rand_sei = getattr(self, f"rand_sei_{prefix}")
        sample_count = getattr(self, f"sample_count_{prefix}")

        sei_mse = getattr(self, f"sei_mse_{prefix}")
        uncond_sei_mse = getattr(self, f"uncond_sei_mse_{prefix}")
        rand_sei_mse = getattr(self, f"rand_sei_mse_{prefix}")

        fluency_loss = getattr(self, f"fluency_loss_{prefix}")
        count = getattr(self, f"count_{prefix}")
        uncond_fluency_loss = getattr(self, f"uncond_fluency_loss_{prefix}")
        uncond_count = getattr(self, f"uncond_count_{prefix}")
        real_fluency_loss = getattr(self, f"real_fluency_loss_{prefix}")
        real_count = getattr(self, f"real_count_{prefix}")

        diversity_list_seq = getattr(self, f"diversity_list_seq_{prefix}")
        uncond_diversity_list_seq = getattr(self, f"uncond_diversity_list_seq_{prefix}")
        real_diversity_list_seq = getattr(self, f"original_list_seq_{prefix}")

        # sei
        avg_sei = sei / sample_count
        avg_uncond_sei = uncond_sei / sample_count
        avg_real_sei = real_sei / sample_count
        avg_rand_sei = rand_sei / sample_count

        avg_sei_mse = sei_mse / sample_count
        avg_uncond_sei_mse = uncond_sei_mse / sample_count
        avg_rand_sei_mse = rand_sei_mse / sample_count

        # fluency
        avg_fluency = torch.exp(fluency_loss / count)
        uncond_avg_fluency = torch.exp(uncond_fluency_loss / uncond_count)
        real_avg_fluency = torch.exp(real_fluency_loss / real_count)

        # diversity
        diversity_list_seq = torch.cat(diversity_list_seq, dim=0)
        result_dict = {}
        for i in range(len(diversity_list_seq)):
            k = 0
            v = diversity_list_seq[i]
            if k in result_dict:
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]
        diversity_score = calculate_weighted_category_diversity(result_dict)

        uncond_diversity_list_seq = torch.cat(uncond_diversity_list_seq, dim=0)
        uncond_result_dict = {}
        for i in range(len(uncond_diversity_list_seq)):
            k = 0
            v = uncond_diversity_list_seq[i]
            if k in uncond_result_dict:
                uncond_result_dict[k].append(v)
            else:
                uncond_result_dict[k] = [v]
        uncond_diversity_score = calculate_weighted_category_diversity(uncond_result_dict)

        real_diversity_list_seq = torch.cat(real_diversity_list_seq, dim=0)
        real_result_dict = {}
        for i in range(len(real_diversity_list_seq)):
            k = 0
            v = real_diversity_list_seq[i]
            if k in real_result_dict:
                real_result_dict[k].append(v)
            else:
                real_result_dict[k] = [v]
        real_diversity_score = calculate_weighted_category_diversity(real_result_dict)

        metrics = {
            f'{prefix}/sei': avg_sei,
            f'{prefix}/uncond_sei': avg_uncond_sei,
            f'{prefix}/real_sei': avg_real_sei,
            f'{prefix}/rand_sei': avg_rand_sei,
            f'{prefix}/sei_mse': avg_sei_mse,
            f'{prefix}/uncond_sei_mse': avg_uncond_sei_mse,
            f'{prefix}/rand_sei_mse': avg_rand_sei_mse,

            f'{prefix}/fluency': avg_fluency,
            f'{prefix}/uncond_fluency': uncond_avg_fluency,
            f'{prefix}/real_fluency': real_avg_fluency,
            f'{prefix}/diversity': diversity_score,
            f'{prefix}/uncond_diversity': uncond_diversity_score,
            f'{prefix}/real_diversity': real_diversity_score,
        }
        return metrics

    def eval_epoch_end(self, prefix):
        if "promoter" in self.dataset.dataset_name:
            if ((not self.hparams.train.remove_test_loader_in_eval) and
                    (not self.hparams.train.remove_val_loader_in_eval)):
                val_metrics = self.eval_promo_metrics('val')
                test_metrics = self.eval_promo_metrics('test')
                metrics = {**val_metrics, **test_metrics}
            else:
                metrics = self.eval_promo_metrics(prefix)

        elif "enhancer" in self.dataset.dataset_name:
            if ((not self.hparams.train.remove_test_loader_in_eval) and
                    (not self.hparams.train.remove_val_loader_in_eval)):
                val_metrics = self.eval_enhan_metrics('val')
                test_metrics = self.eval_enhan_metrics('test')
                metrics = {**val_metrics, **test_metrics}
            else:
                metrics = self.eval_enhan_metrics(prefix)
        elif "chipseq" in self.dataset.dataset_name:
            if ((not self.hparams.train.remove_test_loader_in_eval) and
                    (not self.hparams.train.remove_val_loader_in_eval)):
                val_metrics = self.eval_chipseq_metrics('val')
                test_metrics = self.eval_chipseq_metrics('test')
                metrics = {**val_metrics, **test_metrics}
            else:
                metrics = self.eval_chipseq_metrics(prefix)
        else:
            raise NotImplementedError()

        self.log_dict(
            metrics,
            # on_step=False,
            # on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")
        self.reset_torch_metrics()

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)
        self.reset_torch_metrics()

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end('val')
        # if self.val_stage == 'val':
        #     self.eval_epoch_end(self.val_torchmetrics, self.uncond_val_torchmetrics)
        # elif self.val_stage == 'test':
        #     self.eval_epoch_end(self.test_torchmetrics, self.uncond_test_torchmetrics)
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)
        self.reset_torch_metrics()

    def test_epoch_end(self, outputs):
        self.eval_epoch_end('test')
        # self.eval_epoch_end(self.test_torchmetrics, self.uncond_test_torchmetrics)
        # Log all test torchmetrics
        super().test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so that it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": float(self.current_epoch)}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # # Log any extra info that the models want to expose (e.g. output norms)
        # metrics = {}
        # for module in list(self.modules())[1:]:
        #     if hasattr(module, "metrics"):
        #         metrics.update(module.metrics)
        #
        # self.log_dict(
        #     metrics,
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     add_dataloader_idx=False,
        #     sync_dist=True,
        # )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial
        # sanity check
        self.val_stage = self.val_loader_names[dataloader_idx]
        ema = (
                self.val_loader_names[dataloader_idx].endswith("/ema")
                and self.optimizers().optimizer.stepped
        )
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        # all_params = list(self.parameters())
        # params = [p for p in all_params if not hasattr(p, "_optim")]
        all_named_params = list(self.named_parameters())
        params = [p for name, p in all_named_params if "torchmetrics" not in name and not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for name, p in all_named_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups:", hps)  # TODO: log.info throws error because hps is list of dicts
        for hp in hps:
            params = [p for name, p in all_named_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                        self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizers param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (e.g., if test is duplicate)
        eval_loader_names = []
        eval_loaders = []
        if not self.hparams.train.get("remove_val_loader_in_eval", False):
            eval_loader_names += val_loader_names
            eval_loaders += val_loaders
        if not self.hparams.train.get("remove_test_loader_in_eval", False):
            eval_loader_names += test_loader_names
            eval_loaders += test_loaders
        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        # self.test_loader_names = ["final/" + name for name in test_loader_names]
        self.test_loader_names = test_loader_names
        return test_loaders


# pytorch-lightning utils and entrypoint
def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        log.info(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            log.info(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
            gradient_as_bucket_view=True,
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # special processing for seqlen warmup scheduler (reload)
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    return trainer


def fsspec_exists(filename):
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained_model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )

    # Run initial validation epoch (useful for debugging, fine-tuning)
    if config.train.validate_at_start:
        log.info("Running validation before training")
        trainer.validate(model)

    log.info(f'{config.train.ckpt=} {fsspec_exists(config.train.ckpt)=}')
    # if config.train.get("compile_model", False):
    #     model = torch.compile(model, mode="reduce-overhead")
    if config.train.ckpt is not None and fsspec_exists(config.train.ckpt):
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)

    # if config.train.test:
    #     if config.train.get("cross_validation", False):  # First, load the best validation model
    #         best_val_ckpt = os.path.join(
    #             model.hparams.callbacks.model_checkpoint.dirpath,
    #             f"{model.hparams.callbacks.model_checkpoint.filename}.ckpt",
    #         )
    #         # Update config so we do not load just the backbone
    #         config.train.pretrained_model_state_hook.update({"_name_": None})
    #         # Remove validation loader
    #         config.train.update({"remove_val_loader_in_eval": True})
    #         config.train.update({"remove_test_loader_in_eval": False})
    #         ckpt = torch.load(best_val_ckpt)
    #         log.info(f"Loaded best validation checkpoint from epoch {ckpt['epoch']}")
    #         trainer.validate(model, ckpt_path=best_val_ckpt)
    #     else:
    #         trainer.validate(model)

    if config.train.test:
        best_val_ckpt = os.path.join(
            model.hparams.callbacks.model_checkpoint.dirpath,
            f"{model.hparams.callbacks.model_checkpoint.filename}.ckpt",
        )
        return model_test(config, best_val_ckpt)


def model_test(config, path, evaluate='test'):
    model = SequenceLightningModule(config)
    if config.train.get("cross_validation", False):  # First, load the best validation model
        # Update config so we do not load just the backbone
        config.train.pretrained_model_state_hook.update({"_name_": None})
        config.train.update({"remove_val_loader_in_eval": True})
        config.train.update({"remove_test_loader_in_eval": False})
        ckpt = torch.load(path)
        log.info(f"Loaded best validation checkpoint from epoch {ckpt['epoch']}")
        # initialize a new trainer with only one device
        config.trainer.devices = 1
        if 'strategy' in config.get('trainer', {}):
            del config['trainer']['strategy']
        new_trainer = create_trainer(config)
        if evaluate == 'test':
            final_test_results = new_trainer.test(model, ckpt_path=path)
        elif evaluate == 'valid':
            final_test_results = new_trainer.validate(model, ckpt_path=path)
        else:
            raise NotImplementedError()

        # collect the results, even in distributed training
        average_results = defaultdict(float)
        for result_dict in final_test_results:
            for key, value in result_dict.items():
                average_results[key] += value
        # average on number of devices
        for key in average_results:
            average_results[key] /= len(final_test_results)
        return average_results


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)
    # if config.train.get("compile_model", False):
    #     # See: https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    #     from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
    #     allow_ops_in_compiled_graph()

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    if config.train.only_test:
        model_test(config, config.train.only_test_model_path, evaluate=config.train.evaluate)
    else:
        train(config)


if __name__ == "__main__":
    main()
