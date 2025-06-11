from typing import Any
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim import Optimizer
import torch
from time import time
from typing import Mapping
from pdearena.utils.utils import get_logger

logger = get_logger(__name__)

class ParamCounter(Callback):

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        num_params = sum(p.numel() for p in pl_module.model.parameters())
        pl_module.log(f"num_params", float(num_params), sync_dist=True, rank_zero_only=True, batch_size=1)

class StepCounter(Callback):

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.log(f"estimated_steps_epoch", float(pl_module.steps_per_epoch), sync_dist=True, rank_zero_only=True, batch_size=1)
        pl_module.log(f"estimated_stepping_batches", float(pl_module.estimated_stepping_batches), sync_dist=True, rank_zero_only=True, batch_size=1)
        logger.info(f"Estimated steps per epoch: {pl_module.steps_per_epoch}")
        logger.info(f"Estimated stepping batches: {pl_module.estimated_stepping_batches}")

class StepTimer(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.set_time()
        self.train_step_sum = 0

    def get_time(self):
        return time()
    
    def set_time(self):
        self.last_time = self.get_time()

    def log_elapsed(self, pl_module: LightningModule, mode: str):
        curr_time = self.get_time()
        elapsed = curr_time - self.last_time
        pl_module.log(f"{mode}/step_sec", elapsed, sync_dist=True, on_epoch=True, on_step=True, rank_zero_only=True, batch_size=1)
        self.set_time()
        return elapsed

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.set_time()

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.set_time()

    @rank_zero_only
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.set_time()

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        elapsed = self.log_elapsed(pl_module=pl_module, mode="train")
        self.train_step_sum += elapsed
        avg_train_step_time = self.train_step_sum / trainer.global_step
        estimated_total_time = avg_train_step_time * pl_module.estimated_stepping_batches / 60**2
        estimated_remaining_time = estimated_total_time - trainer.global_step * avg_train_step_time / 60**2
        pl_module.log("train/estimated_total_hours", estimated_total_time, sync_dist=True, on_step=True, rank_zero_only=True, batch_size=1)
        pl_module.log("train/estimated_remaining_hours", estimated_remaining_time, sync_dist=True, on_step=True, rank_zero_only=True, batch_size=1)

    @rank_zero_only
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log_elapsed(pl_module=pl_module, mode="valid")
    
    @rank_zero_only
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log_elapsed(pl_module=pl_module, mode="test")
    
class GPUStatsCallback(Callback):

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        torch.cuda.reset_peak_memory_stats()

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        torch.cuda.reset_peak_memory_stats()

    @rank_zero_only
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        torch.cuda.reset_peak_memory_stats()

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        self.log_gpu_stats(trainer, pl_module, "train")
    
    @rank_zero_only
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log_gpu_stats(trainer, pl_module, "valid")
    
    @rank_zero_only
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log_gpu_stats(trainer, pl_module, "test")
    
    def log_gpu_stats(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        if (trainer.global_step % trainer.log_every_n_steps) == 0:
            mem = torch.cuda.max_memory_reserved() / 1024**3
            pl_module.log(f"{mode}/gpu_mem", mem, sync_dist=True, on_epoch=True, on_step=True, rank_zero_only=True, batch_size=1)

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    https://github.com/Lightning-AI/pytorch-lightning/issues/1462#issuecomment-1190253742
    """
    def __init__(self):
        super().__init__()
        self.warn = True

    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        if (trainer.global_step % trainer.log_every_n_steps) == 0:
            grad_norm, param_norm = self.gradient_norm(pl_module.model)
            pl_module.log("train/grad_norm", grad_norm, sync_dist=True, on_epoch=True, on_step=True, rank_zero_only=True, batch_size=1)
            pl_module.log("train/param_norm", param_norm, sync_dist=True, on_epoch=True, on_step=True, rank_zero_only=True, batch_size=1)

    def gradient_norm(self, model: torch.nn.Module):
        with torch.no_grad():
            total_grad_norm = 0.0
            total_param_norm = 0.0
            no_grad = dict()
            for pname, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.data.detach().norm(2)
                    grad_norm = p.grad.detach().data.norm(2)
                    total_grad_norm += grad_norm.item() ** 2
                    total_param_norm += param_norm.item() ** 2
                elif self.warn:
                    no_grad[pname] = p.shape
            if len(no_grad) > 0:
                self.warn = False
                logger.info(f"The following parameters have no gradient: {no_grad}")
            total_grad_norm = total_grad_norm ** (1. / 2)
            total_param_norm = total_param_norm ** (1. / 2)
        return total_grad_norm, total_param_norm
    
