"""
Lightning callback: detect abnormal flow training loss spikes and dump diagnostics.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from statistics import median
from typing import Deque, Optional

import torch
from lightning.pytorch.callbacks import Callback

from scdp.model.flow_spike_debug import dump_flow_spike_diagnostics

pylogger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    return float(raw) if raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw.strip() else default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class FlowLossSpikeCallback(Callback):
    """
    On abnormally large ``loss/train`` (flow module only), print coefficient / bridge
    diagnostics to stdout and ``{log_dir}/spike_logs/``.

    Enable via ``FLOW_LOSS_SPIKE_DEBUG=1`` (see ``vision/train_flow_matching_entanglement_debug.slurm``).

    Trigger when ``loss > max(threshold_abs, spike_factor * recent_median)`` after
    ``warmup_batches`` samples on global rank 0.
    """

    def __init__(
        self,
        log_dir: str,
        threshold_abs: Optional[float] = None,
        spike_factor: Optional[float] = None,
        warmup_batches: Optional[int] = None,
        max_reports: Optional[int] = None,
        history_size: int = 200,
    ):
        self.log_dir = Path(log_dir)
        self.threshold_abs = (
            threshold_abs
            if threshold_abs is not None
            else _env_float("FLOW_SPIKE_THRESHOLD_ABS", 1.0)
        )
        self.spike_factor = (
            spike_factor
            if spike_factor is not None
            else _env_float("FLOW_SPIKE_FACTOR", 50.0)
        )
        self.warmup_batches = (
            warmup_batches
            if warmup_batches is not None
            else _env_int("FLOW_SPIKE_WARMUP_BATCHES", 100)
        )
        self.max_reports = (
            max_reports
            if max_reports is not None
            else _env_int("FLOW_SPIKE_MAX_REPORTS", 20)
        )
        self.history_size = history_size
        self._loss_history: Deque[float] = deque(maxlen=history_size)
        self._reports = 0

    @classmethod
    def from_env(cls, log_dir: str) -> Optional["FlowLossSpikeCallback"]:
        if not _env_bool("FLOW_LOSS_SPIKE_DEBUG"):
            return None
        return cls(log_dir=log_dir)

    def _is_spike(self, loss_val: float) -> bool:
        if loss_val >= self.threshold_abs:
            return True
        if len(self._loss_history) < max(10, self.warmup_batches // 2):
            return False
        ref = median(self._loss_history)
        if ref <= 0:
            return loss_val > self.threshold_abs
        return loss_val >= self.spike_factor * ref

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero:
            return
        if outputs is None:
            return
        if not hasattr(pl_module, "flow_matching_loss_batch"):
            return

        loss = outputs if torch.is_tensor(outputs) else outputs.get("loss")
        if loss is None:
            return
        loss_val = float(loss.detach().float().cpu())

        step = int(trainer.global_step)
        warmup_done = step >= self.warmup_batches
        is_spike = warmup_done and self._is_spike(loss_val)

        if is_spike and self._reports < self.max_reports:
            self._reports += 1
            recent = median(self._loss_history) if self._loss_history else None
            log_path = (
                self.log_dir
                / "spike_logs"
                / f"spike_step{step}_b{batch_idx}_loss{loss_val:.4e}.log"
            )
            pylogger.warning(
                "Flow loss spike at step=%s batch=%s loss=%.6e (median=%.6e, reports=%s/%s). "
                "See %s",
                step,
                batch_idx,
                loss_val,
                recent if recent is not None else float("nan"),
                self._reports,
                self.max_reports,
                log_path,
            )
            dump_flow_spike_diagnostics(
                pl_module,
                batch,
                loss_value=loss_val,
                global_step=step,
                batch_idx=batch_idx,
                recent_median=recent,
                spike_factor=self.spike_factor,
                log_path=log_path,
            )

        self._loss_history.append(loss_val)
