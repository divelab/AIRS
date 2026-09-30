"""Checkpoint helpers for warm-start / finetune (weights-only load)."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from lightning.pytorch import LightningModule

if TYPE_CHECKING:
    from lightning.pytorch import Trainer

pylogger = logging.getLogger(__name__)

WEIGHTS_ONLY_CKPT_NAME = "init_weights.ckpt"
# Full warm-start from another run (finetune staging). Never overwrite Lightning ``last.ckpt``.
WARMSTART_FULL_CKPT_NAME = "warmstart_full.ckpt"


def strip_checkpoint_weights_only(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return a checkpoint dict with model (+ optional EMA) weights only."""
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint missing 'state_dict'")
    out: Dict[str, Any] = {"state_dict": checkpoint["state_dict"]}
    if "ema_state_dict" in checkpoint:
        out["ema_state_dict"] = checkpoint["ema_state_dict"]
    return out


def save_weights_only_checkpoint(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Write ``init_weights.ckpt``-style file from a full Lightning checkpoint."""
    src_path = Path(src)
    dst_path = Path(dst)
    ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)
    stripped = strip_checkpoint_weights_only(ckpt)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, str(dst_path))
    return dst_path


def load_weights_only_checkpoint(
    model: LightningModule,
    ckpt_path: Union[str, Path],
    *,
    strict: bool = False,
) -> None:
    """Load model (+ EMA) weights; optimizer / scheduler / step counter stay fresh."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stripped = strip_checkpoint_weights_only(ckpt)
    missing, unexpected = model.load_state_dict(stripped["state_dict"], strict=strict)
    if missing:
        pylogger.warning("weights-only load missing keys: %s", missing)
    if unexpected:
        pylogger.warning("weights-only load unexpected keys: %s", unexpected)
    if "ema_state_dict" in stripped and hasattr(model, "ema"):
        try:
            model.ema.load_state_dict(stripped["ema_state_dict"])
        except Exception as exc:
            pylogger.warning("Failed to load EMA from weights-only ckpt: %s", exc)
    pylogger.info(
        "Loaded weights-only checkpoint from %s (optimizer, LR schedule, and global_step reset)",
        ckpt_path,
    )


def parse_global_step_from_ckpt_path(path: Union[str, Path]) -> Optional[int]:
    """Parse ``global_step`` from Lightning names like ``epoch=37-step=147060.ckpt``."""
    name = Path(path).name
    if "step=" not in name:
        return None
    try:
        return int(name.split("step=")[1].split(".ckpt")[0])
    except (IndexError, ValueError):
        return None


def load_global_step_from_ckpt(path: Union[str, Path]) -> int:
    """Return ``global_step`` from filename when possible; otherwise load the checkpoint."""
    resolved = Path(path).resolve() if Path(path).is_symlink() else Path(path)
    step = parse_global_step_from_ckpt_path(resolved)
    if step is not None:
        return step
    ckpt = torch.load(str(resolved), map_location="cpu", weights_only=False)
    return int(ckpt.get("global_step", -1))


def save_fit_end_checkpoint(trainer: "Trainer", storage_dir: Union[str, Path]) -> Path:
    """
    Write the trainer state after ``fit()`` completes.

    Lightning ``ModelCheckpoint.save_last`` may symlink to the last top-val file, which can
    lag ``global_step`` when ``max_steps`` stops mid-epoch. This saves a real full-state
    file at the actual step count for resume / multi-phase handoff.
    """
    import torch.distributed as dist

    storage = Path(storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    step = int(trainer.global_step)
    epoch = int(trainer.current_epoch)
    path = storage / f"epoch={epoch}-step={step}.ckpt"
    trainer.save_checkpoint(path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    last_path = storage / "last.ckpt"
    if trainer.is_global_zero:
        if last_path.exists() or last_path.is_symlink():
            last_path.unlink(missing_ok=True)
        # Copy on rank 0 only. A second trainer.save_checkpoint() here would issue
        # DDP collectives while other ranks already advanced to teardown and hang NCCL.
        shutil.copy2(path, last_path)
        pylogger.info(
            "Saved fit-end checkpoints: %s and last.ckpt (global_step=%s)",
            path.name,
            step,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path


def best_resume_global_step(storage_dir: Union[str, Path]) -> int:
    """Largest ``global_step`` among resume checkpoints, or ``-1`` if none."""
    storage = Path(storage_dir)
    best = _best_full_resume_ckpt(storage)
    if best is not None:
        return load_global_step_from_ckpt(best)
    warmstart = storage / WARMSTART_FULL_CKPT_NAME
    if warmstart.is_file():
        return load_global_step_from_ckpt(warmstart)
    return -1


def resolve_latest_resume_ckpt(storage_dir: Union[str, Path]) -> Path:
    """Checkpoint with the largest ``global_step`` (ignores stale ``last.ckpt``)."""
    ckpt = _best_full_resume_ckpt(Path(storage_dir))
    if ckpt is None:
        raise FileNotFoundError(f"No resume checkpoint found in {storage_dir}")
    return ckpt


def _model_checkpoint_monitor_state(ckpt_path: Union[str, Path]) -> Optional[dict]:
    """Lightning ModelCheckpoint callback state embedded in a full checkpoint."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except FileNotFoundError:
        pylogger.warning("checkpoint missing, skipping: %s", ckpt_path)
        return None
    for v in (ckpt.get("callbacks") or {}).values():
        if isinstance(v, dict) and "best_model_path" in v and "current_score" in v:
            return v
    return None


def resolve_eval_ckpt(
    storage_dir: Union[str, Path],
    *,
    use_last: bool = False,
    use_best: bool = True,
) -> Tuple[str, Optional[float]]:
    """
    Pick a checkpoint for evaluation / inference.

    Priority:
      1. ``last.ckpt`` when ``use_last``
      2. Lowest stored ``current_score`` among ``epoch=*.ckpt`` when ``use_best``
         (ModelCheckpoint monitor, e.g. nmape/val_flow)
      3. Highest epoch index among ``epoch=*.ckpt`` otherwise
    """
    storage = Path(storage_dir)
    if use_last and (storage / "last.ckpt").exists():
        return str(storage / "last.ckpt"), None

    ckpts = sorted(p for p in storage.glob("epoch=*.ckpt") if p.is_file())
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {storage}")

    if use_best:
        best_path = None
        best_score = None
        for p in ckpts:
            st = _model_checkpoint_monitor_state(p)
            if st is None:
                continue
            score = st.get("current_score")
            if score is None:
                continue
            score_f = float(score)
            if best_score is None or score_f < best_score:
                best_score = score_f
                best_path = str(p)
        if best_path is not None:
            pylogger.info(
                "eval checkpoint (best monitor score=%s): %s",
                best_score,
                best_path,
            )
            return best_path, best_score

    ckpt_epochs = [int(p.name.split("-")[0].split("=")[1]) for p in ckpts]
    p = ckpts[int(sorted(range(len(ckpt_epochs)), key=lambda i: ckpt_epochs[i])[-1])]
    st = _model_checkpoint_monitor_state(p)
    score = float(st["current_score"]) if st and st.get("current_score") is not None else None
    pylogger.info("eval checkpoint (latest epoch): %s", p)
    return str(p), score


def _collect_full_resume_candidates(storage: Path) -> list[Path]:
    """Lightning + finetune checkpoints that may resume training (not warmstart-only files)."""
    candidates: list[Path] = []
    last_ckpt = storage / "last.ckpt"
    if last_ckpt.is_file() or last_ckpt.is_symlink():
        candidates.append(last_ckpt)
    candidates.extend(storage.glob("last-v*.ckpt"))
    candidates.extend(storage.glob("epoch=*.ckpt"))
    return candidates


def _best_full_resume_ckpt(storage: Path) -> Optional[Path]:
    """
    Pick the checkpoint with the largest ``global_step``.

    Avoids resuming a stale staged ``last.ckpt`` when newer ``epoch=*.ckpt`` exist
    (e.g. finetune warm-start copied pretrain best into ``last.ckpt`` before Lightning
    started writing ``last-v1.ckpt``).
    """
    candidates = _collect_full_resume_candidates(storage)
    if not candidates:
        return None
    best = max(candidates, key=lambda p: load_global_step_from_ckpt(p))
    best_step = load_global_step_from_ckpt(best)
    pylogger.info(
        "found checkpoint: %s (global_step=%s, chosen from %d candidate(s))",
        best.resolve() if best.is_symlink() else best,
        best_step,
        len(candidates),
    )
    return best


def resolve_fit_checkpoint(
    storage_dir: Union[str, Path],
    *,
    weights_only_on_cold_start: bool = False,
) -> Tuple[Optional[Union[str, Path]], bool]:
    """
    Choose how ``trainer.fit`` should restore training state.

    Returns:
        (ckpt_path, loaded_weights_only)

    - Newest among ``last.ckpt``, ``last-v*.ckpt``, ``epoch=*.ckpt`` by ``global_step``.
    - ``warmstart_full.ckpt`` on first finetune launch (before any Lightning saves).
    - ``init_weights.ckpt`` when ``weights_only_on_cold_start`` → load weights in-process,
      return ``(None, True)``.
    - otherwise → fresh training ``(None, False)``.
    """
    storage = Path(storage_dir)

    resume_ckpt = _best_full_resume_ckpt(storage)
    if resume_ckpt is not None:
        return resume_ckpt, False

    warmstart = storage / WARMSTART_FULL_CKPT_NAME
    if warmstart.is_file():
        pylogger.info("found warm-start checkpoint: %s", warmstart)
        return warmstart, False

    init_ckpt = storage / WEIGHTS_ONLY_CKPT_NAME
    if weights_only_on_cold_start and init_ckpt.is_file():
        pylogger.info("found weights-only init checkpoint: %s", init_ckpt)
        return init_ckpt, True

    return None, False
