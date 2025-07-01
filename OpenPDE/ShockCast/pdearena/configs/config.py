from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, Dict, List
from pdearena.utils.constants import Paths
import math
import os

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # model + data
    time_history: int
    time_future: int
    time_gap: int
    n_input_fields: int
    n_output_fields: int
    trajlen: int
    n_spatial_dim: int

    # model
    model: str
    pretrained_path: Optional[str] = None
    plmodel: str
    max_num_steps: int
    lr: float
    lr_scheduler: str
    scheduler_args: Dict
    optimizer: str
    optimizer_args: Dict
    normalize: bool
    predict_diff: bool
    weight_decay: float
    loss: str

    # data
    task: str
    time_dependent: bool
    fields: List[str]
    data_dir: str
    onestep_batch_size: int
    rollout_batch_size: int
    pin_memory: bool
    num_workers: int
    train_limit_trajectories: int
    valid_limit_trajectories: int
    test_limit_trajectories: int
    
    # train
    use_gradient_checkpoint: bool = False
    seed: int = 42
    save_dir: str
    checkpoint_monitor: str
    check_val_every_n_epoch: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    epochs: int
    resume_from_checkpoint: Optional[str] = None
    devices: int
    log_losses_t: int = None
    strategy: str = "ddp"

    def model_post_init(self, __context: Any) -> None:
        if self.log_losses_t is None:
            self.log_losses_t = self.max_num_steps
        batch_size = self.onestep_batch_size
        devices = self.devices
        if batch_size % devices != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by devices ({devices})")
        self.onestep_batch_size = batch_size // devices
        self.rollout_batch_size = self.rollout_batch_size // devices

class CFDConfig(Config):
    nx: int
    ny: int
    xspacing: float
    yspacing: float
    sim_time: float
    dt_thresh: Optional[float] = None
    time_coarsening: int

    xpad: int = 0
    ypad: int = 0

    include_derivatives: bool = False
    include_cfl_features: bool = False
    noise_level: float = 0.0

    normalize: bool = True
    predict_diff: bool = False
    dt_norm: bool = False

    time_future: int = 1
    time_gap: int = 0
    n_spatial_dim: int = 2
    time_dependent: bool = True
    
    log_losses_dt: float

    pin_memory: bool = True
    time_history: int = 1
    num_workers: int = 8
    train_limit_trajectories: int = -1
    valid_limit_trajectories: int = -1
    test_limit_trajectories: int = -1
    lr_scheduler: str = "cosine_step"
    scheduler_args: Dict = dict()
    optimizer: str = "AdamW"
    optimizer_args: Dict = dict()
    seed: int = 42
    save_dir: str = Paths.output
    checkpoint_monitor: str = "valid/unrolled_loss_mean"
    loss: str = "rel"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.rollout_batch_size = 1
        self.trajlen = math.ceil(self.trajlen / self.time_coarsening)

class PretrainedCFDConfig(CFDConfig):
    cfl_ckpt: str
    cfl_cfg: str = None
    solver_ckpt: str = None
    solver_cfg: str = None
    vel_inds: List[int] = None
    plmodel: str = "shock_cast"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.vel_inds = [i for i, field in enumerate(self.fields) if "vel" in field.lower()]
        if len(self.vel_inds) != 2:
            raise ValueError(f"Expected 2 velocity fields, found {len(self.vel_inds)}")
        self.cfl_cfg = self.cfg_from_ckpt(self.cfl_ckpt)
        if self.solver_ckpt is not None:
            self.solver_cfg = self.cfg_from_ckpt(self.solver_ckpt)

    def cfg_from_ckpt(self, ckpt_path: str):
        return os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))