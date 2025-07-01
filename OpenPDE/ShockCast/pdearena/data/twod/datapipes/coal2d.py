# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import functools
from typing import Optional
import os
import h5py
import torch
import torchdata.datapipes as dp
from torch_geometric.data import Data
from pdearena.data.datapipes_common import build_datapipes
from pdearena.configs.config_coal import CoalConfig
from pdearena.utils.constants import Paths, CoalConstants
from pdearena.data.process.coal_data_grid import (
    TIME, 
    DIAMETER, 
    MACH
)
import random
import numpy as np

START = 0

TRAIN_INDS = [  1,   3,   4,   5,   6,   7,   8,  10,  11,  12,  13,  14,  15,  16,
         17,  18,  19,  20,  22,  24,  25,  26,  27,  28,  29,  30,  31,  32,
         33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  44,  45,  46,  47,
         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
         62,  63,  64,  66,  67,  68,  69,  70,  71,  73,  74,  75,  76,  78,
         79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  93,
         94,  95,  96,  98,  99, 100]
EVAL_INDS = [ 2,  9, 21, 23, 43, 65, 72, 77, 92, 97]

MEAN = "mean"
SD = "sd"
SHIFT = "shift"
SCALE = "scale"
STAT_MAP = {MEAN: "MEAN", SD: "SD", SHIFT: "SHIFT", SCALE: "SCALE"}


FIELDS = 'pressure', 'volume_fraction_coal', 'xVel_gas', 'yVel_gas', 'temperature_gas'

SOUND_SPEED_MEAN = torch.tensor(426.93133544921875).float()
SOUND_SPEED_SD = torch.tensor(42.06951141357422).float()
XVEL_MAG_MEAN = torch.tensor(268.174072265625).float()
XVEL_MAG_SD = torch.tensor(153.15032958984375).float()
YVEL_MAG_MEAN = torch.tensor(15.330653190612793).float()
YVEL_MAG_SD = torch.tensor(18.43839454650879).float()
WAVE_SPEED_MEAN = torch.tensor(695.2813110351562).float()
WAVE_SPEED_SD = torch.tensor(178.580078125).float()
FIELD_MEAN = {'pressure': 419983.4375,
 'temperature_gas': 458.0394592285156,
 'volume_fraction_coal': 0.020837584510445595,
 'xVel_gas': 267.71405029296875,
 'yVel_gas': 2.010812520980835}
FIELD_SD = {'pressure': 202988.46875,
 'temperature_gas': 89.15943908691406,
 'volume_fraction_coal': 0.09924868494272232,
 'xVel_gas': 153.95306396484375,
 'yVel_gas': 23.8947696685791}
DIFF_MEAN = {'pressure': 997.573486328125,
 'temperature_gas': 0.4656902849674225,
 'volume_fraction_coal': -7.20584430382587e-05,
 'xVel_gas': 0.5757118463516235,
 'yVel_gas': 0.005137663800269365}
DIFF_SD = {'pressure': 22432.953125,
 'temperature_gas': 8.856427192687988,
 'volume_fraction_coal': 0.006851378362625837,
 'xVel_gas': 16.091352462768555,
 'yVel_gas': 12.027840614318848}
DIFF_DT_MEAN = {'pressure': 103209080.0,
 'temperature_gas': 47890.33203125,
 'volume_fraction_coal': -7.0409674644470215,
 'xVel_gas': 58467.38671875,
 'yVel_gas': 1561.4239501953125}
DIFF_DT_SD = {'pressure': 2439475456.0,
 'temperature_gas': 955376.9375,
 'volume_fraction_coal': 670.2979736328125,
 'xVel_gas': 1724278.875,
 'yVel_gas': 1256912.375}
YDERIV_MEAN = {'pressure': 118.16006469726562,
 'temperature_gas': 0.5192707777023315,
 'volume_fraction_coal': -0.001993063371628523,
 'xVel_gas': 3.3539395332336426,
 'yVel_gas': 0.0001880847557913512}
YDERIV_SD = {'pressure': 4577.1533203125,
 'temperature_gas': 4.139290809631348,
 'volume_fraction_coal': 0.021669933572411537,
 'xVel_gas': 7.049168586730957,
 'yVel_gas': 3.0569581985473633}
XDERIV_MEAN = {'pressure': -647.6072387695312,
 'temperature_gas': -0.39158615469932556,
 'volume_fraction_coal': 0.00037611511652357876,
 'xVel_gas': 0.19325263798236847,
 'yVel_gas': -0.05775989219546318}
XDERIV_SD = {'pressure': 3778.51904296875,
 'temperature_gas': 3.757704019546509,
 'volume_fraction_coal': 0.015758169814944267,
 'xVel_gas': 3.735201835632324,
 'yVel_gas': 2.5228471755981445}
DELTA_T_STATS = {MEAN: torch.tensor(1.1245557288930286e-05).float(), SD: torch.tensor(2.0462359771045158e-06).float(), SHIFT: torch.tensor([0.0]).float(), SCALE: torch.tensor(1.956871710717678e-05).float()}
DIAMETER_STATS = {MEAN: torch.tensor(7.673081563552842e-05).float(), SD: torch.tensor(4.237297616782598e-05).float(), SHIFT: torch.tensor([0.0]).float(), SCALE: torch.tensor(0.00014965634909458458).float()}
MACH_STATS = {MEAN: torch.tensor(1.6416140794754028).float(), SD: torch.tensor(0.2648249566555023).float(), SHIFT: torch.tensor([0.0]).float(), SCALE: torch.tensor(2.0989537239074707).float()}
# time delta range: 7.813796401023865e-07, 1.956871710717678e-05
# mach range: (1.2013200521469116, 2.0989537239074707)
# diameter range: (1.0385100495113875e-06, 0.00014965634909458458)
# nt range: 186, 350

class CoalDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load Blast dataset."""
    def __init__(self, dp, mode: str, args: CoalConfig, limit_trajectories: Optional[int] = None, normalize: bool = True) -> None:
        super().__init__()
        self.args = args
        self.start_time = args.start_time
        self.nx = args.nx
        self.ny = args.ny
        self.fields = args.fields
        for field in self.fields:
            assert field in FIELDS
        self.time_coarsening = args.time_coarsening
        self.normalize = normalize
        self.dp = dp
        self.mode = mode
        self.limit_trajectories = limit_trajectories
        files = [p for p in self.dp]
        assert len(files) != 0
        self.num = len(files)
        self.args = args
        self.sys_rand = random.SystemRandom()

        if args.predict_diff:
            if args.dt_norm:
                yfield_mean = DIFF_DT_MEAN
                yfield_sd = DIFF_DT_SD
            else:
                yfield_mean = DIFF_MEAN
                yfield_sd = DIFF_SD
        else:            
            yfield_mean = FIELD_MEAN
            yfield_sd = FIELD_SD

        self.xmeans = torch.tensor([[FIELD_MEAN[field] for field in self.fields]], dtype=torch.float32)
        self.xsds = torch.tensor([[FIELD_SD[field] for field in self.fields]], dtype=torch.float32)    
        self.ymeans = torch.tensor([[yfield_mean[field] for field in self.fields]], dtype=torch.float32)
        self.ysds = torch.tensor([[yfield_sd[field] for field in self.fields]], dtype=torch.float32)

        self.xderivmeans = torch.tensor([[XDERIV_MEAN[field] for field in self.fields]], dtype=torch.float32)
        self.xderivsds = torch.tensor([[XDERIV_SD[field] for field in self.fields]], dtype=torch.float32)
        self.yderivmeans = torch.tensor([[YDERIV_MEAN[field] for field in self.fields]], dtype=torch.float32)
        self.yderivsds = torch.tensor([[YDERIV_SD[field] for field in self.fields]], dtype=torch.float32)

        self.delta_t_mean = torch.tensor([DELTA_T_STATS[MEAN]])
        self.delta_t_sd = torch.tensor([DELTA_T_STATS[SD]])
        self.delta_t_shift = torch.tensor([DELTA_T_STATS[SHIFT]])
        self.delta_t_scale = torch.tensor([DELTA_T_STATS[SCALE]])

        self.use_mach = args.use_mach
        self.use_diameter = args.use_diameter

        self.dt_thresh = args.dt_thresh

    def len(self):
        # Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_work
        # ers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.
        return self.num

    def norm(self, diameter, mach):
        if self.normalize:
            diameter = (diameter - DIAMETER_STATS[SHIFT]) / DIAMETER_STATS[SCALE]
            mach = (mach - MACH_STATS[SHIFT]) / MACH_STATS[SCALE]
        return diameter, mach

    def spacing_mask(self, times: torch.Tensor) -> torch.BoolTensor:
        mask  = torch.zeros_like(times, dtype=torch.bool)
        last_kept_time = -torch.inf
        for i, t in enumerate(times):
            if t - last_kept_time > self.dt_thresh:
                mask[i] = True
                last_kept_time = t
        assert times[mask].diff().min() > self.dt_thresh
        return mask

    def __iter__(self):
        training = self.mode == "train" and self.normalize
        for path in self.dp:
            data_dict = dict()
            keys = [DIAMETER, MACH, *self.fields]
            with h5py.File(path, "r") as f:
                times = torch.from_numpy(f[TIME][:]).float()
                nt = len(times)
                start = (times - self.start_time).abs().argmin() # START
                end = nt - 1
                if training:
                    start = start + self.sys_rand.randint(0, self.time_coarsening - 1)
                times = times[start:end:self.time_coarsening]
                inds = torch.arange(nt)[start:end:self.time_coarsening]
                # if not training and self.dt_thresh is not None and self.dt_thresh > 0:
                if self.dt_thresh is not None and self.dt_thresh > 0:
                    mask = self.spacing_mask(times)
                    times = times[mask]
                    inds = inds[mask]
                inds = inds.tolist()
                times = times - times[0]
                delta_t = torch.diff(times)
                nt = len(times)
                if training:

                    time_resolution = min(nt, self.args.trajlen)

                    # Max number of previous points solver can eat
                    reduced_time_resolution = time_resolution - self.args.time_history
                    # Number of future points to predict
                    max_start_time = reduced_time_resolution - self.args.time_future - self.args.time_gap

                    # Choose initial random time point at the PDE solution manifold
                    start_time = self.sys_rand.randint(0, max_start_time)

                    # Different starting points of one batch according to field_x(t_0), field_y(t_0), ...
                    end_time = start_time + self.args.time_history
                    target_start_time = end_time + self.args.time_gap
                    target_end_time = target_start_time + self.args.time_future

                    inds = inds[start_time:target_end_time]
                    times = times[start_time:target_end_time]
                    delta_t = torch.diff(times)

                for k in keys:
                    data = f[k]
                    if k in FIELDS:
                        data = torch.tensor(data[inds, :self.nx, :self.ny], dtype=torch.float32).transpose(1, 2)
                        assert data.shape[1] == self.ny
                        assert data.shape[2] == self.nx
                    else:
                        data = data[()]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        elif isinstance(data, np.ndarray):
                            data = torch.from_numpy(data).to(dtype=torch.float32)
                        elif isinstance(data, np.float64):
                            data = torch.tensor(data, dtype=torch.float32)

                    data_dict[k] = data

                diameter = data_dict[DIAMETER]
                mach = data_dict[MACH]
                u = torch.stack([field_data for field, field_data in data_dict.items() if field in self.fields], dim=1)
                u = u.flatten(-2).permute(2, 0, 1)  # [N, T, C]
                diameter, mach = self.norm(
                    diameter=diameter,
                    mach=mach
                )
                z = []
                if self.use_diameter:
                    z.append(diameter)
                if self.use_mach:
                    z.append(mach)
                z = torch.tensor([z], dtype=torch.float32)
                yield Data(
                    u=u, 
                    dt=delta_t, 
                    times=times, 
                    z=z,
                    mean_x=self.xmeans,
                    sd_x=self.xsds,
                    mean_y=self.ymeans,
                    sd_y=self.ysds,
                    mean_xderiv=self.xderivmeans,
                    sd_xderiv=self.xderivsds,
                    mean_yderiv=self.yderivmeans,
                    sd_yderiv=self.yderivsds,
                    delta_t_mean=self.delta_t_mean,
                    delta_t_sd=self.delta_t_sd,
                    delta_t_shift=self.delta_t_shift,
                    delta_t_scale=self.delta_t_scale,
                    sound_speed_mean=SOUND_SPEED_MEAN,
                    sound_speed_sd=SOUND_SPEED_SD,
                    xvel_mag_mean=XVEL_MAG_MEAN,
                    xvel_mag_sd=XVEL_MAG_SD,
                    yvel_mag_mean=YVEL_MAG_MEAN,
                    yvel_mag_sd=YVEL_MAG_SD,
                    wave_speed_mean=WAVE_SPEED_MEAN,
                    wave_speed_sd=WAVE_SPEED_SD,
                    nx=self.nx,
                    ny=self.ny
                )

def _filter(fname, inds):
    fname = os.path.basename(fname)
    return ("h5" in fname) and (get_file_idx(fname) in inds)
    
def _train_filter(fname):
    return _filter(fname=fname, inds=TRAIN_INDS)

def _eval_filter(fname):
    return _filter(fname=fname, inds=EVAL_INDS)

def _valid_filter(fname):
    return _eval_filter(fname)

def _test_filter(fname):
    return _eval_filter(fname)

onestep_train_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
    onestep=True,
)
trajectory_train_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train_rollout",
    onestep=False,
)
onestep_valid_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_coalSquare_grid = functools.partial(
    build_datapipes,
    dataset_opener=CoalDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=False,
)

def get_file_idx(fname):
    return int(fname.split("_")[1].split(".")[0])

if __name__ == "__main__":
    # %%
    from pdearena.data.twod.datapipes.coal2d import *
    import os
    torch.manual_seed(1)
    from pdearena.data.datamodule import PDEDataModule
    from pdearena.utils.metrics import ErrorStats, MinMax
    from pdearena.configs.registry import get_config
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"

    # %%
    args: CoalConfig = get_config("coal_unetmod64v2")
    args.devices = 1
    args.time_coarsening = 1
    args.fields = FIELDS
    datamod = PDEDataModule(args)
    datamod.setup("train")    
    
    # %%
    # file_ids = [get_file_idx(f) for f in os.listdir(args.data_dir)]
    # shuffle = torch.randperm(len(file_ids), generator=torch.Generator().manual_seed(12))
    # file_ids = [file_ids[i] for i in shuffle]
    # ntrain = int(len(file_ids) * 0.9)
    # train_inds = file_ids[:ntrain]
    # eval_inds = file_ids[ntrain:]
    # len(train_inds), len(eval_inds)
    # # %%
    # files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".h5")]
    # train_files = [f for f in files if get_file_idx(os.path.basename(f)) in train_inds]
    # eval_files = [f for f in files if get_file_idx(os.path.basename(f)) in eval_inds]
    # train_machs = []
    # train_diameters = []
    # for file in train_files:
    #     with h5py.File(file, "r") as f:
    #         train_machs.append(f[MACH][()])
    #         train_diameters.append(f[DIAMETER][()])
    # train_machs = torch.tensor(train_machs)
    # train_diameters = torch.tensor(train_diameters)
    # eval_machs = []
    # eval_diameters = []
    # for file in eval_files:
    #     with h5py.File(file, "r") as f:
    #         eval_machs.append(f[MACH][()])
    #         eval_diameters.append(f[DIAMETER][()])
    # eval_machs = torch.tensor(eval_machs)
    # eval_diameters = torch.tensor(eval_diameters)
    # # %%
    # plt.scatter(train_machs, train_diameters, label="train")
    # plt.scatter(eval_machs, eval_diameters, label="eval")
    # %%

    dpipe = datamod.train_dp1.dp.source_datapipe   
    dpipe.mode = "calculate" 
    dpipe.normalize = False

    device = "cuda"
    nfields = len(args.fields)
    stats = ErrorStats(n_fields=nfields, track_sd=True).to(device)
    stats_sound_speed = ErrorStats(n_fields=1, track_sd=True).to(device)
    stats_xvel_mag = ErrorStats(n_fields=1, track_sd=True).to(device)
    stats_yvel_mag = ErrorStats(n_fields=1, track_sd=True).to(device)
    stats_wave_speed = ErrorStats(n_fields=1, track_sd=True).to(device)
    stats_diff = ErrorStats(n_fields=nfields, track_sd=True).to(device)
    stats_diff_dt = ErrorStats(n_fields=nfields, track_sd=True).to(device)
    stats_yderiv = ErrorStats(n_fields=nfields, track_sd=True).to(device)
    stats_xderiv = ErrorStats(n_fields=nfields, track_sd=True).to(device)
    init_stats = lambda: dict(meansd_stats=ErrorStats(n_fields=1, track_sd=True), range_stats=MinMax())
    stats_delta_t = init_stats()
    stats_mach = init_stats()
    stats_diameter = init_stats()
    accumulate = lambda stats, val: [stats["meansd_stats"](val), stats["range_stats"](val)]
    delta_t = []
    diameters = []
    machs = []
    nt = []
    # %%
    xvel_ind = [i for i, field in enumerate(args.fields) if "xvel" in field.lower()]
    yvel_ind = [i for i, field in enumerate(args.fields) if "yvel" in field.lower()]
    assert len(xvel_ind) == 1
    assert len(yvel_ind) == 1
    xvel_ind = xvel_ind[0]
    yvel_ind = yvel_ind[0]
    temp_inds = [i for i, field in enumerate(args.fields) if "temp" in field.lower()]
    temp_ind = temp_inds[0]
    assert len(temp_inds) == 1
    gamma = 1.4
    R = 287
    # %%
    for sol in tqdm(dpipe, total=len(TRAIN_INDS)):
        vals_fine = []
        u = sol.u.to(device)

        xvel = u[:, :, xvel_ind: xvel_ind + 1]
        yvel = u[:, :, yvel_ind: yvel_ind + 1]
        temp = u[:, :, temp_ind: temp_ind + 1]
        sound_speed = (gamma * R * temp) ** 0.5
        stats_sound_speed(sound_speed.reshape(-1, 1))

        xvel_mag = xvel.abs()
        stats_xvel_mag(xvel_mag.reshape(-1, 1))
        yvel_mag = yvel.abs()
        stats_yvel_mag(yvel_mag.reshape(-1, 1))
        wave_speed = torch.max(xvel_mag + sound_speed, yvel_mag + sound_speed)
        stats_wave_speed(wave_speed.reshape(-1, 1))


        dy, dx = torch.gradient(u.unflatten(dim=0, sizes=(sol.ny, sol.nx)), dim=[0, 1])
        # break
        stats_yderiv(dy.reshape(-1, 1, nfields))
        stats_xderiv(dx.reshape(-1, 1, nfields))

        dt = sol.dt.to(device)
        nt.append(len(sol.times))
        diff = u[:, 1:] - u[:, :-1]
        diff_dt = diff / dt.view(1, -1, 1)
        stats(u.reshape(-1, 1, nfields))
        stats_diff(diff.reshape(-1, 1, nfields))
        stats_diff_dt(diff_dt.reshape(-1, 1, nfields))
        accumulate(stats_delta_t, sol.dt)
        delta_t.extend(sol.dt.tolist())
        diameter, mach = sol.z.flatten().chunk(2)
        accumulate(stats_diameter, diameter)
        accumulate(stats_mach, mach)
        diameters.append(diameter.item())
        machs.append(mach.item())
        # break
    # %%
    # vel_inds = [i for i, field in enumerate(args.fields) if "vel" in field.lower()]
    # assert len(vel_inds) == 2
    # temp_inds = [i for i, field in enumerate(args.fields) if "temp" in field.lower()]
    # assert len(temp_inds) == 1
    # temp_ind = temp_inds[0]
    # vel = u[:, :, vel_inds]
    # temp = u[:, :, temp_ind: temp_ind + 1]
    # vel.shape, temp.shape
    # sound_speed = temp ** 0.5
    # wave_speed_x = vel[:, :, :1].abs() + sound_speed
    # wave_speed_y = vel[:, :, 1:].abs() + sound_speed
    # wave_speed_max = torch.max(wave_speed_x, wave_speed_y)
    # sound_speed.shape, wave_speed_x.shape, wave_speed_y.shape, wave_speed_max.shape 
    # %%
    dy.shape
    dy.shape
    u = u.unflatten(0, [args.ny, args.nx]).cpu()
    u.shape

    # %%
    # import matplotlib.pyplot as plt
    # tidx = 0
    # fidx = 3
    # plt.imshow(u[..., tidx, fidx].cpu(), origin="lower")
    # # %%
    # u.shape
    # # %%
    # dy, dx = torch.gradient(u.cpu(), dim=[0, 1])
    # dy.shape, dx.shape
    # # %%
    # (torch.stack(torch.gradient(u.cpu(), spacing=0.5, dim=[0,1])) - 2 * torch.stack([dy, dx])).norm()  #  == 2 * torch.stack(torch.gradient(u.cpu(), dim=[0, 1]))
    # # %%
    # plt.imshow(dy[..., tidx, fidx].cpu(), origin="lower")
    # plt.colorbar()
    # # %%
    # plt.imshow(dx[..., tidx, fidx].cpu(), origin="lower")
    # plt.colorbar()
    # %%
    # %%
    from pprint import pprint
    tensor_str = lambda x: f"torch.tensor({x}).float()"
    sound_speed_stats = stats_sound_speed.compute()
    sound_speed_mean =  tensor_str(sound_speed_stats["mean"].item())
    sound_speed_sd = tensor_str(sound_speed_stats["sd"].item())
    print(f"SOUND_SPEED_MEAN = {sound_speed_mean}")
    print(f"SOUND_SPEED_SD = {sound_speed_sd}")
    xvel_mag_stats = stats_xvel_mag.compute()
    xvel_mag_mean =  tensor_str(xvel_mag_stats["mean"].item())
    xvel_mag_sd = tensor_str(xvel_mag_stats["sd"].item())
    print(f"XVEL_MAG_MEAN = {xvel_mag_mean}")
    print(f"XVEL_MAG_SD = {xvel_mag_sd}")
    yvel_mag_stats = stats_yvel_mag.compute()
    yvel_mag_mean = tensor_str(yvel_mag_stats["mean"].item())
    yvel_mag_sd = tensor_str(yvel_mag_stats["sd"].item())
    print(f"YVEL_MAG_MEAN = {yvel_mag_mean}")
    print(f"YVEL_MAG_SD = {yvel_mag_sd}")
    wave_speed_stats = stats_wave_speed.compute()
    wave_speed_mean = tensor_str(wave_speed_stats["mean"].item())
    wave_speed_sd = tensor_str(wave_speed_stats["sd"].item())
    print(f"WAVE_SPEED_MEAN = {wave_speed_mean}")
    print(f"WAVE_SPEED_SD = {wave_speed_sd}")
    

    pde_stats = stats.compute()
    pde_mean = {k:v.item() for k, v in zip(args.fields, pde_stats["mean"])}
    pde_sd = {k: v.item() for k, v in zip(args.fields, pde_stats["sd"])}
    print("FIELD_MEAN = ", end="")
    pprint(pde_mean)
    print("FIELD_SD = ", end="")
    pprint(pde_sd)

    diff_stats = stats_diff.compute()
    diff_mean = {k:v.item() for k, v in zip(args.fields, diff_stats["mean"])}
    diff_sd = {k: v.item() for k, v in zip(args.fields, diff_stats["sd"])}
    print("DIFF_MEAN = ", end="")
    pprint(diff_mean)
    print("DIFF_SD = ", end="")
    pprint(diff_sd)

    diff_dt_stats = stats_diff_dt.compute()
    diff_dt_mean = {k:v.item() for k, v in zip(args.fields, diff_dt_stats["mean"])}
    diff_dt_sd = {k: v.item() for k, v in zip(args.fields, diff_dt_stats["sd"])}
    print("DIFF_DT_MEAN = ", end="")
    pprint(diff_dt_mean)
    print("DIFF_DT_SD = ", end="")
    pprint(diff_dt_sd)
    
    yderiv_stats = stats_yderiv.compute()
    yderiv_mean = {k:v.item() for k, v in zip(args.fields, yderiv_stats["mean"])}
    yderiv_sd = {k: v.item() for k, v in zip(args.fields, yderiv_stats["sd"])}
    print("YDERIV_MEAN = ", end="")
    pprint(yderiv_mean)
    print("YDERIV_SD = ", end="")
    pprint(yderiv_sd)

    xderiv_stats = stats_xderiv.compute()
    xderiv_mean = {k:v.item() for k, v in zip(args.fields, xderiv_stats["mean"])}
    xderiv_sd = {k: v.item() for k, v in zip(args.fields, xderiv_stats["sd"])}
    print("XDERIV_MEAN = ", end="")
    pprint(xderiv_mean)
    print("XDERIV_SD = ", end="")
    pprint(xderiv_sd)


    # print("SD_REGISTRY = {}".format(pde_sd))
    # stat_dict = {k:{MEAN: mean.item(), SD: sd.item()} for k, mean, sd in zip(args.fields, pde_stats["mean"], pde_stats["sd"])}
    # print("STAT_REGISTRY = {}".format(stat_dict))
    

    # %%
    str_stats = lambda stats: "{" + ", ".join([f"{STAT_MAP[stat_name]}: torch.tensor({stat.tolist()}).float()" for stat_name, stat in stats.items()]) + "}"
    for name, stat in dict(DELTA_T_STATS=stats_delta_t, DIAMETER_STATS=stats_diameter, MACH_STATS=stats_mach).items():
        mean_sd_stat = stat["meansd_stats"].compute()
        mean = mean_sd_stat["mean"]
        sd = mean_sd_stat["sd"]
        range_stat = stat["range_stats"].compute()
        scale = range_stat["max"]
        stat = {MEAN: mean, SD: sd, SHIFT: torch.zeros(1), SCALE: scale}
        print(f"{name} = {str_stats(stat)}")
    
    print(f"time delta range: {min(delta_t)}, {max(delta_t)}")
    print(f"mach range: {min(machs), max(machs)}")
    print(f"diameter range: {min(diameters), max(diameters)}")
    print(f"nt range: {min(nt)}, {max(nt)}")


