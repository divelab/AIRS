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
from pdearena.configs.config_blast import BlastConfig
from pdearena.utils.constants import Paths, BlastConstants
from pdearena.data.process.blast_data_grid import (
    TIME, 
    RATIO
)
import random
import numpy as np

START = 0

TRAIN_INDS = [  2,   3,   4,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
         17,  18,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
         33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
         48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  61,  62,
         63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  77,  78,
         79,  80,  81,  82,  83,  84,  85,  86,  87,  89,  90,  91,  92,  93,
         94,  95,  97,  98,  99, 100]
EVAL_INDS = [ 5, 19, 32, 47, 59, 67, 76, 88, 96]

MEAN = "mean"
SD = "sd"
SHIFT = "shift"
SCALE = "scale"
STAT_MAP = {MEAN: "MEAN", SD: "SD", SHIFT: "SHIFT", SCALE: "SCALE"}


FIELDS = 'pressure', 'density', 'xVel', 'yVel', 'temperature', 'Mach_Number'

SOUND_SPEED_MEAN = torch.tensor(317.4776916503906).float()
SOUND_SPEED_SD = torch.tensor(85.56594848632812).float()
XVEL_MAG_MEAN = torch.tensor(115.70452117919922).float()
XVEL_MAG_SD = torch.tensor(111.01715850830078).float()
YVEL_MAG_MEAN = torch.tensor(115.70951080322266).float()
YVEL_MAG_SD = torch.tensor(111.01481628417969).float()
WAVE_SPEED_MEAN = torch.tensor(477.697265625).float()
WAVE_SPEED_SD = torch.tensor(125.2004623413086).float()
FIELD_MEAN = {'Mach_Number': 0.6590468287467957,
 'density': 5.059147357940674,
 'pressure': 384963.9375,
 'temperature': 269.07318115234375,
 'xVel': 36.62590789794922,
 'yVel': 36.625}
FIELD_SD = {'Mach_Number': 0.578612208366394,
 'density': 5.150228977203369,
 'pressure': 449536.78125,
 'temperature': 144.78753662109375,
 'xVel': 156.11178588867188,
 'yVel': 156.11402893066406}
DIFF_MEAN = {'Mach_Number': 0.01021216344088316,
 'density': -8.544474985683337e-05,
 'pressure': -533.6358642578125,
 'temperature': 0.019504981115460396,
 'xVel': -2.559070587158203,
 'yVel': -2.5586423873901367}
DIFF_SD = {'Mach_Number': 0.1847444772720337,
 'density': 1.3052656650543213,
 'pressure': 119152.9140625,
 'temperature': 51.342952728271484,
 'xVel': 53.50609588623047,
 'yVel': 53.50682067871094}
DIFF_DT_MEAN = {'Mach_Number': 90.53337860107422,
 'density': -0.828594982624054,
 'pressure': -4758189.0,
 'temperature': 1503.470458984375,
 'xVel': -21005.392578125,
 'yVel': -21000.203125}
DIFF_DT_SD = {'Mach_Number': 1703.38037109375,
 'density': 12728.0927734375,
 'pressure': 1174653952.0,
 'temperature': 493743.84375,
 'xVel': 502577.25,
 'yVel': 502579.15625}
YDERIV_MEAN = {'Mach_Number': -0.0029211656656116247,
 'density': -0.023317022249102592,
 'pressure': 662.228515625,
 'temperature': 2.050149917602539,
 'xVel': -0.6546342372894287,
 'yVel': -0.0012713709147647023}
YDERIV_SD = {'Mach_Number': 0.05153568834066391,
 'density': 0.508761465549469,
 'pressure': 31423.740234375,
 'temperature': 19.9741153717041,
 'xVel': 11.187089920043945,
 'yVel': 15.5509033203125}
XDERIV_MEAN = {'Mach_Number': -0.00292291515506804,
 'density': -0.023348385468125343,
 'pressure': 662.3056640625,
 'temperature': 2.0535550117492676,
 'xVel': -0.0012925831833854318,
 'yVel': -0.6552644371986389}
XDERIV_SD = {'Mach_Number': 0.05153092369437218,
 'density': 0.5090032815933228,
 'pressure': 31428.12890625,
 'temperature': 19.98293113708496,
 'xVel': 15.550966262817383,
 'yVel': 11.18224811553955}
DELTA_T_STATS = {MEAN: torch.tensor(0.00011242056643823162).float(), SD: torch.tensor(2.1944772015558556e-05).float(), SHIFT: torch.tensor([0.0]).float(), SCALE: torch.tensor(0.00019949255511164665).float()}
RATIO_STATS = {MEAN: torch.tensor(25.588335037231445).float(), SD: torch.tensor(13.811295509338379).float(), SHIFT: torch.tensor([0.0]).float(), SCALE: torch.tensor(49.51499938964844).float()}
# time delta range: 7.151532918214798e-05, 0.00019949255511164665
# ratio range: (tensor([1.9850]), tensor([49.5150]))
# nt range: 26, 50

class BlastDatasetOpener(dp.iter.IterDataPipe):
    """DataPipe to load Blast dataset."""
    def __init__(self, dp, mode: str, args: BlastConfig, limit_trajectories: Optional[int] = None, normalize: bool = True) -> None:
        super().__init__()
        self.args = args
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

        self.use_ratio = args.use_ratio

    def len(self):
        # Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_work
        # ers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.
        return self.num

    def norm(self, ratio):
        if self.normalize:
            ratio = (ratio - RATIO_STATS[SHIFT]) / RATIO_STATS[SCALE]
        return ratio

    def __iter__(self):
        training = self.mode == "train" and self.normalize
        for path in self.dp:
            data_dict = dict()
            keys = [RATIO, *self.fields]
            with h5py.File(path, "r") as f:
                times = torch.from_numpy(f[TIME][:]).float()
                nt = len(times)
                start = 1
                end = nt - 1
                if training:
                    start = start + self.sys_rand.randint(0, self.time_coarsening - 1)
                times = times[start:end:self.time_coarsening]
                inds = torch.arange(nt)[start:end:self.time_coarsening]
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

                ratio = data_dict[RATIO]
                u = torch.stack([field_data for field, field_data in data_dict.items() if field in self.fields], dim=1)
                u = u.flatten(-2).permute(2, 0, 1)  # [N, T, C]
                ratio = self.norm(
                    ratio=ratio
                )
                z = []
                if self.use_ratio:
                    z.append(ratio)
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

onestep_train_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train",
    onestep=True,
)
trajectory_train_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
    filter_fn=_train_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="train_rollout",
    onestep=False,
)
onestep_valid_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=True,
)
trajectory_valid_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
    filter_fn=_valid_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="valid",
    onestep=False,
)

onestep_test_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
    filter_fn=_test_filter,
    lister=dp.iter.FileLister,
    sharder=dp.iter.ShardingFilter,
    mode="test",
    onestep=True,
)

trajectory_test_datapipe_blast_grid = functools.partial(
    build_datapipes,
    dataset_opener=BlastDatasetOpener,
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
    from pdearena.data.twod.datapipes.blast2d import *
    import os
    torch.manual_seed(1)
    from pdearena.data.datamodule import PDEDataModule
    from pdearena.utils.metrics import ErrorStats, MinMax
    from pdearena.configs.registry import get_config
    from tqdm import tqdm
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # %%
    args: BlastConfig = get_config("blast_unetmod64v2")
    args.devices = 1
    args.time_coarsening = 1
    args.fields = FIELDS
    args.use_ratio = True
    datamod = PDEDataModule(args)
    datamod.setup("train")    
    
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
    stats_ratio = init_stats()
    accumulate = lambda stats, val: [stats["meansd_stats"](val), stats["range_stats"](val)]
    delta_t = []
    ratios = []
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
        ratio = sol.z.flatten()
        accumulate(stats_ratio, ratio.view(-1, 1))
        ratios.append(ratio)
    dy.shape
    u = u.unflatten(0, [args.ny, args.nx]).cpu()
    u.shape
    args.fields
    # %%

    # import matplotlib.pyplot as plt
    # tidx = 10
    # fidx = 1
    # plt.imshow(u[..., tidx, fidx], origin="lower")
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
    # # %%
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
    for name, stat in dict(DELTA_T_STATS=stats_delta_t, RATIO_STATS=stats_ratio).items():
        mean_sd_stat = stat["meansd_stats"].compute()
        mean = mean_sd_stat["mean"]
        sd = mean_sd_stat["sd"]
        range_stat = stat["range_stats"].compute()
        scale = range_stat["max"]
        stat = {MEAN: mean, SD: sd, SHIFT: torch.zeros(1), SCALE: scale}
        print(f"{name} = {str_stats(stat)}")
    
    print(f"time delta range: {min(delta_t)}, {max(delta_t)}")
    print(f"ratio range: {min(ratios), max(ratios)}")
    print(f"nt range: {min(nt)}, {max(nt)}")

