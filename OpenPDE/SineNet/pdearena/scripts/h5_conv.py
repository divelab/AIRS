# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import xarray as xr
import h5py
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

def collect_data2h5(datapath, batch, mode, save_dir):
    datals = os.path.join(datapath, "seed=*", "run*", "output.nc")
    # datals = os.path.join(datapath, "run*", "output.nc")
    print(f"Loading {datals} from {datapath}...")
    data = xr.open_mfdataset(datals, concat_dim="b", combine="nested", parallel=True)
    print("...loaded. Converting to numpy...")
    n = len(data['u'])
    data = {field:np.concatenate([field_data[idx].to_numpy()[None] for idx in tqdm(range(n))]) for field, field_data in tqdm(data.items())}
    print("...converted. Batching...")
    n_batches = n // batch + (n % batch != 0)
    os.makedirs(save_dir, exist_ok=True)
    for b_idx in tqdm(range(n_batches)):
        bstart = b_idx * batch
        bend = (b_idx + 1) * batch
        with h5py.File(os.path.join(save_dir, f"{mode}_{b_idx}.h5"), "w") as f:
            dataset = f.create_group(mode)
            for field, field_data in data.items():
                field_batch = field_data[bstart:bend]
                dataset.create_dataset(field, data=field_batch)
    print("...done")

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--savepath", type=str, required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--mode", type=str, required=True)
args = parser.parse_args()
collect_data2h5(datapath=args.path, batch=args.batch, mode=args.mode, save_dir=args.savepath)
