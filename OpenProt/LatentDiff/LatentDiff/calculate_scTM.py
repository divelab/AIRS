import numpy as np
import os
import sys
sys.path.append("../protein_autoencoder/")
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import radius_graph
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import argparse
from torch_geometric.data import Data
from utils import RMSD
from datasets_config import pdb_protein
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import re
import shutil
import itertools
import subprocess
import multiprocessing
import logging
from typing import List, Tuple
import numpy as np

device = torch.device('cuda:0')

# run_tmalign is adapted from https://github.com/microsoft/foldingdiff/blob/ac28993b0a3f89440f5c65ef6c37c5db342bdae1/foldingdiff/tmalign.py
def run_tmalign(query: str, reference: str, fast: bool = False) -> float:
    """
    Run TMalign on the two given input pdb files
    """
    assert os.path.isfile(query)
    assert os.path.isfile(reference)

    # Check if TMalign is installed
    exec = shutil.which("TMalign")
    if not exec:
        raise FileNotFoundError("TMalign not found in PATH")

    # Build the command
    cmd = f"{exec} {query} {reference}"
    if fast:
        cmd += " -fast"
    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        logging.warning(f"Tmalign failed on {query}|{reference}, returning NaN")
        return np.nan

    # Parse the outpu
    score_lines = []
    for line in output.decode().split("\n"):
        if line.startswith("TM-score"):
            score_lines.append(line)

    # Fetch the chain number
    key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
    score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
    results_dict = {key_getter(s): score_getter(s) for s in score_lines}
    # print(results_dict)
    return results_dict["Chain_2"]  # Normalize by reference length


def calculate_scTM(root, save_path=None, num_samples=100):


    folder = 'generated'
    src = f"./sample/omegafold_predictions/"
    dst = os.path.join(root, folder, 'ProteinMPNN/omegafold_predictions', '')

    # generated sequence is predicted by ProteinMPNN
    # calculate scTM score
    scTM_list = []
    best_seq = []

    for i in range(num_samples):
        best_score = 0
        seq = 0
        for j in range(8):

            query_pdb = os.path.join(root, folder, f"pdbs/generated_{i}.pdb")
            reference_pdb = os.path.join(root, folder, f"ProteinMPNN/omegafold_predictions/generate{i}_seq{j}.pdb")

            # print(i,j)
            try:
                # scTM = tmalign.run_tmalign(query_pdb, reference_pdb)
                scTM = run_tmalign(query_pdb, reference_pdb)
            except:
                print(i, j)
                print(query_pdb)
                print(reference_pdb)
                continue
            if scTM > best_score:
                best_score = scTM
                seq = j

        best_seq.append(seq)
        scTM_list.append(best_score)

    title_font = {'fontsize': 30}
    axis_font = {'fontsize': 30}
    tick_font = {'fontsize': 20}

    # plot scTM score histogram
    fig, ax = plt.subplots(figsize=(9, 8))
    _, _, _ = ax.hist(scTM_list)
    ax.axvline(x=0.5, color='0.8', ls='--', lw=2)
    # ax.legend()
    ax.set_title('scTM scores', **title_font)
    ax.set_xlabel('Score value', **axis_font)
    ax.set_ylabel('Count', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    fig.savefig(os.path.join(save_path, 'scTM.pdf'))

    # get length of generated proteins and calculate percentage of scTM>0.5 in each length range
    path = os.path.join(root, 'generated', 'seq_len.txt')
    length_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            length = int(re.findall(r">seq[0-9]+ ([0-9]+)", line.rstrip())[0])
            length_list.append(length)
    length_arr = np.array(length_list)


    with open(os.path.join(save_path, 'scTM.txt'), 'w') as f:

        scTM_arr = np.array(scTM_list)
        percentage = (scTM_arr > 0.5).sum() / scTM_arr.size
        print('Total percentage scTM > 0.5: {:.4}, {}/{}'.format( percentage, (scTM_arr > 0.5).sum(), scTM_arr.size) )
        f.write(f'Total percentage scTM > 0.5: {percentage}, {(scTM_arr > 0.5).sum()}/{scTM_arr.size}\n')

        scTM_part_1 = scTM_arr[length_arr <= 70]
        percentage = (scTM_part_1 > 0.5).sum() / scTM_part_1.size
        print('50~70 percentage scTM > 0.5: {:.4}, {}/{}'.format( percentage, (scTM_part_1 > 0.5).sum(), scTM_part_1.size ) )
        f.write(f'50~70 percentage scTM > 0.5: {percentage}, {(scTM_part_1 > 0.5).sum()}/{scTM_part_1.size}\n')

        scTM_part_2 = scTM_arr[length_arr > 70]
        percentage = (scTM_part_2 > 0.5).sum() / scTM_part_2.size
        print('70~128 percentage scTM > 0.5: {:.4}, {}/{}'.format( percentage, (scTM_part_2 > 0.5).sum(), scTM_part_2.size ) )
        f.write(f'70~128 percentage scTM > 0.5: {percentage}, {(scTM_part_2 > 0.5).sum()}/{scTM_part_2.size}\n')


    idx = np.where(scTM_arr > 0.5)[0]
    best_seq_idx = np.array(best_seq)[idx]


    for num, seq in zip(idx, best_seq_idx):

        query_pdb = os.path.join(root, folder, f"pdbs/generated_{num}.pdb")
        reference_pdb = os.path.join(root, folder, f"ProteinMPNN/omegafold_predictions/generate{num}_seq{seq}.pdb")

        save_folder = os.path.join(root, folder, 'good_results', '')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        query_dst = os.path.join(save_folder, f"generate{num}_scTM_{scTM_arr[num]}_len_{length_arr[num]}.pdb")
        shutil.copyfile(query_pdb, query_dst)

        filename = f'generate{num}_seq{seq}.pdb'
        result = open(save_folder + "/" + filename[:-4] + "_converted.pdb", 'w')
        with open(reference_pdb, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ' CA ' in line:
                    result.write(line)
        result.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein generation")
    parser.add_argument('--root', type=str, default='', help='')
    parser.add_argument('--suffix', type=str, default='', help='')
    parser.add_argument('--save_folder', type=str, default='', help='')
    parser.add_argument('--num_samples', type=int, default=100, help='')


    args = parser.parse_args()

    calculate_scTM(root=args.root, save_path=args.root, num_samples=args.num_samples)




