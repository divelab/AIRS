from pathlib import Path
import pandas as pd
import torch
import numpy as np
import pyBigWig
import tabix
import os
from selene_sdk.targets import Target
from tqdm import tqdm
import random

from src.dataloaders.utils.selene_utils import MemmapGenome
from src.tasks.utils import index_mapping


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        blacklists：those gene region that should be excluded. e.g., repeat region, always false positive, etc.
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            # make blacklist = 0 or replacement value
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class PromoterDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            tokenizer_name='char',
            seqlength=1024,
            split="train",
            n_tsses=100000,
            rand_offset=0,
            load_prob=False,
            strand='both',
            condition='no_cond',
            normalize_method='normal',
            reverse_aug=True,
            rc_aug=True,
    ):
        self.shuffle = False

        class ModelParameters:
            seifeatures_file = '../../../data/promoter_design/target.sei.names'
            seimodel_file = '../../../data/promoter_design/best.sei.model.pth.tar'

            ref_file = '../../../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
            ref_file_mmap = '../../../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
            tsses_file = '../../../data/promoter_design/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

            fantom_files = [
                "../../../data/promoter_design/agg.plus.bw.bedgraph.bw",
                "../../../data/promoter_design/agg.minus.bw.bedgraph.bw"
            ]
            fantom_blacklist_files = [
                "../../../data/promoter_design/fantom.blacklist8.plus.bed.gz",
                "../../../data/promoter_design/fantom.blacklist8.minus.bed.gz"
            ]

            n_time_steps = 400

            random_order = False
            speed_balanced = True
            ncat = 4
            num_epochs = 200

            lr = 5e-4

        self.config = ModelParameters()

        self.tsses = pd.read_table(self.config.tsses_file, sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]

        if strand == 'pos':
            self.tsses = self.tsses[self.tsses['strand'] == '+']
        elif strand == 'neg':
            self.tsses = self.tsses[self.tsses['strand'] == '-']

        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength
        self.load_prob = load_prob
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.condition = condition
        self.normalize_method = normalize_method
        self.reverse_aug = reverse_aug
        self.rc_aug = rc_aug

        self.init_worker()

    def init_worker(self):
        self.genome = MemmapGenome(
            input_path=self.config.ref_file,
            memmapfile=self.config.ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            self.config.fantom_files,
            ['cage_plus', 'cage_minus'],
            (2000,),
            self.config.fantom_blacklist_files
        )
        self.chr_lens = self.genome.get_chr_lens()

        # normalize the signals
        if self.normalize_method == 'normal':
            if self.split == 'train':
                all_signals = self.get_all_signals()
                self.mean = np.mean(all_signals)
                self.std = np.std(all_signals)
            elif self.split == 'valid' or self.split == 'test':
                self.mean, self.std = 0, 1

    def get_all_signals(self):
        all_elements = []
        for tssi in range(len(self)):
            chrm, pos, strand = (self.tsses['chr'].values[tssi],
                                 self.tsses['TSS'].values[tssi],
                                 self.tsses['strand'].values[tssi])
            offset = 1 if strand == '-' else 0

            offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)

            start = pos - int(self.seqlength / 2) + offset
            end = pos + int(self.seqlength / 2) + offset
            signal = self.tfeature.get_feature_data(chrm, start, end)
            if strand == '-':
                signal = signal[::-1, ::-1].copy()
            flattened_array = signal.flatten()
            all_elements.extend(flattened_array)
        final_array = np.array(all_elements)
        return final_array

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = (self.tsses['chr'].values[tssi],
                             self.tsses['TSS'].values[tssi],
                             self.tsses['strand'].values[tssi])
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)

        start = pos - int(self.seqlength / 2) + offset
        end = pos + int(self.seqlength / 2) + offset

        signal = self.tfeature.get_feature_data(chrm, start, end)  # 2 * 1024
        if strand == '-':
            signal = signal[::-1, ::-1].copy()

        if self.normalize_method == 'normal':
            signal = (signal - self.mean) / self.std

        if self.load_prob:
            # use concat token + signal embedding
            seq = self.genome.get_encoding_from_coords(chrm, start, end, strand)  # 1024 * 4
            # # add token N as a new axis
            # seq_5d = torch.zeros((1024, 5), dtype=seq.dtype)
            # seq_5d[:, :4] = seq
            # mask_N = (seq.sum(dim=1) == 0)
            # seq_5d[mask_N, 4] = 1
            # seq = seq_5d

            input_embed = np.concatenate([seq, signal.T], axis=-1).astype(np.float32)
            input_embed = torch.tensor(input_embed, dtype=torch.float32)

            # sequence tensor - target
            seq_str = self.genome.get_str_seq(chrm, start, end, strand)
            if self.tokenizer_name == 'char':
                seq = self.tokenizer(seq_str, padding="max_length",
                                     max_length=self.seqlength, add_special_tokens=False)['input_ids']
            seq = torch.LongTensor(seq)
            seq = torch.LongTensor([index_mapping[int(label)] for label in seq])
            return input_embed, torch.tensor([]), seq

        else:
            # rc_augmentation
            if self.rc_aug:
                rand_num = random.randint(0, 1)
                if rand_num == 0:
                    pass
                elif rand_num == 1:
                    strand = '-' if strand == '+' else '+'
                    signal = signal[::-1, ::-1].copy()

            seq_str = self.genome.get_str_seq(chrm, start, end, strand)
            if self.tokenizer_name == 'char':
                seq = self.tokenizer(seq_str, padding="max_length",
                                     max_length=self.seqlength, add_special_tokens=True)['input_ids']
            seq = torch.LongTensor(seq)

            # reverse augmentation
            if self.reverse_aug:
                rand_num = random.randint(0, 1)
                if rand_num == 0:
                    pass
                elif rand_num == 1:
                    seq[1:-1] = seq[1:-1].flip(dims=[0])
                    signal = signal[:, ::-1].copy()
                else:
                    NotImplementedError()

            data = seq[:-1].clone()
            target = seq[1:].clone()

            signal = signal.T
            signal = torch.tensor(signal, dtype=torch.float32)

            assert len(data) == self.seqlength + 1
            assert len(target) == self.seqlength + 1
            assert len(signal) == self.seqlength
            return {
                "data": data,
                "target": target,
                "condition": signal,
            }
            # return data, target, signal
            # concat_data = torch.concat([data.unsqueeze(-1), signal], dim=-1)
            #
            # return concat_data, target

    def reset(self):
        np.random.seed(0)