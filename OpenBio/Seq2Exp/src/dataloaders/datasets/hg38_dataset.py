"""Dataset for sampling arbitrary intervals from the human genome.

"""

import math
from pathlib import Path

import pandas as pd
import torch
from pyfaidx import Fasta

from src.dataloaders.utils.mlm import mlm_getitem
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement

MAX_ALLOWED_LENGTH = 2 ** 20


class FastaInterval:
    """Retrieves sequences from a fasta file given a chromosome and start/end indices."""
    def __init__(
            self,
            *,
            fasta_file,
            return_seq_indices=False,
            rc_aug=False,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "Path to fasta file must exist!"

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.rc_aug = rc_aug

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])

    @staticmethod
    def _compute_interval(start, end, max_length, i_shift):
        if max_length == MAX_ALLOWED_LENGTH:
            return start, end
        if max_length < MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0
            return start + i_shift * max_length, start + (i_shift + 1) * max_length
        else:
            raise ValueError(f"`max_length` {max_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!")

    def __call__(
            self,
            chr_name,
            start,
            end,
            max_length,
            i_shift,
            return_augs=False,
    ):
        """
        max_length passed from dataset, not from init
        """
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        start, end = self._compute_interval(start, end, max_length, i_shift)

        if end > chromosome_length:
            # Shift interval down
            start = start - (end - chromosome_length)
            end = chromosome_length
            assert start == chromosome_length - max_length

        if start < 0:
            # Shift interval up
            end = end - start
            start = 0
            assert end == max_length

        if end > chromosome_length:
            # This may occur if start + MAX_ALLOWED_LENGTH extends beyond the end of the chromosome
            start = chromosome_length - max_length
            end = chromosome_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        return seq


class HG38Dataset(torch.utils.data.Dataset):
    """Loop through bed file, retrieve (chr, start, end), query fasta file for sequence."""

    def __init__(
            self,
            split,
            bed_file,
            fasta_file,
            max_length,
            mlm=False,
            mlm_probability=0.15,
            pad_max_length=None,
            tokenizer=None,
            tokenizer_name=None,
            add_eos=False,
            return_seq_indices=False,
            rc_aug=False,
            return_augs=False,
    ):
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        if self.mlm and self.mlm_probability <= 0.0:
            raise ValueError(f"`mlm_probability` has to be > 0.0, got {self.mlm_probability}.")
        if self.mlm:
            # TODO: see if this helps
            # self.eligible_replacements = torch.tensor(
            #     tokenizer("ACGT", add_special_tokens=False)["input_ids"], dtype=torch.long
            # )
            self.eligible_replacements = None
        else:
            self.eligible_replacements = None
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos

        if max_length <= MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0, f"`max_length` must be a power of 2!"
            self.shifts = MAX_ALLOWED_LENGTH // max_length
        else:
            raise ValueError(f"`max_length` {max_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!")

        bed_path = Path(bed_file)
        assert bed_path.exists(), "Path to .bed file must exist!"

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep="\t", names=["chr_name", "start", "end", "split"])
        # select only split df
        self.df = df_raw[df_raw["split"] == split]
        # Update end points so that sequences are all length == MAX_ALLOWED_LENGTH
        self.df.loc[:, "end"] = self.df["start"] + MAX_ALLOWED_LENGTH

        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            return_seq_indices=return_seq_indices,
            rc_aug=rc_aug
        )

    @staticmethod
    def replace_value(x, old_value, new_value):
        """Helper for replacing values in a tensor."""
        return torch.where(x == old_value, new_value, x)

    def __len__(self):
        return len(self.df) * self.shifts

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row_idx, shift_idx = idx // self.shifts, idx % self.shifts
        row = self.df.iloc[row_idx]
        chr_name, start, end = (row.iloc[0], row.iloc[1], row.iloc[2])

        seq = self.fasta(
            chr_name,
            start,
            end,
            max_length=self.max_length,
            i_shift=shift_idx,
            return_augs=self.return_augs,
        )
        if end - start != MAX_ALLOWED_LENGTH:
            print(row, "\nLength: ", end - start)

        if self.tokenizer_name == "char":
            seq = self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
                add_special_tokens=False
            )

            seq = seq["input_ids"]  # get input_ids

            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq.append(self.tokenizer.sep_token_id)

        elif self.tokenizer_name == "bpe":
            seq = self.tokenizer(
                seq,
                # add_special_tokens=False,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            )
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens

        # convert to tensor
        seq = torch.LongTensor(seq)

        # replace N token with a pad token, so we can ignore it in the loss
        seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int["N"], self.tokenizer.pad_token_id)

        if self.mlm:
            data, target = mlm_getitem(
                seq,
                mlm_probability=self.mlm_probability,
                contains_eos=self.add_eos,
                tokenizer=self.tokenizer,
                eligible_replacements=self.eligible_replacements,
            )

        else:
            data = seq[:-1].clone()
            target = seq[1:].clone()

        return data, target
