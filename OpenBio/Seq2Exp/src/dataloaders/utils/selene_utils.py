"""
This module provides the selene-based utilities for training and using
Orca sequence models for multiscale genome interaction prediction. This
module contains code from selene.
"""

import os
from collections import namedtuple

import pkg_resources
from functools import wraps

import pandas as pd
import numpy as np
import pyfaidx
from cooltools.lib.numutils import adaptive_coarsegrain
import cooler
import pyranges

from selene_sdk.sequences import Genome
from selene_sdk.samplers import OnlineSampler
from selene_sdk.utils import get_indices_and_probabilities
from selene_sdk.targets import Target
from tqdm import tqdm

SampleIndices = namedtuple("SampleIndices", ["indices", "weights"])

import random
import tabix

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}


def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class MemmapGenome(Genome):
    """
    Memmapped version of selene.sequence.Genome. Faster for sequence
    retrieval by storing all precomputed one-hot encodings in a memmapped
    file (~40G for human genome).

    The memmapfile can be an exisiting memmapped file or a path where you
    want to create the memmapfile. If the specified memmapfile does not
    exist, it will be created the first time you call any method of
    MemmapGenome or if MemmapGenome is initialized with `init_unpickable=True`.
    Therefore the first call will take some time for the
    creation of memmapfile if it does not exist. Also,  if
    memmapfile has not been created, be careful not to run multiple
    instances of MemmapGenome in parallel (such as with Dataloader),
    because as each process will try to create the file.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file, that is, a `*.fasta` file with
        a corresponding `*.fai` file in the same directory. This file
        should contain the target organism's genome sequence.
    init_unpickleable : bool, optional
        Default is False. If False, delay part of initialization code
        to executed only when a relevant method is called. This enables
        the object to be pickled after instantiation. `init_unpickleable` should
        be `False` when used when multi-processing is needed e.g. DataLoader.
    memmapfile : str or None, optional
        Specify the numpy.memmap file for storing the encoding
        of the genome. If memmapfile does not exist, it will be
        created when the encoding is requested for the first time.

    Attributes
    ----------
    genome : pyfaidx.Fasta
        The FASTA file containing the genome sequence.
    chrs : list(str)
        The list of chromosome names.
    len_chrs : dict
        A dictionary mapping the names of each chromosome in the file to
        the length of said chromosome.

    """

    def __init__(
            self,
            input_path,
            blacklist_regions=None,
            bases_order=None,
            init_unpicklable=False,
            memmapfile=None,
    ):
        self.initialized = False
        super().__init__(
            input_path, blacklist_regions=blacklist_regions, bases_order=bases_order,
        )
        self.memmapfile = memmapfile
        self.input_path = input_path
        self.blacklist_regions = blacklist_regions
        if init_unpicklable:
            self._unpicklable_init()

    def _unpicklable_init(self):
        if not self.initialized:
            print("Initializing Genome")
            self.genome = pyfaidx.Fasta(self.input_path)
            self.chrs = sorted(self.genome.keys())
            self.len_chrs = self._get_len_chrs()
            self._blacklist_tabix = None

            if self.blacklist_regions == "hg19":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk", "sequences/data/hg19_blacklist_ENCFF001TDO.bed.gz"
                    )
                )
            elif self.blacklist_regions == "hg38":
                self._blacklist_tabix = tabix.open(
                    pkg_resources.resource_filename(
                        "selene_sdk", "sequences/data/hg38.blacklist.bed.gz"
                    )
                )
            elif self.blacklist_regions is not None:  # user-specified file
                self._blacklist_tabix = tabix.open(self.blacklist_regions)

            self.lens = np.array([self.len_chrs[c] for c in self.chrs])
            self.inds = {
                c: ind for c, ind in zip(self.chrs, np.concatenate([[0], np.cumsum(self.lens)]))
            }
            if self.memmapfile is not None and os.path.isfile(self.memmapfile):
                # load memmap file
                self.sequence_data = np.memmap(self.memmapfile, dtype="float32", mode="r")
                self.sequence_data = np.reshape(
                    self.sequence_data, (4, int(self.sequence_data.shape[0] / 4))
                )
            else:
                # convert all sequences into encoding
                self.sequence_data = np.zeros((4, self.lens.sum()), dtype=np.float32)
                print('Iterating through chromosomes')
                for c in tqdm(self.chrs):
                    sequence = self.genome[c][:].seq
                    encoding = self.sequence_to_encoding(sequence)
                    self.sequence_data[
                    :, self.inds[c]: self.inds[c] + self.len_chrs[c]
                    ] = encoding.T
                if self.memmapfile is not None:
                    # create memmap file
                    mmap = np.memmap(
                        self.memmapfile, dtype="float32", mode="w+", shape=self.sequence_data.shape
                    )
                    mmap[:] = self.sequence_data
                    self.sequence_data = np.memmap(
                        self.memmapfile, dtype="float32", mode="r", shape=self.sequence_data.shape
                    )
            print("Finished Initializing Genome")
            self.initialized = True

    def init(func):
        # delay initlization to allow  multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            self._unpicklable_init()
            return func(self, *args, **kwargs)

        return dfunc

    @init
    def get_encoding_from_coords(self, chrom, start, end, strand="+", pad=False):
        """
        Gets the one-hot encoding of the genomic sequence at the
        queried coordinates.

        Parameters
        ----------
        chrom : str
            The name of the chromosome or region, e.g. "chr1".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '+'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.


        Returns
        -------
        numpy.ndarray, dtype=numpy.float32
            The :math:`L \\times 4` encoding of the sequence, where
            :math:`L = end - start`.

        Raises
        ------
        AssertionError
            If it cannot retrieve encoding that matches the length `L = end - start`
            such as when end > chromosome length and pad=False
        """
        if pad:
            # padding with 0.25 if coordinates extend beyond chr boundary
            if end > self.len_chrs[chrom]:
                pad_right = end - self.len_chrs[chrom]
                qend = self.len_chrs[chrom]
            else:
                qend = end
                pad_right = 0

            if start < 0:
                pad_left = 0 - start
                qstart = 0
            else:
                pad_left = 0
                qstart = start

            encoding = np.hstack(
                [
                    np.ones((4, pad_left)) * 0.25,
                    self.sequence_data[:, self.inds[chrom] + qstart: self.inds[chrom] + qend],
                    np.ones((4, pad_right)) * 0.25,
                ]
            )
        else:
            assert end <= self.len_chrs[chrom] and start >= 0
            encoding = self.sequence_data[:, self.inds[chrom] + start: self.inds[chrom] + end]

        if strand == "-":
            encoding = encoding[::-1, ::-1]
        assert encoding.shape[1] == end - start
        return encoding.T

    @init
    def get_str_seq(self, chrom, start, end, strand='+'):
        """
        extract the DNA string sequence
        """
        seq = str(self.genome[chrom][start:end])
        if strand == '-':
            seq = string_reverse_complement(seq)
        return seq

    @init
    def get_encoding_from_coords_check_unk(self, chrom, start, end, strand="+", pad=False):
        """Gets the one-hot encoding of the genomic sequence at the
        queried coordinates and check whether the sequence contains
        unknown base(s).

        Parameters
        ----------
        chrom : str
            The name of the chromosome or region, e.g. "chr1".
        start : int
            The 0-based start coordinate of the first position in the
            sequence.
        end : int
            One past the 0-based last position in the sequence.
        strand : {'+', '-', '.'}, optional
            Default is '+'. The strand the sequence is located on. '.' is
            treated as '+'.
        pad : bool, optional
            Default is `False`. Pad the output sequence with 'N' if `start`
            and/or `end` are out of bounds to return a sequence of length
            `end - start`.

        Returns
        -------
        tuple(numpy.ndarray, bool)
            * `tuple[0]` is the :math:`L \\times 4` encoding of the sequence, where
            :math:`L = end - start`.
            `L` = 0 for the NumPy array returned.
            * `tuple[1]` is the boolean value that indicates whether the
            sequence contains any unknown base(s) specified in self.UNK_BASE


        Raises
        ------
        AssertionError
            If it cannot retrieve encoding that matches the length `L = end - start`
            such as when end > chromosome length and pad=False
        """
        encoding = self.get_encoding_from_coords(chrom, start, end, strand=strand, pad=strand)
        return encoding, np.any(encoding[0, :] == 0.25)


def _adaptive_coarsegrain(ar, countar, max_levels=12):
    """
    Wrapper for cooltools adaptive coarse-graining to add support
    for non-square input for interchromosomal predictions.
    """
    assert np.all(ar.shape == countar.shape)
    if ar.shape[0] < 9 and ar.shape[1] < 9:
        ar_padded = np.empty((9, 9))
        ar_padded.fill(np.nan)
        ar_padded[: ar.shape[0], : ar.shape[1]] = ar

        countar_padded = np.empty((9, 9))
        countar_padded.fill(np.nan)
        countar_padded[: countar.shape[0], : countar.shape[1]] = countar
        return adaptive_coarsegrain(ar_padded, countar_padded, max_levels=max_levels)[
               : ar.shape[0], : ar.shape[1]
               ]

    if ar.shape[0] == ar.shape[1]:
        return adaptive_coarsegrain(ar, countar, max_levels=max_levels)
    elif ar.shape[0] > ar.shape[1]:
        padding = np.empty((ar.shape[0], ar.shape[0] - ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain(
            np.hstack([ar, padding]), np.hstack([countar, padding]), max_levels=max_levels
        )[:, : ar.shape[1]]
    elif ar.shape[0] < ar.shape[1]:
        padding = np.empty((ar.shape[1] - ar.shape[0], ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain(
            np.vstack([ar, padding]), np.vstack([countar, padding]), max_levels=max_levels
        )[: ar.shape[0], :]


class Genomic2DFeatures(Target):
    """
    Stores one or multple datasets of Hi-C style 2D data in cooler format.

    Parameters
    ----------
    input_paths : list(str) or str
        List of paths to the Cooler datasets or a path to a single
        Cooler dataset. For mcool files,
        the path should include the resolution. Please refer to
        cooler.Cooler documentation for support of mcool files.
    features : list(str) or str
        The list of dataset names that should match the `input_path`.
    shape : tuple(int, int)
        The shape of the output array (# of bins by # of bins).
    cg : bool, optional
        If `yes`, adpative coarse-graining is applied to the output.

    Attributes
    ----------
    data : list(cooler.Cooler)
        The list of Cooler objects for the cooler files.
    n_features : int
        The number of cooler files.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    shape : tuple(int, int)
        The shape of the output array (# of bins by # of bins).
    cg : bool
        Whether adpative coarse-graining is applied to the output.

    """

    def __init__(self, input_paths, features, shape, normmat=None, cg=False, shrinkage=False):
        """
        Constructs a new `Genomic2DFeatures` object.
        """
        if isinstance(input_paths, str) and isinstance(features, str):
            input_paths = [input_paths]
            features = [features]

        self.input_paths = input_paths
        self._initialized = False

        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])
        self.shape = shape
        self.cg = cg
        self.shrinkage = shrinkage
        self.normmat = normmat

    def get_feature_data(self, chrom, start, end, chrom2=None, start2=None, end2=None):
        if not self._initialized:
            self.data = [cooler.Cooler(path) for path in self.input_paths]
            self._initialized = True
        self.chrom = chrom
        self.start = start
        self.end = end
        if chrom2 is not None and start2 is not None and end2 is not None:
            query = ((chrom, start, end), (chrom2, start2, end2))
        else:
            query = ((chrom, start, end),)
        if self.cg:
            out = []
            for c in self.data:
                rawmat = c.matrix(balance=False).fetch(*query)
                balmat = c.matrix(balance=True).fetch(*query)
                cgmat = _adaptive_coarsegrain(
                    balmat, rawmat
                ).astype(np.float32)

                if self.shrinkage:
                    if len(query) == 2:
                        w1 = np.array(c.bins().fetch(query[0])['weight'].values)
                        w2 = np.array(c.bins().fetch(query[1])['weight'].values)
                    else:
                        w1 = w2 = np.array(c.bins().fetch(query[0])['weight'].values)
                    shmat = (rawmat * rawmat + cgmat / w1[:, np.newaxis] / w2[np.newaxis, :] * float(self.shrinkage)) / \
                            (rawmat + float(self.shrinkage)) * \
                            w1[:, np.newaxis] * w2[np.newaxis, :]

                    out.append(shmat)
                else:
                    out.append(cgmat)
        else:
            if self.shrinkage:
                assert self.normmat is not None
                out = []
                for c in self.data:
                    rawmat = c.matrix(balance=False).fetch(*query)
                    balmat = c.matrix(balance=True).fetch(*query).astype(np.float32)
                    if len(query) == 2:
                        w1 = np.array(c.bins().fetch(query[0])['weight'].values)
                        w2 = np.array(c.bins().fetch(query[1])['weight'].values)
                    else:
                        w1 = w2 = np.array(c.bins().fetch(query[0])['weight'].values)
                    shmat = (rawmat * rawmat + self.normmat / w1[:, np.newaxis] / w2[np.newaxis, :] * float(
                        self.shrinkage)) / \
                            (rawmat + float(self.shrinkage)) * \
                            w1[:, np.newaxis] * w2[np.newaxis, :]

                    out.append(shmat)
            else:
                out = [c.matrix(balance=True).fetch(*query).astype(np.float32) for c in self.data]
        if len(out) == 1:
            out = out[0]
        else:
            out = np.concatenate([o[None, :, :] for o in out], axis=0)
        return out


class MultibinGenomicFeatures(Target):
    """
    Multibin version of selene.targets.GenomicFeatures
    Stores the dataset specifying features for genomic regions.
    Accepts a `*.bed` file with the following columns,
    in order:
    ::
        [chrom, start, end, strand, feature]
    `start` and `end` is 0-based as in bed file format.

    Note that unlike selene_sdk.targets.GenomicFeatures which queries
    the tabix data file out-of-core, MultibinGenomicFeatures requires
    more memory as it loads the entire bed file in memory as a pyranges
    table for higher query speed.

    Parameters
    ----------
    input_path : str
        Path to the bed file.
    features : list(str)
        The non-redundant list of genomic features names. The output array
        will have the same feature order as specified in this list.
    bin_size : int
        The length of the bin(s) in which we check for features
    step_size : int
        The interval between two adjacent bins.
    shape : tuple(int, int)
        The shape of the output array (n_features by n_bins).
    mode : str, optional
        For `mode=='any'`, any overlap will get 1, and no overlap will get 0.
        For `mode=='center', only overlap with the center basepair of each bin
        will get 1, otherwise 0.
        For `mode=='proportion'`, the proportion of overlap will be returned.

    Attributes
    ----------
    data : pyranges.PyRanges
        The data stored in PyRanges object.
    n_features : int
        The number of distinct features.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    index_feature_dict : dict
        A dictionary mapping indices (`int`) to feature names (`str`),
        where the index is the position of the feature in the input
        features.
    bin_size : int
        The length of the bin(s) in which we check for features
    step_size : int
        The interval between two adjacent bins.
    shape : tuple(int, int)
        The shape of the output array (n_features by n_bins).
    mode : str
        - For `mode=='any'`, any overlap will get assigned 1, and no overlap will
        get assigned 0.
        - For `mode=='center', only overlap with the center basepair of each bin
        will get assigned 1, otherwise assigned 0.
        - For `mode=='proportion'`, the proportion of overlap will be assigned.

    """

    def __init__(self, input_path, features, bin_size, step_size, shape, mode="center"):
        """
        Constructs a new `MultibinGenomicFeatures` object.
        """
        self.input_path = input_path
        self.n_features = len(features)

        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])

        self.index_feature_dict = dict(list(enumerate(features)))

        self.bin_size = bin_size
        self.step_size = step_size

        self.initialized = False
        self.shape = shape
        self.mode = mode

    def init(func):
        # delay initlization to allow multiprocessing (not necessary here
        # but kept for consistency)
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                self.data = pyranges.read_bed(self.input_path)
                self.initialized = True
            return func(self, *args, **kwargs)

        return dfunc

    @init
    def get_feature_data(self, chrom, start, end):
        """
        For a genomic region specified, return a `number of features`
        by `number of bins` array for overlap of each genomic bin and
        each feature. How the overlap is quantified depends on the
        `mode` attribute specified during initialization.

        For `mode=='any'`, any overlap will get assigned 1, and no overlap will
        get assigned 0.
        For `mode=='center', only overlap with the center basepair of each bin
        will get assigned 1, otherwise assigned 0.
        For `mode=='proportion'`, the proportion of overlap will be assigned.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. '1', '2', ..., 'X', 'Y').
        start : int
            The 0-based first position in the region.
        end : int
            One past the 0-based last position in the region.

        Returns
        -------
        numpy.ndarray
            :math:`L \\times N` array, where :math:`L = ``number of bins`
            and :math:`N =` `self.n_features`.

        """

        n_bins = int((end - start - self.bin_size) / self.step_size) + 1
        targets = np.zeros((self.n_features, n_bins), dtype=np.float32)

        if self.mode == "center":
            b = pyranges.PyRanges(
                pd.DataFrame(
                    dict(
                        Chromosome=chrom,
                        Start=start
                              + np.linspace(0, n_bins * self.bin_size, n_bins + 1)[:-1]
                              + self.bin_size / 2,
                        End=start
                            + np.linspace(0, n_bins * self.bin_size, n_bins + 1)[:-1]
                            + self.bin_size / 2
                            + 1,
                        Index=np.arange(n_bins),
                    )
                )
            )
        else:
            b = pyranges.PyRanges(
                pd.DataFrame(
                    dict(
                        Chromosome=chrom,
                        Start=start + np.linspace(0, n_bins * self.bin_size, n_bins + 1)[:-1],
                        End=start
                            + np.linspace(0, n_bins * self.bin_size, n_bins + 1)[:-1]
                            + self.bin_size,
                        Index=np.arange(n_bins),
                    )
                )
            )

        rows = self.data.join(b)
        if len(rows) > 0:
            rows_featurename = np.array(rows.Name)
            rows_index = np.array(rows.Index)
            if self.mode == "proportion":
                rows_start = np.array(rows.Start)
                rows_end = np.array(rows.End)
                for i in range(len(rows)):
                    targets[self.feature_index_dict[rows_featurename[i]], rows_index[i]] += (
                                                                                                    rows_end[i] -
                                                                                                    rows_start[i]
                                                                                            ) / self.bin_size
            else:
                for i in range(len(rows)):
                    targets[self.feature_index_dict[rows_featurename[i]], rows_index[i]] = 1

        return targets.astype(np.float32)


class RandomPositionsSamplerHiC(OnlineSampler):
    """This sampler randomly selects a region in the genome and retrieves
    sequence and relevant Hi-C and optionally multibin genomic
    data from that region. This implementation is modified based on
    selene_sdk.samplers.RandomPositionSampler.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        A genome to retrieve sequence from.
    target : Genomic2DFeatures
        Genomic2DFeatures object that loads the cooler files.
    features : list(str)
        List of names that correspond to the cooler files.
    target_1d : MultibinGenomicFeatures or None, optional
        MultibinGenomicFeatures object that loads 1D genomic feature data.
    background_cis_file : str or None, optional
        Path to the numpy file that stores the distance-based
        expected background balanced scores for cis-interactions. If
        specified with background_trans_file, the sampler will
        return corresponding background array that matches with
        the 2D feature retrieved.
    background_trans_file : str or None, optional
        Path to the numpy file that stores the expected background
        balanced scores for trans-interactions. See doc for
        `background_cis_file` for more detail.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str), optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['chrX', 'chrY']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str), optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000000. Model is trained on sequences of size
        `sequence_length` where genomic features are retreived
        for the same regions as the sequences.
    max_seg_length : int or None, optional
        Default is None. If specified and cross_chromosome is True,
        bound the maximum length of each sequence segment.
    length_schedule : list(float, list(int, int)) or None, optional
        Default is None. If specified and cross_chromosome is True,
        decide the sequence segment length to sample according to the
        length schedule (before trimming to fit in the sequence length).
        The length schedule is in the format of `[p, [min_len, max_len]]`,
        which means, with probability `p`, decide the length by randomly
        sampling an integer between `min_len` and `max_len`, and retrieve
        the maximal remaining length as default with probability `1-p`.
    position_resolution : int, optional
        Default is 1. Preprocess the sampled start position by
        `start = start - start % position_resolution`. Useful for binned
        data.
    random_shift : int, optional
        Default is 0. Shift the coordinates to retrieve
        sequence by a random integer in the range of [-random_shift, random_shift).
    random_strand : bool, optional
        Default is True. If True, randomly select the strand of the
        sequence, otherwise alway use the '+' strand.
    cross_chromosome : bool, optional
        Default is True. If True, allows sampling multiple segments of
        sequences and the corresponding features. The default is sampling
        the maximum length allowed by sequence_length, thus multiple segments
        will only be sampled if `sequence_length` is larger than the minimum
        chromosome length or when max_seg_length and length_schedule is specified
        to limit the sequence segment length.
    permute_segments : bool, optional
        Default is False. If True, permute the order of segments when
        multiple segments are sampled.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.


    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        A genome to retrieve sequence from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    target_1d : MultibinGenomicFeatures or None, optional
        MultibinGenomicFeatures object that loads 1D genomic feature data.
    background_cis : numpy.ndarray
        One-dimensional numpy.ndarray that stores the distance-based
        expected background balanced scores for cis-interactions.
    background_trans : float
        The expected background balanced score for trans-interactions.
    bg : bool
        Whether the sample will retrieve background arrays.
    validation_holdout : list(str)
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str)
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        Model is trained on sequences of size
        `sequence_length` where genomic features are retreived
        for the same regions as the sequences.
    max_seg_length : int or None
        Default is None. If specified and cross_chromosome is True,
        bound the maximum length of each sequence segment.
    length_schedule : list(float, list(int, int)) or None
        Default is None. If specified and cross_chromosome is True,
        decide the sequence segment length to sample according to the
        length schedule (before trimming to fit in the sequence length).
        The length schedule is in the format of `[p, [min_len, max_len]]`,
        which means, with probability `p`, decide the length by randomly
        sampling an integer between `min_len` and `max_len`, and retrieve
        the maximal remaining length as default with probability `1-p`.
    position_resolution : int
        Default is 1. Preprocess the sampled start position by
        `start = start - start % position_resolution`. Useful for binned
        data.
    random_shift : int
        Default is 0. Shift the coordinates to retrieve
        sequence by a random integer in the range of [-random_shift, random_shift).
    random_strand : bool
        Default is True. If True, randomly select the strand of the
        sequence, otherwise alway use the '+' strand.
    cross_chromosome : bool
        Default is True. If True, allows sampling multiple segments of
        sequences and the corresponding features. The default is sampling
        the maximum length allowed by sequence_length, thus multiple segments
        will only be sampled if `sequence_length` is larger than the minimum
        chromosome length or when max_seg_length and length_schedule is specified
        to limit the sequence segment length.
    permute_segments : bool
        Default is False. If True, permute the order of segments when
        multiple segments are sampled.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """

    def __init__(
            self,
            reference_sequence,
            target,
            features,
            target_1d=None,
            background_cis_file=None,
            background_trans_file=None,
            seed=436,
            validation_holdout=["chr6", "chr7"],
            test_holdout=["chr8", "chr9"],
            sequence_length=1000000,
            max_seg_length=None,
            length_schedule=None,
            position_resolution=1,
            random_shift=0,
            random_strand=True,
            cross_chromosome=True,
            permute_segments=False,
            mode="train",
    ):
        super(RandomPositionsSamplerHiC, self).__init__(
            reference_sequence,
            target,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=sequence_length,
            mode=mode,
        )

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []
        self.initialized = False
        self.position_resolution = position_resolution
        self.random_shift = random_shift
        self.random_strand = random_strand
        if background_cis_file is not None and background_trans_file is not None:
            self.background_cis = np.hstack(
                [np.exp(np.load(background_cis_file)), np.repeat(np.nan, 2000)]
            )
            self.background_trans = np.exp(np.load(background_trans_file))
            self.bg = True
        else:
            self.bg = False
        self.max_seg_length = max_seg_length
        self.length_schedule = length_schedule
        self.target_1d = target_1d
        self.cross_chromosome = cross_chromosome
        self.permute_segments = permute_segments
        if len(validation_holdout) == 0:
            self.modes = ["train"]

    def init(func):
        # delay initlization to allow  multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                self._partition_genome_by_chromosome()
                for mode in self.modes:
                    self._update_randcache(mode=mode)
                self.initialized = True
            return func(self, *args, **kwargs)

        return dfunc

    def _partition_genome_by_chromosome(self):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(index)
            else:
                self._sample_from_mode["train"].indices.append(index)

            self.sample_from_intervals.append((chrom, 0, len_chrom))
            self.interval_lengths.append(len_chrom)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = self._sample_from_mode[mode]._replace(
                indices=indices, weights=weights
            )

    def _retrieve_multi(self, chroms, starts, ends, strands=None):
        retrieved_seqs = []
        if self.target_1d:
            retrieved_1ds = []
        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            if strands is not None:
                strand = strands[i]
            else:
                strand = "+"
            if self.random_shift > 0:
                r = np.random.randint(-self.random_shift, self.random_shift)
            else:
                r = 0

            retrieved_seq = self.reference_sequence.get_encoding_from_coords(
                chrom, start + r, end + r, strand, pad=True
            )
            retrieved_seqs.append(retrieved_seq)

            if self.target_1d:
                retrieved_1d = self.target_1d.get_feature_data(chrom, start, end)
                if strand == "-":
                    retrieved_1d = retrieved_1d[:, ::-1]
                retrieved_1ds.append(retrieved_1d)

        retrieved_targets = []
        if self.bg:
            background_targets = []
        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            if strands is not None:
                strand = strands[i]
            else:
                strand = "+"
            retrieved_targets_row = []
            if self.bg:
                background_targets_row = []
            for j, (chrom2, start2, end2) in enumerate(zip(chroms, starts, ends)):
                if strands is not None:
                    strand2 = strands[j]
                else:
                    strand2 = "+"
                retrieved_target = self.target.get_feature_data(
                    chrom, start, end, chrom2=chrom2, start2=start2, end2=end2
                )
                if self.bg:
                    if chrom2 != chrom:
                        background_target = np.full_like(retrieved_target, self.background_trans)
                    else:
                        binsize = (end - start) / retrieved_target.shape[-2]
                        acoor = np.linspace(start, end, retrieved_target.shape[-2] + 1)[:-1]
                        bcoor = np.linspace(start2, end2, retrieved_target.shape[-1] + 1)[:-1]
                        background_target = self.background_cis[
                            (np.abs(acoor[:, None] - bcoor[None, :]) / binsize).astype(int)
                        ]

                if strand == "-":
                    retrieved_target = np.flip(retrieved_target, -2)
                    if self.bg:
                        background_target = np.flip(background_target, -2)
                if strand2 == "-":
                    retrieved_target = np.flip(retrieved_target, -1)
                    if self.bg:
                        background_target = np.flip(background_target, -1)
                retrieved_targets_row.append(retrieved_target)
                if self.bg:
                    background_targets_row.append(background_target)
            retrieved_targets.append(retrieved_targets_row)
            if self.bg:
                background_targets.append(background_targets_row)

        if self.bg:
            if self.target_1d:
                return (retrieved_seqs, retrieved_targets, background_targets, retrieved_1ds)
            else:
                return (retrieved_seqs, retrieved_targets, background_targets)
        else:
            if self.target_1d:
                return (retrieved_seqs, retrieved_targets, retrieved_1ds)
            else:
                return (retrieved_seqs, retrieved_targets)

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights,
        )
        self._randcache[mode]["sample_next"] = 0

    @init
    def sample(self, batch_size=1, mode=None, coordinate_only=False):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
        coordinate_only : bool, optional
            Default is False. If True, only return the coordinates.

        Returns
        -------
        sequences, targets, ...: tuple(numpy.ndarray, numpy.ndarray, ...)
            A tuple containing the numeric representation of the
            sequence examples, their corresponding 2D targets, and optionally 1D targets
            (if target_1d were specified) and background
            matrices (if background_cis_file and background_trans_file were
            specified). The shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` depends on target.shape. For example it
            will be :math:`B \\times M \\times M`,
            when :math:`M \\times M` is target.shape. The shape of 1D targets
            is :math:`B \\times K \\times F`, where :math:`K = ``number of bins`
            and :math:`F =` `self.n_features`. The shape of background matrices
            are the same as `targets`.

        """
        mode = mode if mode else self.mode
        if not coordinate_only:
            sequences = np.zeros((batch_size, self.sequence_length, 4))
            targets = np.zeros((batch_size, *self.target.shape))
            if self.bg:
                normmats = np.zeros((batch_size, *self.target.shape))
            if self.target_1d:
                target_1ds = np.zeros((batch_size, *self.target_1d.shape))

        n_samples_drawn = 0
        allcoords = []
        while n_samples_drawn < batch_size:
            current_length = 0
            chroms = []
            starts = []
            ends = []
            strands = []
            while current_length < self.sequence_length:
                if len(chroms) == 0 or self.cross_chromosome:
                    sample_index = self._randcache[mode]["sample_next"]
                    if sample_index == len(self._randcache[mode]["cache_indices"]):
                        self._update_randcache()
                        sample_index = 0

                    rand_interval_index = self._randcache[mode]["cache_indices"][sample_index]
                    self._randcache[mode]["sample_next"] += 1

                    chrom, cstart, cend = self.sample_from_intervals[rand_interval_index]

                next_length = self.sequence_length - current_length
                if self.length_schedule is not None and self.cross_chromosome:
                    if np.random.rand() < self.length_schedule[0]:
                        next_length = np.fmin(
                            next_length,
                            np.random.randint(
                                self.length_schedule[1][0], self.length_schedule[1][1]
                            ),
                        )

                if self.max_seg_length is not None and self.cross_chromosome:
                    next_length = np.fmin(next_length, self.max_seg_length)

                start_position = np.random.randint(cstart, np.fmax(cstart + 1, cend - next_length))
                start_position -= start_position % self.position_resolution

                if start_position + next_length > cend:
                    if (
                            self.cross_chromosome
                            or (self.length_schedule is not None)
                            or (self.max_seg_length is not None)
                    ):
                        end_position = cend
                    else:
                        continue
                else:
                    end_position = start_position + next_length

                end_position -= end_position % self.position_resolution
                if end_position == start_position:
                    continue
                if not self.reference_sequence.coords_in_bounds(
                        chrom, start_position, end_position
                ):
                    continue

                current_length += end_position - start_position
                chroms.append(chrom)
                starts.append(start_position)
                ends.append(end_position)
                if self.random_strand:
                    strand = self.STRAND_SIDES[np.random.randint(0, 2)]
                else:
                    strand = "+"
                strands.append(strand)

            if self.permute_segments:
                perm = np.random.permutation(np.arange(len(chroms)))
                chroms = [chroms[i] for i in perm]
                starts = [starts[i] for i in perm]
                ends = [ends[i] for i in perm]
                strands = [strands[i] for i in perm]

            allcoords.append([chroms, starts, ends, strands])
            n_samples_drawn += 1

        for i, (chroms, starts, ends, strands) in enumerate(allcoords):
            if not coordinate_only:
                retrieve_output = self._retrieve_multi(chroms, starts, ends, strands)
                if self.bg:
                    if self.target_1d:
                        seq, seq_targets, seq_background, seq_target_1ds = retrieve_output
                    else:
                        seq, seq_targets, seq_background = retrieve_output
                else:
                    if self.target_1d:
                        seq, seq_targets, seq_target_1ds = retrieve_output
                    else:
                        seq, seq_targets = retrieve_output

                if not isinstance(seq, list):
                    sequences[i, :, :] = seq
                    targets[i, :] = seq_targets
                    if self.bg:
                        normmats[i, :] = seq_background
                    if self.target_1d:
                        target_1ds[i, :] = seq_target_1ds
                else:
                    offset = 0
                    for s in seq:
                        sequences[i, offset: offset + s.shape[0], :] = s
                        offset = offset + s.shape[0]

                    if self.target_1d:
                        offset = 0
                        for t in seq_target_1ds:
                            target_1ds[i, :, offset: offset + t.shape[1]] = t
                            offset = offset + t.shape[1]

                    offsetx = 0
                    for row in seq_targets:
                        offsety = 0
                        for t in row:
                            if targets.ndim == 3:
                                targets[
                                i,
                                offsetx: offsetx + t.shape[0],
                                offsety: offsety + t.shape[1],
                                ] = t
                            else:
                                targets[
                                i,
                                :,
                                offsetx: offsetx + t.shape[-2],
                                offsety: offsety + t.shape[-1],
                                ] = t
                            offsety = offsety + t.shape[-1]
                        offsetx = offsetx + t.shape[-2]
                    assert offsetx == targets.shape[-2]
                    assert offsety == targets.shape[-1]

                    if self.bg:
                        offsetx = 0
                        for row in seq_background:
                            offsety = 0
                            for t in row:
                                if normmats.ndim == 3:
                                    normmats[
                                    i,
                                    offsetx: offsetx + t.shape[-2],
                                    offsety: offsety + t.shape[-1],
                                    ] = t
                                else:
                                    normmats[
                                    i,
                                    :,
                                    offsetx: offsetx + t.shape[-2],
                                    offsety: offsety + t.shape[-1],
                                    ] = t
                                offsety = offsety + t.shape[-1]
                            offsetx = offsetx + t.shape[-2]
                        assert offsetx == normmats.shape[-2]
                        assert offsety == normmats.shape[-1]

        if coordinate_only:
            return allcoords
        else:
            if self.bg:
                if self.target_1d:
                    return (sequences, targets, normmats, target_1ds)
                else:
                    return (sequences, targets, normmats)
            else:
                if self.target_1d:
                    return (sequences, targets, target_1ds)
                else:
                    return (sequences, targets)
