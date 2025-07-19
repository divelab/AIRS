"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import copy
from typing import Any, List, Union

import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

from caduceus.tokenization_caduceus import CaduceusTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.datasets.generate_promoter import PromoterDataset
from src.dataloaders.datasets.generate_enhancers import EnhancerDataset
from src.dataloaders.datasets.generate_chipseq import ChIPSeqData


logger = src.utils.train.get_logger(__name__)


def dataset_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset.dataset.init_worker()
    elif isinstance(dataset, torch.utils.data.Dataset):
        dataset.init_worker()
    else:
        raise ValueError()


class HG38(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "hg38"  # this name is how the dataset config finds the right dataloader

    def __init__(self, bed_file, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2,
                 rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, shuffle=False,
                 num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 mlm=False, mlm_probability=0.15,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.dataset_worker_init = None

        # handle if file paths are None (default paths)
        if self.bed_file is None:
            self.bed_file = default_data_path / self._name_ / "human-sequences.bed"
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / "hg38.ml.fa"

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.mlm = mlm
        self.mlm_probability = mlm_probability

        # To be instantiated in `setup`
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            logger.info("**Using Char-level tokenizer**")
            # self.tokenizer = CharacterTokenizer(
            #     characters=["A", "C", "G", "T", "N"],
            #     model_max_length=self.max_length,
            #     add_special_tokens=False,
            # )
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.max_length,
                add_special_tokens=False
            )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, "dataset_train"):
            self.dataset_train.fasta.seqs.close()
            del self.dataset_train.fasta.seqs

        # delete old datasets to free memory
        if hasattr(self, "dataset_test"):
            self.dataset_test.fasta.seqs.close()
            del self.dataset_test.fasta.seqs

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(split=split,
                        bed_file=self.bed_file,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        mlm=self.mlm,
                        mlm_probability=self.mlm_probability, )
            for split, max_len in
            zip(["train", "valid", "test"], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        return

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(
                self.dataset_train,
                **distributed_sampler_kwargs
            ) if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    "epoch": self.fast_forward_epochs,
                    "counter": self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler,
                                   worker_init_fn=self.dataset_worker_init,
                                   **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,
                                 worker_init_fn=self.dataset_worker_init, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval,
                                 worker_init_fn=self.dataset_worker_init, **kwargs)

    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
            # TD [2022-08-07] ["epoch_loop.batch_progress"]["total"]["completed"] is 1 iteration
            # behind, so we're using the optimizer"s progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
                "completed"]
        # At this point the train loader hasn't been constructed yet


class Generators(HG38):
    _name_ = "generator"  # this name is how the dataset config finds the right dataloader

    def __init__(self, dataset_name, condition, tokenizer_name=None, max_length=1024, protein_len=0, rc_aug=False,
                 batch_size=32, batch_size_eval=None, shuffle=False, num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 n_tsses=100000, load_prob=False, strand='both', normalize_method='none',
                 reverse_aug=False, top_score=True, OOD=False,
                 *args, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.protein_len = protein_len
        self.n_tsses = n_tsses
        self.load_prob = load_prob
        self.strand = strand
        self.condition = condition
        self.normalize_method = normalize_method
        self.reverse_aug = reverse_aug
        self.top_score = top_score
        self.OOD = OOD
        self.rc_aug = rc_aug
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_worker_init = dataset_worker_init_fn
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
        # To be instantiated in `setup`
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            logger.info("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length,
                add_special_tokens=False,
                padding_side='right',
            )
            # self.tokenizer = CaduceusTokenizer(
            #     model_max_length=self.max_length,
            #     add_special_tokens=False
            # )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.vocab_size = len(self.tokenizer.get_vocab())
        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, "dataset_train") and hasattr(self.dataset_train, "genome"):
            self.dataset_train.genome.genome.close()
            del self.dataset_train.genome.genome

        if hasattr(self, "dataset_val") and hasattr(self.dataset_val, "genome"):
            self.dataset_val.genome.genome.close()
            del self.dataset_val.genome.genome

        if hasattr(self, "dataset_test") and hasattr(self.dataset_test, "genome"):
            self.dataset_test.genome.genome.close()
            del self.dataset_test.genome.genome

        if self.dataset_name == 'promoter':
            self.dataset_train, self.dataset_val, self.dataset_test = [
                PromoterDataset(
                    self.tokenizer,
                    tokenizer_name=self.tokenizer_name,
                    seqlength=self.max_length,
                    n_tsses=self.n_tsses,
                    rand_offset=offset,
                    split=split,
                    load_prob=self.load_prob,
                    strand=self.strand,
                    condition=self.condition,
                    normalize_method=self.normalize_method,
                    reverse_aug=self.reverse_aug,
                    rc_aug=self.rc_aug,
                ) for split, offset in zip(['train', 'valid', 'test'],
                                           [10, 0, 0])
            ]
            if self.normalize_method == 'normal':
                self.dataset_val.mean = self.dataset_train.mean
                self.dataset_val.std = self.dataset_train.std
                self.dataset_test.mean = self.dataset_train.mean
                self.dataset_test.std = self.dataset_train.std

        elif self.dataset_name == 'mel_enhancer' or self.dataset_name == 'flybrain_enhancer':
            self.dataset_train, self.dataset_val, self.dataset_test = [
                EnhancerDataset(
                    self.tokenizer,
                    tokenizer_name=self.tokenizer_name,
                    dataset_name=self.dataset_name,
                    split=split,
                    max_length=self.max_length,
                    load_prob=self.load_prob,
                ) for split in ['train', 'valid', 'test']
            ]

        elif self.dataset_name == 'chipseq':
            self.dataset_train, self.dataset_val, self.dataset_test = [
                ChIPSeqData(
                    self.tokenizer,
                    tokenizer_name=self.tokenizer_name,
                    split=split,
                    max_length=self.max_length,
                    protein_len=self.protein_len,
                    load_prob=self.load_prob,
                ) for split in ['train', 'valid', 'test']
            ]

