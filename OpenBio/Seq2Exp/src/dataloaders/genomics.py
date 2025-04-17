"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import copy
from typing import Any, List, Union

import os
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader
import pandas as pd
from torch.utils.data import Subset

from caduceus.tokenization_caduceus import CaduceusTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.datasets.dim2_datasets import Dim2Dataset, Dim2GraphDataset
from src.dataloaders.datasets.promo_enhan_inter import PromoterEnhancerDataset, ExperInteractDataset
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler

logger = src.utils.train.get_logger(__name__)


def cage_pred_collate_fn(batch):
    graph_data, signals, token_dna = zip(*batch)
    batch_graph = Batch.from_data_list(graph_data)
    signals = torch.stack(signals, dim=0)
    token_dna = torch.stack(token_dna, dim=0)

    return batch_graph, signals, token_dna


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
        self.collate_fn = None
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
                                   shuffle=shuffle, sampler=sampler, collate_fn=self.collate_fn,
                                   worker_init_fn=self.dataset_worker_init,
                                   **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,
                                 collate_fn=self.collate_fn,
                                 worker_init_fn=self.dataset_worker_init,
                                 **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval,
                                 collate_fn=self.collate_fn,
                                 worker_init_fn=self.dataset_worker_init,
                                 **kwargs)

    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, collate_fn=None, worker_init_fn=None, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
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


class CAGEPredictors(HG38):
    _name_ = "cage_pred"

    def __init__(self, exper_name_cage, exper_name_dnase, exper_name_h3k27ac, exper_name_h3k4me3,
                 cell_type, organism='human', genome='hg19', assay_type='HiC', qval=0.1,
                 resolution=5000, consider_seq_len=600_0000, pred_seq_len=200_0000,
                 valid_chr='1,11', test_chr='2,12', disable_graph=False,
                 tokenizer_name=None,
                 batch_size=32, batch_size_eval=None, shuffle=False, num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.exper_name_cage = exper_name_cage
        self.exper_name_dnase = exper_name_dnase
        self.exper_name_h3k27ac = exper_name_h3k27ac
        self.exper_name_h3k4me3 = exper_name_h3k4me3
        self.cell_type = cell_type
        self.organism = organism
        self.genome = genome
        self.assay_type = assay_type
        self.qval = qval
        self.resolution = resolution
        self.consider_seq_len = consider_seq_len
        self.pred_seq_len = pred_seq_len
        self.valid_chr = valid_chr
        self.test_chr = test_chr
        self.disable_graph = disable_graph

        self.collate_fn = cage_pred_collate_fn
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
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
        if self.tokenizer_name == 'char':
            logger.info("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.resolution,
                add_special_tokens=False,
            )
            self.vocab_size = len(self.tokenizer.get_vocab())
        elif self.tokenizer_name == 'one_hot':
            logger.info("**Using One-Hot embeddings**")
            self.tokenizer = None
            self.vocab_size = 0
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        # delete old datasets to free memory
        if hasattr(self, "dataset_train"):
            if self.tokenizer_name == 'one_hot':
                del self.dataset_train.chr_embed_dict
            elif self.tokenizer_name == 'char':
                self.dataset_train.fasta.close()
                del self.dataset_train.fasta

        if hasattr(self, "dataset_val"):
            if self.tokenizer_name == 'one_hot':
                del self.dataset_val.chr_embed_dict
            elif self.tokenizer_name == 'char':
                self.dataset_val.fasta.close()
                del self.dataset_val.fasta

        if hasattr(self, "dataset_test"):
            if self.tokenizer_name == 'one_hot':
                del self.dataset_test.chr_embed_dict
            elif self.tokenizer_name == 'char':
                self.dataset_test.fasta.close()
                del self.dataset_test.fasta

        # TODO: set exper_str
        self.dataset_train, self.dataset_val, self.dataset_test = [
            Dim2Dataset(
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                cell_type=self.cell_type,
                exper_str=self.exper_name,
                genome=self.genome,
                organism=self.organism,
                consider_region=self.consider_seq_len,
                pred_region=self.pred_seq_len,
                resolution=self.resolution,
                assay_type=self.assay_type,
                qval=self.qval,
                valid_chr=self.valid_chr,
                test_chr=self.test_chr,
                data_split=split,
                disable_graph=self.disable_graph,
            )
            for split in ["train", "valid", "test"]
        ]


class CAGEGraphPredict(CAGEPredictors):
    _name_ = "cage_pred_graph"

    def __init__(self, exper_name_cage, exper_name_dnase, exper_name_h3k27ac, exper_name_h3k4me3,
                 cell_type, organism='human', genome='hg19', assay_type='HiC', qval=0.1,
                 resolution=5000, consider_seq_len=600_0000, pred_seq_len=200_0000,
                 valid_chr='1,11', test_chr='2,12',
                 disable_graph=False, disable_hic=False, disable_adj=False, keep_close=None,
                 node_sample="3,4,5", tokenizer_name=None,
                 batch_size=32, batch_size_eval=None, shuffle=False, num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.exper_name_cage = exper_name_cage
        self.exper_name_dnase = exper_name_dnase
        self.exper_name_h3k27ac = exper_name_h3k27ac
        self.exper_name_h3k4me3 = exper_name_h3k4me3
        self.cell_type = cell_type
        self.organism = organism
        self.genome = genome
        self.assay_type = assay_type
        self.qval = qval
        self.resolution = resolution
        self.consider_seq_len = consider_seq_len
        self.pred_seq_len = pred_seq_len
        self.valid_chr = valid_chr
        self.test_chr = test_chr
        self.disable_graph = disable_graph
        self.disable_hic = disable_hic
        self.disable_adj = disable_adj
        self.keep_close = keep_close

        self.graph_neighbor = [int(neighbor) for neighbor in node_sample.split(',')]
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
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

    def init_datasets(self):
        # delete old datasets to free memory
        if hasattr(self, "graph_datasets"):
            if self.tokenizer_name == 'one_hot':
                del self.graph_datasets.chr_embed_dict
            elif self.tokenizer_name == 'char':
                self.graph_datasets.fasta.close()
                del self.graph_datasets.fasta

        self.graph_datasets = Dim2GraphDataset(
            self.tokenizer,
            self.cell_type,
            exper_name_cage=self.exper_name_cage,
            exper_name_dnase=self.exper_name_dnase,
            exper_name_h3k27ac=self.exper_name_h3k27ac,
            exper_name_h3k4me3=self.exper_name_h3k4me3,
            genome=self.genome,
            organism=self.organism,
            resolution=self.resolution,
            assay_type=self.assay_type,
            qval=self.qval,
            valid_chr=self.valid_chr,
            test_chr=self.test_chr,
            tokenizer_name=self.tokenizer_name,
            disable_graph=self.disable_graph,
            disable_hic=self.disable_hic,
            disable_adj=self.disable_adj,
            keep_close=self.keep_close,
        )

        self.collate_fn = None

    def train_dataloader(self, **kwargs: Any) -> NeighborLoader:
        """ The train dataloader """
        loader = self._data_loader(self.graph_datasets.merge_graph, batch_size=self.batch_size,
                                   shuffle=self.shuffle, sampler=None, collate_fn=self.collate_fn,
                                   mask=self.graph_datasets.merge_graph.train_mask,
                                   graph_neighbors=self.graph_neighbor,
                                   **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[NeighborLoader, List[NeighborLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.graph_datasets.merge_graph, batch_size=self.batch_size_eval,
                                 mask=self.graph_datasets.merge_graph.val_mask,
                                 graph_neighbors=self.graph_neighbor,
                                 collate_fn=self.collate_fn, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[NeighborLoader, List[NeighborLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.graph_datasets.merge_graph, batch_size=self.batch_size_eval,
                                 mask=self.graph_datasets.merge_graph.test_mask,
                                 graph_neighbors=self.graph_neighbor,
                                 collate_fn=self.collate_fn, **kwargs)

    @staticmethod
    def _data_loader(dataset: Data, batch_size: int, shuffle: bool = False, sampler=None,
                     collate_fn=None, mask=None, graph_neighbors=None, **kwargs) -> NeighborLoader:
        return NeighborLoader(
            dataset,
            num_neighbors=graph_neighbors,
            batch_size=batch_size,
            input_nodes=mask,
            collate_fn=collate_fn,
            shuffle=shuffle,
            **kwargs,
        )


class PromoEnhanInter(CAGEPredictors):
    _name_ = "promo_enhan_inter"

    def __init__(
            self,
            data_folder,
            dataset_name,
            expr_type='CAGE',
            usePromoterSignal=True,
            signal_type='H3K27ac',
            cell_type='K562',
            distance_threshold=None,
            hic_threshold=None,
            n_enhancers=50,
            n_extraFeat=0,
            seq_range=0,
            k_fold=0,
            tokenizer_name=None,
            pretrained_model=False,
            pretrained_model_name=None,
            blacklist_ver=None,
            resolution=200_000,
            zero_dist=False, zero_activity=False, zero_hic=False,
            omit_enhancers=False, only_seqs=False,
            batch_size=32, batch_size_eval=None, shuffle=False, num_workers=0,
            fault_tolerant=False, ddp=False,
            fast_forward_epochs=None, fast_forward_batches=None,
            *args, **kwargs,
    ):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.expr_type = expr_type
        self.usePromoterSignal = usePromoterSignal
        self.signal_type = signal_type
        self.cell_type = cell_type
        self.distance_threshold = distance_threshold
        self.hic_threshold = hic_threshold
        self.n_enhancers = n_enhancers
        self.n_extraFeat = n_extraFeat
        self.seq_range = seq_range
        self.k_fold = k_fold
        self.tokenizer_name = tokenizer_name
        self.pretrained_model = pretrained_model
        self.pretrained_model_name = pretrained_model_name
        self.blacklist_ver = blacklist_ver
        self.resolution = resolution

        self.zero_dist = zero_dist
        self.zero_activity = zero_activity
        self.zero_hic = zero_hic
        self.omit_enhancers = omit_enhancers
        self.only_seqs = only_seqs

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

    def init_datasets(self):
        self.split_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                 'leave_chrom_out_crossvalidation_split_18377genes.csv'))

        if hasattr(self, "all_ds"):
            self.all_ds.data_h5.close()
            del self.all_ds.data_h5
            self.all_ds.genome.close()
            del self.all_ds.genome

        if self.dataset_name == 'EPInformer':
            self.all_ds = PromoterEnhancerDataset(
                data_folder=self.data_folder,
                expr_type=self.expr_type,
                usePromoterSignal=self.usePromoterSignal,
                signal_type=self.signal_type,
                cell_type=self.cell_type,
                distance_threshold=self.distance_threshold,
                hic_threshold=self.hic_threshold,
                n_enhancers=self.n_enhancers,
                n_extraFeat=self.n_extraFeat,
                zero_dist=self.zero_dist,
                zero_activity=self.zero_activity,
                zero_hic=self.zero_hic,
                omit_enhancers=self.omit_enhancers,
                only_seqs=self.only_seqs,
            )
        elif self.dataset_name == 'gene_express':
            self.all_ds = ExperInteractDataset(
                data_folder=self.data_folder,
                split_df=self.split_df,
                expr_type=self.expr_type,
                cell_type=self.cell_type,
                distance_threshold=self.distance_threshold,
                hic_threshold=self.hic_threshold,
                n_enhancers=self.n_enhancers,
                seq_range=self.seq_range,
                zero_dist=self.zero_dist,
                zero_activity=self.zero_activity,
                zero_hic=self.zero_hic,
                omit_enhancers=self.omit_enhancers,
                only_seqs=self.only_seqs,
                tokenizer_name=self.tokenizer_name,
                pretrained_model=self.pretrained_model,
                pretrained_model_name=self.pretrained_model_name,
                blacklist_ver=self.blacklist_ver,
            )

        cur_fold = 'fold_' + str(self.k_fold)

        train_ensid = self.split_df[self.split_df[cur_fold] == 'train']['ENSID']
        valid_ensid = self.split_df[self.split_df[cur_fold] == 'valid']['ENSID']
        test_ensid = self.split_df[self.split_df[cur_fold] == 'test']['ENSID']

        ensid_list = [eid.decode() for eid in self.all_ds.data_h5['ensid'][:]]
        ensid_df = pd.DataFrame(ensid_list, columns=['ensid'])
        ensid_df['idx'] = np.arange(len(ensid_list))
        ensid_df = ensid_df.set_index('ensid')
        train_idx = ensid_df.loc[[ensid for ensid in train_ensid if ensid in ensid_df.index]]['idx'].to_numpy()
        valid_idx = ensid_df.loc[[ensid for ensid in valid_ensid if ensid in ensid_df.index]]['idx'].to_numpy()
        test_idx = ensid_df.loc[[ensid for ensid in test_ensid if ensid in ensid_df.index]]['idx'].to_numpy()

        self.dataset_train = Subset(self.all_ds, train_idx)
        self.dataset_val = Subset(self.all_ds, valid_idx)
        self.dataset_test = Subset(self.all_ds, test_idx)

        self.collate_fn = None
