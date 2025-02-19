import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import json, os
import h5py
import scipy
from scipy.sparse import csr_matrix
import pysam
from torch.utils.data.dataloader import DataLoader, Sampler
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse, from_scipy_sparse_matrix
from torch_geometric.data import Data, Batch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import src.utils.train
from src.dataloaders.datasets.char_tokenizer import CharacterTokenizer
from src.dataloaders.utils.dna import dna_str_to_one_hot

data_root_path = ''
logger = src.utils.train.get_logger(__name__)


def save_one_hot_embed(genome='hg19', pred_region=200_0000, consider_region=600_0000):
    len_name = f'{pred_region}' if pred_region == consider_region else f'{pred_region}_{consider_region}'
    dict_file_path = f'{data_root_path}/data/{genome}_{len_name}_cnn_one_hot_map.json'

    chr_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
    fasta_file = f'{data_root_path}/data/hg19/hg19_chr.fa'
    fasta_open = pysam.Fastafile(fasta_file)

    all_dfs = []
    for each_chr in chr_list:
        bed_file = f'{data_root_path}/data/{genome}/chr_split/{each_chr}_{len_name}.bed'
        df = pd.read_csv(bed_file, sep='\t', header=None)
        df.columns = ['chrom', 'chromStart', 'chromEnd']

        all_dfs.append(df)
    df_samples = pd.concat(all_dfs, ignore_index=True)

    data_dict = {}
    dna_one_hot_chrs = [[] for _ in range(len(chr_list))]
    for idx in tqdm(range(df_samples.shape[0]), desc='saving one hot embedding'):
        chr, start, end = df_samples.iloc[idx, :]

        # save embeddings for each chr
        save_idx = chr_list.index(chr)

        seq_dna = fasta_open.fetch(chr, start, end)
        token_dna = dna_str_to_one_hot(seq_dna)
        dna_one_hot_chrs[save_idx].append(token_dna)

        # save chr position to index of bed file
        key_ = f"{chr}_{start}_{end}"
        chr_list_index = len(dna_one_hot_chrs[save_idx]) - 1
        data_dict[key_] = chr_list_index

    # save embed for each chr
    for chr_idx in range(len(dna_one_hot_chrs)):
        each_chr_one_hot = dna_one_hot_chrs[chr_idx]
        chr_name = chr_list[chr_idx]
        print(f'saving {chr_name}')

        array_file_path = f'{data_root_path}/data/{genome}/one_hot_embed/{genome}_{len_name}_{chr_name}_cnn_one_hot.npz'

        each_chr_one_hot = np.stack(each_chr_one_hot, axis=0)
        np.savez_compressed(array_file_path, each_chr_one_hot)

    # all_dna_one_hot = np.stack(all_dna_one_hot, axis=0)
    #
    # np.savez_compressed(array_file_path, all_dna_one_hot)
    with open(dict_file_path, 'w') as json_file:
        json.dump(data_dict, json_file)


def write_bed_file(genome='hg19',
                   window_size=600_0000,
                   trans_len=200_0000,
                   chr_length_file=f'{data_root_path}/data/chr_length.json'):
    with open(chr_length_file, 'r') as file:
        chr_lengths = json.load(file)
    chr_len = chr_lengths[genome]

    for chrom, size in chr_len.items():
        chr_bed_file = f'{data_root_path}/data/hg19/chr_split/{chrom}_{trans_len}_{window_size}.bed'
        if not os.path.exists(chr_bed_file):
            print('Write new bed files!')
            with open(chr_bed_file, 'w') as bed_file:
                for start in range(0, size, trans_len):
                    if start + window_size <= size:
                        end = start + window_size
                        bed_file.write(f"{chrom}\t{start}\t{end}\n")


class Dim2GraphDataset:
    def __init__(
            self,
            tokenizer,
            cell_type,
            exper_name_cage,
            exper_name_dnase,
            exper_name_h3k27ac,
            exper_name_h3k4me3,
            genome='hg19',
            organism='human',
            resolution=5000,
            assay_type='HiC',
            qval=0.1,
            valid_chr='1,11',
            test_chr='2,12',
            tokenizer_name='char',
            disable_graph=False,
            disable_hic=False,
            disable_adj=False,
            keep_close=None,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.cell_type = cell_type
        self.exper_name_cage = exper_name_cage
        self.exper_name_dnase = exper_name_dnase
        self.exper_name_h3k27ac = exper_name_h3k27ac
        self.exper_name_h3k4me3 = exper_name_h3k4me3
        self.assay_type = assay_type
        self.genome = genome
        self.tokenizer_name = tokenizer_name
        self.disable_graph = disable_graph
        self.disable_hic = disable_hic
        self.disable_adj = disable_adj
        self.keep_close = keep_close
        self.epi_resolution = 100

        qval_to_fdr = {
            0.1: '1',
            0.01: '01',
            0.001: '001',
            0.5: '5',
            0.9: '9',
        }
        self.fdr = qval_to_fdr[qval]

        if organism == 'human' and genome == 'hg19':
            self.chr_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
            self.fasta_file = f'{data_root_path}/data/hg19/hg19_chr.fa'

        # valid and test chr list
        self.valid_chr_list = ['chr' + chr_num for chr_num in valid_chr.split(',')]
        self.test_chr_list = ['chr' + chr_num for chr_num in test_chr.split(',')]

        # chr length
        with open(f'{data_root_path}/data/chr_length.json', 'r') as file:
            chr_lengths = json.load(file)
        self.chr_len = chr_lengths[genome]

        self.init_datasets()

    def init_datasets(self):
        # initialize the one hot embedding / char tokenizer
        if self.tokenizer_name == 'one_hot':
            print(f"reading one-hot embedding, it may need some time")
            self.chr_embed_dict = {}
            for chr_name in self.chr_list:
                array_file_path = f'{data_root_path}/data/{self.genome}/one_hot_embed/{self.genome}_{self.resolution}_{chr_name}_cnn_one_hot.npz'
                with np.load(array_file_path) as data:
                    self.chr_embed_dict[chr_name] = data['arr_0'].copy()

            with open(f'{data_root_path}/data/{self.genome}/one_hot_embed/{self.genome}_{self.resolution}_cnn_one_hot_map.json', 'r') as json_file:
                self.data_dict = json.load(json_file)
        elif self.tokenizer_name == 'char':
            self.fasta = pysam.Fastafile(self.fasta_file)

        # read and return the data
        all_num_nodes = []
        graph_list = []
        for each_chr in tqdm(self.chr_list, desc='Creating Graph Dataset'):
            cur_chr_length = self.chr_len[each_chr]
            # read resolution bed split file
            bed_resolution_file = f'{data_root_path}/data/{self.genome}/chr_split/{each_chr}_{self.resolution}.bed'
            df_resolution = pd.read_csv(bed_resolution_file, sep='\t', header=None)
            df_resolution.columns = ['chrom', 'chromStart', 'chromEnd']
            num_nodes = len(df_resolution)
            all_num_nodes.append(num_nodes)

            # attribute x (chr, start, end)
            x_value = df_resolution.apply(lambda row: f"{row['chrom']}_{row['chromStart']}_{row['chromEnd']}", axis=1).tolist()
            dna_embeds = []
            for each_str_x in x_value:
                each_token_dna = self.tokenizer_str(each_str_x)
                dna_embeds.append(each_token_dna)
            dna_embeds = torch.stack(dna_embeds, dim=0)
            x_value = dna_embeds
            assert len(x_value) == num_nodes

            # H3K4me3
            H3K4me3_file = os.path.join(f'{data_root_path}', 'data', '1D', f'{self.cell_type}_H3K4me3',
                                        f'{self.exper_name_h3k4me3}_1_RPGC_{each_chr}.h5')
            H3K4me3_file_open = h5py.File(H3K4me3_file, 'r')
            H3K4me3_value = H3K4me3_file_open['seqs_cov'][:].squeeze(-1)
            # assert len(H3K4me3_value) == num_nodes
            H3K4me3_file_open.close()
            H3K4me3_value = torch.tensor(H3K4me3_value, dtype=torch.float32)
            assert len(H3K4me3_value) == cur_chr_length // self.epi_resolution
            H3K4me3_value = H3K4me3_value[:num_nodes*self.resolution//self.epi_resolution]
            H3K4me3_value = H3K4me3_value.reshape(num_nodes, self.resolution//self.epi_resolution)

            # H3K27ac
            H3K27ac_file = os.path.join(f'{data_root_path}', 'data', '1D', f'{self.cell_type}_H3K27ac',
                                        f'{self.exper_name_h3k27ac}_1_RPGC_{each_chr}.h5')
            H3K27ac_file_open = h5py.File(H3K27ac_file, 'r')
            H3K27ac_value = H3K27ac_file_open['seqs_cov'][:].squeeze(-1)
            # assert len(H3K27ac_value) == num_nodes
            H3K27ac_file_open.close()
            H3K27ac_value = torch.tensor(H3K27ac_value, dtype=torch.float32)
            assert len(H3K27ac_value) == cur_chr_length // self.epi_resolution
            H3K27ac_value = H3K27ac_value[:num_nodes*self.resolution//self.epi_resolution]
            H3K27ac_value = H3K27ac_value.reshape(num_nodes, self.resolution//self.epi_resolution)

            # DNase seq
            DNase_file = os.path.join(f'{data_root_path}', 'data', '1D', f'{self.cell_type}_DNase-seq',
                                      f'{self.exper_name_dnase}_1_RPGC_{each_chr}.h5')
            DNase_file_open = h5py.File(DNase_file, 'r')
            DNase_value = DNase_file_open['seqs_cov'][:].squeeze(-1)
            # assert len(DNase_value) == num_nodes
            DNase_file_open.close()
            DNase_value = torch.tensor(DNase_value, dtype=torch.float32)
            assert len(DNase_value) == cur_chr_length // self.epi_resolution
            DNase_value = DNase_value[:num_nodes*self.resolution//self.epi_resolution]
            DNase_value = DNase_value.reshape(num_nodes, self.resolution//self.epi_resolution)

            # attribute y, CAGE value
            CAGE_file = f'{data_root_path}/data/1D/{self.cell_type}_CAGE/{self.exper_name_cage}_1_RPGC_{each_chr}.h5'
            CAGE_file_open = h5py.File(CAGE_file, 'r')
            y_value = CAGE_file_open['seqs_cov'][:].squeeze(-1)
            assert len(y_value) == num_nodes
            CAGE_file_open.close()
            y_value = torch.tensor(y_value, dtype=torch.float32)

            # attribute tss
            tss_bin_file = f"{data_root_path}/data/TSS/distal_regulation_group_{self.genome}_tss_bins_{each_chr}.npy"
            tss_bin = np.load(tss_bin_file, allow_pickle=True)
            assert len(tss_bin) == num_nodes
            # 90% can = 1, very sparse
            tss_bin = torch.tensor(tss_bin, dtype=torch.float32)

            # attribute edge index, hic matrix
            # read hic matrix
            hic_matrix_file = (f"{data_root_path}/data/3D/{self.cell_type}_{self.assay_type}/{self.assay_type}_"
                               f"{self.cell_type}_FDR_{self.fdr}_matrix_{each_chr}.npz")
            sparse_matrix = scipy.sparse.load_npz(hic_matrix_file)[:num_nodes, :num_nodes]

            sparse_matrix.data = np.minimum(sparse_matrix.data, 1000)
            sparse_matrix.data = np.log2(sparse_matrix.data + 1)
            hic_slice_lil = sparse_matrix.tolil()
            hic_slice_lil.setdiag(0)
            sparse_matrix = hic_slice_lil.tocsr()
            # sparse_matrix.data[sparse_matrix.data > 0] = 1
            edge_index, edge_attr = from_scipy_sparse_matrix(sparse_matrix)

            # create adj connection
            new_edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)] +
                                          [[i + 1, i] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()
            new_edge_attr = torch.ones(new_edge_index.shape[1], dtype=torch.float32)
            if self.keep_close:
                edge_diff = torch.abs(edge_index[0] - edge_index[1])
                mask = edge_diff < self.keep_close
                edge_index = edge_index[:, mask]
                edge_attr = edge_attr[mask]

            if self.disable_hic:
                edge_index = new_edge_index
                edge_attr = new_edge_attr
            elif self.disable_adj:
                pass
            else:
                edge_index = torch.cat([edge_index, new_edge_index], dim=1)
                edge_attr = torch.cat([edge_attr, new_edge_attr])
            if self.disable_graph:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0,), dtype=torch.float32)

            # create graph data
            data = Data(x=x_value, edge_index=edge_index, edge_attr=edge_attr, y=y_value, num_nodes=num_nodes)
            data.tss = tss_bin
            data.dnase = DNase_value
            data.h3k27ac = H3K27ac_value
            data.h3k4me3 = H3K4me3_value

            graph_list.append(data)

        # create train, valid, test mask
        total_num_nodes = sum(all_num_nodes)

        train_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
        valid_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
        start_idx = 0
        for idx, each_chr in enumerate(self.chr_list):
            cur_start, cur_end = start_idx, start_idx + all_num_nodes[idx]
            if each_chr in self.valid_chr_list:
                valid_mask[cur_start:cur_end] = True
            elif each_chr in self.test_chr_list:
                test_mask[cur_start:cur_end] = True
            else:
                train_mask[cur_start:cur_end] = True
            start_idx += all_num_nodes[idx]
        assert torch.sum(train_mask).item() + torch.sum(valid_mask).item() + torch.sum(test_mask).item() == total_num_nodes

        # merge graph data
        merge_graph = Batch.from_data_list(graph_list)
        self.merge_graph = Data(
            x=merge_graph.x,
            edge_attr=merge_graph.edge_attr,
            edge_index=merge_graph.edge_index,
            y=merge_graph.y,
            num_nodes=merge_graph.num_nodes,
            train_mask=train_mask,
            val_mask=valid_mask,
            test_mask=test_mask,
            tss=merge_graph.tss,
            h3k4me3=merge_graph.h3k4me3,
            h3k27ac=merge_graph.h3k27ac,
            dnase=merge_graph.dnase,
        )

    def tokenizer_str(self, chrm_str):
        chrm, start, end = chrm_str.split("_")
        if self.tokenizer_name == 'one_hot':
            cnn_idx = self.data_dict[f"{chrm}_{start}_{end}"]
            token_dna = self.chr_embed_dict[chrm][cnn_idx]
            token_dna = torch.tensor(token_dna, dtype=torch.float32)
        elif self.tokenizer_name == 'char':
            seq_dna = self.fasta.fetch(chrm, start, end)
            token_dna = self.tokenizer(seq_dna, padding="max_length",
                                       max_length=self.resolution, add_special_tokens=False)['input_ids']
            token_dna = torch.LongTensor(token_dna)
        return token_dna


class Dim2Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            cell_type,
            exper_str,
            genome='hg19',
            organism='human',
            consider_region=6000000,
            pred_region=2000000,
            resolution=5000,
            assay_type='HiC',
            qval=0.1,
            valid_chr='1,11',
            test_chr='2,12',
            data_split='train',
            tokenizer_name='char',
            disable_graph=False,
    ):
        self.resolution = resolution
        self.consider_region = consider_region
        self.pred_region = pred_region
        self.cell_type = cell_type
        self.exper_str = exper_str
        self.assay_type = assay_type
        self.genome = genome
        self.disable_graph = disable_graph

        self.pooled_length = consider_region // resolution

        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

        qval_to_fdr = {
            0.1: '1',
            0.01: '01',
            0.001: '001',
            0.5: '5',
            0.9: '9',
        }
        self.fdr = qval_to_fdr[qval]

        if organism == 'human' and genome == 'hg19':
            self.chr_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
            self.fasta_file = f'{data_root_path}/data/hg19/hg19_chr.fa'
        # elif organism == 'human' and genome == 'hg38':
        #     self.chr_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
        #     self.fasta_file = data_path + '/data/genome/GRCh38.primary_assembly.genome.fa'
        # elif organism == 'mouse':
        #     self.chr_list = ['chr' + str(i) for i in range(1, 20)] + ['chrX']
        #     self.fasta_file = data_path + '/data/genome/mm10.fa'

        # select chr list
        if data_split == 'valid':
            self.chr_list = ['chr' + chr_num for chr_num in valid_chr.split(',')]
        elif data_split == 'test':
            self.chr_list = ['chr' + chr_num for chr_num in test_chr.split(',')]
        elif data_split == 'train':
            valid_chr_range = ['chr' + chr_num for chr_num in valid_chr.split(',')]
            test_chr_range = ['chr' + chr_num for chr_num in test_chr.split(',')]
            self.chr_list = [chr_str for chr_str in self.chr_list
                             if chr_str not in valid_chr_range
                             and chr_str not in test_chr_range]

        # init datasets
        self.init_datasets()

    def init_datasets(self):
        # create & read the chr bed file
        all_dfs = []
        for each_chr in self.chr_list:
            bed_file = f'{data_root_path}/data/{self.genome}/chr_split/{each_chr}_{self.pred_region}_{self.consider_region}.bed'
            df = pd.read_csv(bed_file, sep='\t', header=None)
            df.columns = ['chrom', 'chromStart', 'chromEnd']
            all_dfs.append(df)
        self.df_samples = pd.concat(all_dfs, ignore_index=True)

        # initialize the one hot embedding
        if self.tokenizer_name == 'one_hot':
            print(f"reading one-hot embedding, it may need some time")
            self.chr_embed_dict = {}
            for chr_name in self.chr_list:
                array_file_path = f'{data_root_path}/data/{self.genome}/one_hot_embed/{self.genome}_{self.pred_region}_{self.consider_region}_{chr_name}_cnn_one_hot.npz'
                with np.load(array_file_path) as data:
                    self.chr_embed_dict[chr_name] = data['arr_0'].copy()

            with open(f'{data_root_path}/data/{self.genome}_{self.pred_region}_{self.consider_region}_cnn_one_hot_map.json', 'r') as json_file:
                self.data_dict = json.load(json_file)
        elif self.tokenizer_name == 'char':
            self.fasta = pysam.Fastafile(self.fasta_file)

        self.chr_CAGE, self.chr_adj, self.chr_tss = {}, {}, {}
        for each_chr in self.chr_list:
            # print(f"processing {each_chr}")
            # read resolution bed split file
            bed_resolution_file = f'{data_root_path}/data/{self.genome}/chr_split/{each_chr}_{self.resolution}.bed'
            df_resolution = pd.read_csv(bed_resolution_file, sep='\t', header=None)
            df_resolution.columns = ['chrom', 'chromStart', 'chromEnd']
            num_nodes = len(df_resolution)

            # read CAGE file
            CAGE_file = f'{data_root_path}/data/1D/{self.cell_type}_CAGE/{self.exper_str}_1_RPGC_{each_chr}.h5'

            CAGE_file_open = h5py.File(CAGE_file, 'r')
            # seq_pool_len = CAGE_file_open['seqs_cov'].shape[1]
            # num_targets = 1
            # num_targets_tfr = num_targets
            # targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype='float32')

            targets_y = CAGE_file_open['seqs_cov'][:].squeeze(-1)
            assert len(targets_y) == num_nodes
            self.chr_CAGE[each_chr] = targets_y
            CAGE_file_open.close()

            # read tss
            tss_bin_file = f"{data_root_path}/data/TSS/distal_regulation_group_{self.genome}_tss_bins_{each_chr}.npy"
            tss_bin = np.load(tss_bin_file, allow_pickle=True)
            self.chr_tss[each_chr] = tss_bin
            assert len(tss_bin) == num_nodes
            # 90% can = 1, very sparse

            # read hic matrix
            hic_matrix_file = (f"{data_root_path}/data/3D/{self.cell_type}_{self.assay_type}/{self.assay_type}_"
                               f"{self.cell_type}_FDR_{self.fdr}_matrix_{each_chr}.npz")
            sparse_matrix = scipy.sparse.load_npz(hic_matrix_file)[:num_nodes, :num_nodes]
            self.chr_adj[each_chr] = sparse_matrix

    def __len__(self):
        return len(self.df_samples)

    def __getitem__(self, idx):
        chr, start, end = self.df_samples.iloc[idx,:]
        start_pool, end_pool = start // self.resolution, end // self.resolution
        assert end_pool - start_pool == self.pooled_length

        # tss
        tss_idx = self.chr_tss[chr][start_pool:end_pool]
        tss_idx = torch.tensor(tss_idx, dtype=torch.float32)

        # adj matrix
        hic_slice = self.chr_adj[chr][start_pool:end_pool, start_pool:end_pool]

        hic_slice.data = np.minimum(hic_slice.data, 1000)
        hic_slice.data = np.log2(hic_slice.data + 1)
        # hic_slice.setdiag(0)
        hic_slice_lil = hic_slice.tolil()
        hic_slice_lil.setdiag(0)
        hic_slice = hic_slice_lil.tocsr()

        hic_slice.data[hic_slice.data > 0] = 1
        edge_index, edge_attr = from_scipy_sparse_matrix(hic_slice)

        graph_data = Data(edge_index=edge_index, num_nodes=self.pooled_length)
        if self.disable_graph:
            graph_data = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=self.pooled_length)

        # DNA string
        if self.tokenizer_name == 'one_hot':
            # token_dna = DNA_str_to_one_hot(seq_dna)
            cnn_idx = self.data_dict[f"{chr}_{start}_{end}"]
            token_dna = self.chr_embed_dict[chr][cnn_idx]
            token_dna = torch.tensor(token_dna, dtype=torch.float32)
        elif self.tokenizer_name == 'char':
            seq_dna = self.fasta.fetch(chr, start, end)
            token_dna = self.tokenizer(seq_dna, padding="max_length",
                                 max_length=self.consider_region, add_special_tokens=False)['input_ids']
            token_dna = torch.LongTensor(token_dna)
        else:
            raise NotImplementedError()

        # target cage
        target_CAGE = self.chr_CAGE[chr][start_pool:end_pool]
        target_CAGE = torch.tensor(target_CAGE, dtype=torch.float32)

        # concat signals at resolution level
        signals = torch.concat([tss_idx.unsqueeze(-1), target_CAGE.unsqueeze(-1)], dim=-1)

        return graph_data, signals, token_dna


if __name__ == '__main__':
    save_one_hot_embed(pred_region=5000, consider_region=5000)

    # def custom_collate_fn(batch):
    #     graph_data, tss_idx, token_dna, target_CAGE = zip(*batch)
    #     batch_graph = Batch.from_data_list(graph_data)
    #     tss_idx = torch.tensor(np.array(tss_idx), dtype=torch.float32)
    #     token_dna = torch.tensor(np.array(token_dna), dtype=torch.float32)
    #     target_CAGE = torch.tensor(np.array(target_CAGE), dtype=torch.float32)
    #
    #     return batch_graph, tss_idx, token_dna, target_CAGE
    #
    #
    # def evaluation_step(model, eva_dataloader, device):
    #     model.eval()
    #     epoch_loss = 0.0
    #     with torch.no_grad():
    #         for batch_data in tqdm(eva_dataloader, desc='evaluation'):
    #             batch_graph, tss_idx, dna_ids, targets = batch_data
    #             batch_graph = batch_graph.to(device)
    #             dna_ids = dna_ids.to(device)
    #             targets = targets.to(device)
    #
    #             pred_y, h, att = model(dna_ids, batch_graph)
    #             loss = loss_func(pred_y, targets)
    #
    #             epoch_loss += loss.item()
    #
    #     epoch_loss = epoch_loss / len(eva_dataloader)
    #     return epoch_loss
    #
    #
    # def set_seed(seed):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # set_seed(1)
    #
    # import logging
    # logging.basicConfig(filename=f'{data_root_path}/tmp_log/graphreg.log',
    #                     level=logging.INFO,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    #
    # organism = 'human'          # human/mouse
    # cell_line = 'GM12878'          # K562/GM12878/hESC/mESC
    # resolution = 5000
    # genome='hg19'               # hg19/hg38/mm10
    # pred_type = 'seq'               # seq/epi
    # assay_type = 'HiC'        # HiC/HiChIP/MicroC/HiCAR
    # qval = 0.1                    # 0.1/0.01/0.001
    # exper_name = 'ENCFF915EIJ'
    # consider_seq_len = 600_0000
    # pred_seq_len = 200_0000
    #
    # num_epochs = 100
    # lr = 2e-4
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # model
    # model = SeqGraphRegE2E(F=4, N=consider_seq_len // resolution, n_gat_layers=3).to(device)
    #
    # # dataset and dataloader
    # train_dataset, valid_dataset, test_dataset = [
    #     Dim2Dataset(cell_line, exper_name, genome=genome, organism=organism, consider_region=consider_seq_len,
    #                 pred_region=pred_seq_len, resolution=resolution, assay_type=assay_type,
    #                 qval=qval,
    #                 pred_type=pred_type, data_split=each_data_split) for each_data_split in ['train', 'valid', 'test']
    # ]
    # dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    # val_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)
    #
    # # optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    #
    # loss_func = nn.PoissonNLLLoss(log_input=False, full=True)
    #
    # best_loss = 1e8
    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0.0
    #     pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    #     for it, batch_data in pbar:
    #         batch_graph, tss_idx, dna_ids, targets = batch_data
    #         batch_graph = batch_graph.to(device)
    #         tss_idx = tss_idx.to(device)
    #         dna_ids = dna_ids.to(device)
    #         targets = targets.to(device)
    #
    #         pred_y, h, att = model(dna_ids, batch_graph)
    #         loss = loss_func(pred_y, targets)
    #         # loss = poisson_loss(targets, pred_y)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
    #         optimizer.step()
    #
    #         epoch_loss += loss.item()
    #
    #         pbar.set_description(
    #             f"Iteration {it}, epoch {epoch}: train loss {loss.item():.5f}")
    #         # logger.info(f"batch loss {loss.item():.5f}")
    #         # break
    #     epoch_loss = epoch_loss / len(dataloader)
    #     logger.info(f"epoch {epoch}, training loss {epoch_loss:.5f}")
    #
    #     # validation
    #     val_loss = evaluation_step(model, val_dataloader, device)
    #     logger.info(f"val loss {val_loss:.5f}")
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         torch.save(model.state_dict(), f'{data_root_path}/tmp_saved_model/cnn_model.pt')
    #
    # # testing
    # model.load_state_dict(torch.load(f'{data_root_path}/tmp_saved_model/cnn_model.pt'))
    # test_loss = evaluation_step(model, test_dataloader, device)
    # logger.info(f"test loss {test_loss:.5f}")
