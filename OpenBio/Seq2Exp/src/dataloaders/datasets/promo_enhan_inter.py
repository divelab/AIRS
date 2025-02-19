import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, Dataset
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
import pyfaidx
import pyBigWig
from transformers import AutoModelForMaskedLM, AutoTokenizer
import gzip
import pickle

from src.dataloaders.utils.dna import dna_str_to_one_hot, one_hot_to_dna_str
from src.dataloaders.utils.epinformer import gene_pro_enh_pos


def is_region_blacklisted(chrom, start, end, df_blacklist):
    # Step 1: 筛选出与给定染色体匹配的行
    df_chrom = df_blacklist[df_blacklist['chrom'] == chrom]

    # Step 2: 查找是否存在任何与给定起始和结束位置重叠的区域
    overlap = df_chrom[(df_chrom['start'] <= end) & (df_chrom['end'] >= start)]

    # 如果有重叠的区域，返回 True，否则返回 False
    if not overlap.empty:
        return True, overlap
    else:
        return False, None


def mask_blacklisted_region(chrom, start, end, sequence, df_blacklist, mask_token, data_type='seq'):
    """
    这个函数会将sequence中属于blacklist区域的部分替换为'N'。
    chrom: 染色体
    start, end: 这段序列在染色体上的起止位置
    sequence: 已经提取好的序列（包括padding）
    df_blacklist: blacklist DataFrame
    mask_char: 用来替换blacklist区域的字符，默认为'N'
    """
    # 找到与给定位置重叠的黑名单区域
    is_blacklisted, overlapping_regions = is_region_blacklisted(chrom, start, end, df_blacklist)

    # 如果有重叠区域，修改序列
    if is_blacklisted:
        sequence = list(sequence)  # 将序列转换为列表以便修改
        for _, region in overlapping_regions.iterrows():
            # 计算重叠部分的相对索引
            region_start = max(start, region['start'])  # 黑名单区域的起始
            region_end = min(end, region['end'])  # 黑名单区域的结束

            # 相对序列的索引
            rel_start = region_start - start
            rel_end = region_end - start

            # 替换序列中对应的部分为 mask_char
            sequence[rel_start:rel_end + 1] = [mask_token] * (rel_end - rel_start + 1)

        # 将列表转换回字符串
        if data_type == 'seq':
            sequence = ''.join(sequence)
        elif data_type == 'hic':
            sequence = np.array(sequence)

    return sequence


def get_padded_seq(chr_length, data_source, chrm, tss, each_total_lens, genome='hg38', data_type='seq', df_blacklist=None):
    chrom_length = chr_length[genome][chrm]
    start = max(0, tss - each_total_lens)
    end = min(chrom_length, tss + each_total_lens)
    left_padding = max(0, each_total_lens - tss)
    right_padding = max(0, (tss + each_total_lens) - chrom_length)

    if data_type == 'seq':
        sequence = data_source[chrm][start:end].seq
        # sequence = mask_blacklisted_region(chrm, start, end, sequence, df_blacklist, mask_token='N', data_type=data_type)
        padded_sequence = 'N' * left_padding + sequence + 'N' * right_padding
    elif data_type == 'signal':
        sequence = data_source.values(chrm, start, end)
        # sequence = mask_blacklisted_region(chrm, start, end, sequence, df_blacklist, mask_token=0.0, data_type=data_type)
        padded_sequence = [0.0] * left_padding + sequence + [0.0] * right_padding
    elif data_type == 'hic':
        sequence = data_source[chrm][f'{chrm}_{tss}'][:]
        # sequence = mask_blacklisted_region(chrm, start, end, sequence, df_blacklist, mask_token=0.0, data_type=data_type)
        padded_sequence = np.pad(sequence, (left_padding, right_padding), mode='constant', constant_values=0)
    elif data_type == 'mask':
        sequence = data_source[chrm][start:end]
        padded_sequence = np.pad(sequence, (left_padding, right_padding), mode='constant', constant_values=0)
    else:
        raise NotImplementedError()

    return padded_sequence


class PromoterEnhancerDataset(Dataset):
    def __init__(
            self,
            data_folder,
            expr_type='CAGE',
            usePromoterSignal=True,
            signal_type='H3K27ac',
            cell_type='K562',
            distance_threshold=None,
            hic_threshold=None,
            n_enhancers=50,
            n_extraFeat=1,
            zero_dist=False, zero_activity=False, zero_hic=False,
            omit_enhancers=False, only_seqs=False,
    ):
        self.expr_type = expr_type
        self.cell_type = cell_type
        self.data_folder = data_folder
        self.n_enhancers = n_enhancers
        self.signal_type = signal_type
        self.n_extraFeat = n_extraFeat
        self.usePromoterSignal = usePromoterSignal
        self.distance_threshold = distance_threshold
        self.hic_threshold = hic_threshold
        self.zero_dist = zero_dist
        self.zero_activity = zero_activity
        self.zero_hic = zero_hic
        self.omit_enhancers = omit_enhancers
        self.only_seqs = only_seqs

        self.init_datasets()
        self.init_worker()

    def init_datasets(self):
        if self.cell_type == 'K562':
            promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                   'K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt'),
                                      sep='\t', index_col='symbol')
            promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
            self.promoter_df = promoter_df

        elif self.cell_type == 'GM12878':
            promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt'), sep='\t', index_col='symbol')
            promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
            self.promoter_df = promoter_df

        elif self.cell_type == 'H1':
            promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'H1_ABC/Neighborhoods/GeneList.txt'), sep='\t', index_col='symbol')
            promoter_df['PromoterActivity'] = np.sqrt(promoter_df['H3K27ac.RPM.TSS1Kb']*promoter_df['DHS.RPM.TSS1Kb'])
            self.promoter_df = promoter_df
            self.ensid_names = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'leave_chrom_out_crossvalidation_split_18377genes.csv'))[['ENSID', 'Gene name']]

        self.expr_df = pd.read_csv(
            os.path.join(self.data_folder, 'promo_enhan_inter', 'GM12878_K562_18377_gene_expr_fromXpresso.csv'),
            index_col='ENSID')
        with open(os.path.join(self.data_folder, 'chr_length.json'), 'r') as file:
            self.chr_lengths = json.load(file)

    def init_worker(self):
        self.genome = pyfaidx.Fasta(os.path.join(self.data_folder, 'hg38', 'hg38.ml.fa'))
        if self.cell_type == 'K562':
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                  'K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5'),
                                     'r')
        elif self.cell_type == 'GM12878':
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                  'GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5'),
                                     'r')
        elif self.cell_type == 'H1':
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                  'H1_promoter_enhancer_encoding.h5'),
                                     'r')
            self.cage = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'CAGE', self.cell_type, 'H1_CAGE_CNhs13964.bw'))

    def __len__(self):
        return len(self.data_h5['ensid'])

    def __getitem__(self, idx):
        sample_ensid = self.data_h5['ensid'][idx].decode()  # e.g. 'ENSG00000000003'
        seq_code = self.data_h5['pe_code'][idx]  # 61, 2000, 4, this is the sequence
        enhancer_distance = self.data_h5['distance'][idx,1:]  # 60
        enhancer_intensity = self.data_h5['activity'][idx,1:]  # 60
        enhancer_contact = self.data_h5['hic'][idx,1:]  # 60

        if self.cell_type == 'H1':
            gene_name = self.ensid_names[self.ensid_names['ENSID'] == sample_ensid]['Gene name'].item()
            chrm, tss = self.promoter_df.loc[gene_name][['chr', 'tss']]
        else:
            chrm, tss = self.promoter_df.loc[sample_ensid][['chr', 'tss']]

        if self.omit_enhancers:
            self.zero_dist = True
            self.zero_activity = True
            self.zero_hic = True
            seq_code[1:,:,:] = np.zeros_like(seq_code[1:,:,:])

        if self.only_seqs:
            self.zero_dist = True
            self.zero_activity = True
            self.zero_hic = True

            # promo_str = self.genome[chrm][tss-1000:tss+1000].seq
            # promo_str_h5 = one_hot_to_dna_str(seq_code[0])
            # assert promo_str == promo_str_h5
            bs, length, _ = seq_code.shape
            each_total_lens = bs * length // 2

            seq_code_seqs = get_padded_seq(self.chr_lengths, self.genome, chrm, tss, each_total_lens, data_type='seq')

            # seq_code_seqs = self.genome[chrm][tss - each_total_lens: tss + each_total_lens].seq
            assert len(seq_code_seqs) == each_total_lens * 2
            seq_code = dna_str_to_one_hot(seq_code_seqs).reshape(seq_code.shape)
            mid_idx = (bs - 1) // 2
            seq_code = np.concatenate((seq_code[mid_idx:(mid_idx+1)],
                                       seq_code[:mid_idx],
                                       seq_code[mid_idx+1:]), axis=0)

        if self.signal_type == 'H3K27ac':
            promoter_activity = self.promoter_df.loc[gene_name]['PromoterActivity'] if self.cell_type == 'H1' else self.promoter_df.loc[sample_ensid]['PromoterActivity']
        elif self.signal_type == 'DNase':
            promoter_activity = self.promoter_df.loc[sample_ensid]['normalized_dhs']  # scalar
        promoter_code = seq_code[:1]
        enhancers_code = seq_code[1:]

        rnaFeat = list(self.expr_df.loc[sample_ensid][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore','UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float))
        # len=8
        if self.usePromoterSignal:
            pe_activity = np.concatenate([[0], enhancer_intensity]).flatten()
            rnaFeat = np.array(rnaFeat + [promoter_activity])  # len: 9
        else:
            pe_activity = np.concatenate([[0], enhancer_intensity]).flatten()
            rnaFeat = np.array(rnaFeat + [0])

        if self.distance_threshold is not None:
            # only keep enhancer with distance < threshold
            # keep the one-hot embedding, and the distance
            enhancer_distance = enhancer_distance.flatten()
            enhancers_zero = np.zeros_like(enhancers_code)  # 60,2000,4
            enhancers_zero[abs(enhancer_distance) < self.distance_threshold] = enhancers_code[abs(enhancer_distance) < self.distance_threshold]
            enhancers_code = enhancers_zero

            enhancer_distance_zero = np.zeros_like(enhancer_distance)
            enhancer_distance_zero[abs(enhancer_distance) < self.distance_threshold] = enhancer_distance[abs(enhancer_distance) < self.distance_threshold]
            enhancer_distance = enhancer_distance_zero

        if self.hic_threshold is not None:
            enhancer_contact = enhancer_contact.flatten()
            enhancers_zero = np.zeros_like(enhancers_code)
            enhancers_zero[enhancer_contact > self.hic_threshold] = enhancers_code[enhancer_contact > self.hic_threshold]
            enhancers_code = enhancers_zero

            enhancer_contact_zero = np.zeros_like(enhancer_contact)
            enhancer_contact_zero[enhancers_code[enhancer_contact > self.hic_threshold ]] = enhancer_contact[enhancers_code[enhancer_contact > self.hic_threshold]]
            enhancer_contact = enhancer_contact_zero

        pe_hic = np.concatenate([[0], enhancer_contact]).flatten()
        pe_hic = np.log10(1+pe_hic)  # 61
        pe_distance = np.concatenate([[0], enhancer_distance/1000]).flatten()  # 61
        if self.n_extraFeat == 1:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis]],axis=-1)
        elif self.n_extraFeat == 2:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis], pe_activity[:,np.newaxis]],axis=-1)
        elif self.n_extraFeat == 3:
            pe_distance = np.zeros_like(pe_distance) if self.zero_dist else pe_distance
            pe_activity = np.zeros_like(pe_activity) if self.zero_activity else pe_activity
            pe_hic = np.zeros_like(pe_hic) if self.zero_hic else pe_hic
            pe_hic = np.nan_to_num(pe_hic, nan=0.0)
            pe_feat = np.concatenate([pe_distance[:,np.newaxis], pe_hic[:,np.newaxis], pe_activity[:,np.newaxis], ],axis=-1)
        else:
            pe_feat = np.concatenate([pe_distance[:,np.newaxis]],axis=-1)

        promoter_code_tensor = torch.from_numpy(promoter_code).float()  # 1,2000,4
        pe_feat_tensor = torch.from_numpy(pe_feat[:self.n_enhancers+1]).float()  # 61,3
        if self.n_extraFeat == 0: # Use promoter only
            enhancers_code = np.zeros_like(enhancers_code[:self.n_enhancers, :])
        enhancers_code_tensor = torch.from_numpy(enhancers_code[:self.n_enhancers, :]).float()  # 60,2000,4
        pe_code_tensor = torch.concat([promoter_code_tensor, enhancers_code_tensor])  # 61,2000,4
        rnaFeat_tensor = torch.from_numpy(rnaFeat).float()  # 9

        if self.expr_type == 'CAGE':
            if self.cell_type == 'H1':
                cage_expr = get_padded_seq(self.chr_lengths, self.cage, chrm, tss, 192, data_type='signal')
                cage_expr = torch.tensor(cage_expr, dtype=torch.float32)
                cage_expr = torch.nan_to_num(cage_expr, nan=0.0)
                cage_expr = cage_expr.sum(dim=0, keepdim=True)
                expr_tensor = torch.log10(cage_expr+1).float()
            else:
                cage_expr = np.log10(self.expr_df.loc[sample_ensid][self.cell_type + '_CAGE_128*3_sum']+1)
                expr_tensor = torch.from_numpy(np.array([cage_expr])).float()
        elif self.expr_type == 'RNA':
            rna_expr = self.expr_df.loc[sample_ensid]['Actual_' + self.cell_type]
            expr_tensor = torch.from_numpy(np.array([rna_expr])).float()  # scalar
        else:
            assert False, 'label not exists!'
        return pe_code_tensor, rnaFeat_tensor, pe_feat_tensor, expr_tensor, sample_ensid


class ExperInteractDataset(Dataset):
    def __init__(
            self,
            data_folder,
            split_df=None,
            expr_type='CAGE',
            cell_type='K562',
            distance_threshold=None,
            hic_threshold=None,
            n_enhancers=50,
            seq_range=200_000,
            tokenizer_name='one_hot',
            pretrained_model=False,
            pretrained_model_name=None,
            blacklist_ver=None,
            zero_dist=False, zero_activity=False, zero_hic=False,
            omit_enhancers=False, only_seqs=False,
    ):
        self.split_df = split_df
        self.expr_type = expr_type
        self.cell_type = cell_type
        self.data_folder = data_folder
        self.n_enhancers = n_enhancers
        self.distance_threshold = distance_threshold
        self.hic_threshold = hic_threshold
        self.seq_range = seq_range
        self.tokenizer_name = tokenizer_name
        self.pretrained_model = pretrained_model
        self.pretrained_model_name = pretrained_model_name
        self.blacklist_ver = blacklist_ver

        self.zero_dist = zero_dist
        self.zero_activity = zero_activity
        self.zero_hic = zero_hic
        self.omit_enhancers = omit_enhancers
        self.only_seqs = only_seqs

        if tokenizer_name == 'char':
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

        self.all_chrs = ['chr' + str(i) for i in list(range(1, 23)) + ['X']]

        with open(os.path.join(self.data_folder, 'chr_length.json'), 'r') as file:
            self.chr_lengths = json.load(file)

        self.expr_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'GM12878_K562_18377_gene_expr_fromXpresso.csv'), index_col='ENSID')

        # RNA_seq
        self.rna_seq = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'RNA_seq', '57epigenomes.RPKM.pc'), sep='\t')

        # Read experimentally validated enhancers
        # Orignal CRISPRi-FlowFISH data from Fulco et al. 2019
        flowfish_df = pd.read_excel(os.path.join(self.data_folder, 'promo_enhan_inter', 'CRISPRi-FlowFISH_Fulco2019', '41588_2019_538_MOESM3_ESM.xlsx'),
                                         sheet_name='Supplementary Table 6a', skiprows=[0])
        # select the significant enhancers
        # flowfish_df['significant_enhancer'] = flowfish_df['Significant'] & (flowfish_df['Fraction change in gene expr'] < 0)
        # flowfish_df = flowfish_df[flowfish_df['significant_enhancer']]

        RNA_feats = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter', 'GM12878_K562_18377_gene_expr_fromXpresso.csv'), index_col='Gene stable ID')[
            ['Gene name', 'UTR5LEN_log10zscore', 'CDSLEN_log10zscore', 'INTRONLEN_log10zscore', 'UTR3LEN_log10zscore',
             'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY_log10zscore']]
        ensid_gene = RNA_feats[['Gene name']].reset_index()
        flowfish_df = flowfish_df.merge(ensid_gene, left_on='Gene', right_on='Gene name', how='left')
        # flowfish_df['Xpresso_gene'] = False
        # intersec_ensid = list(set(flowfish_df['Gene stable ID']).intersection(set(RNA_feats.index)))
        # flowfish_df = flowfish_df.set_index('Gene stable ID')
        # flowfish_df.loc[intersec_ensid, 'Xpresso_gene'] = True
        # # Keep the Xpresso selected genes (protein-coding gene)
        # flowfish_df = flowfish_df[flowfish_df['Xpresso_gene']].reset_index()
        # flowfish_df['mid'] = flowfish_df['start'] + (flowfish_df['end'] - flowfish_df['start']) / 2
        # flowfish_df['Distance_withDirect'] = flowfish_df['mid'] - flowfish_df['Gene TSS']
        # flowfish_df['Distance'] = np.abs(flowfish_df['Distance_withDirect'])
        # # Retain the enhancer-gene within 100kb to the TSS
        # flowfish_100kb_df = flowfish_df[flowfish_df['Distance'] <= 100_000]
        # expr_flowfish_100kb_df = self.expr_df[[f'K562_CAGE_128*3_sum', f'Actual_K562']].merge(flowfish_100kb_df, left_index=True, right_on='Gene stable ID', how='right').reset_index()
        # # All gene
        # gene_list = flowfish_100kb_df['Gene name'].dropna().unique()
        self.flowfish_df = flowfish_df
        self.validated_gene_list = self.flowfish_df['Gene stable ID'].dropna().unique()

        self.init_datasets()
        self.init_worker()

    def init_worker(self):
        self.genome = pyfaidx.Fasta(os.path.join(self.data_folder, 'hg38', 'hg38.ml.fa'))
        if self.cell_type == 'K562':
            self.h3k27ac = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'H3K27ac', self.cell_type, 'ENCFF465GBD.bigWig'))
            self.dnase = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'DNase-seq', self.cell_type, 'ENCFF414OGC.bigWig'))
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter', 'K562_DNase_ENCFF257HEE_2kb_4DNFITUOMFUQ_enhancer_promoter_encoding.h5'), 'r')
        elif self.cell_type == 'GM12878':
            self.h3k27ac = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'H3K27ac', self.cell_type, 'ENCFF798KYP.bigWig'))
            self.dnase = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'DNase-seq', self.cell_type, 'ENCFF960FMM.bigWig'))
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter', 'GM12878_DNase_ENCFF020WZB_2kb_4DNFI1UEG1HD_promoter_enhancer_encoding.h5'), 'r')
        elif self.cell_type == 'H1':
            self.h3k27ac = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'H3K27ac', self.cell_type, 'ENCFF771GNB.bigWig'))
            self.dnase = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'DNase-seq', self.cell_type, 'ENCFF232GUZ.bigWig'))
            self.data_h5 = h5py.File(os.path.join(self.data_folder, 'promo_enhan_inter', 'H1_promoter_enhancer_encoding.h5'), 'r')
            self.cage = pyBigWig.open(os.path.join(self.data_folder, 'promo_enhan_inter', 'CAGE', self.cell_type, 'H1_CAGE_CNhs13964.bw'))

        # MACS peak regions
        macs_path = os.path.join(self.data_folder, 'promo_enhan_inter', 'DNase-seq', self.cell_type, 'peaks/peaks_bigWig/macs3.bigWig')
        self.dnase_peaks = pyBigWig.open(macs_path) if os.path.exists(macs_path) else None
        # HiC
        self.hic_dict = {}
        for chrm in self.all_chrs:
            chr_hic_file = h5py.File(
                os.path.join(self.data_folder, 'promo_enhan_inter', 'HiC', self.cell_type, f"hic_feat_{chrm}.h5"), 'r')
            self.hic_dict[chrm] = chr_hic_file

    def init_datasets(self):
        if self.cell_type == 'K562':
            self.promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                        'K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated',
                                                        'DNase_ENCFF257HEE_Neighborhoods',
                                                        'GeneList.txt'),
                                           sep='\t', index_col='symbol')
            self.promoter_df['PromoterActivity'] = np.sqrt(self.promoter_df['H3K27ac.RPM.TSS1Kb']*self.promoter_df['DHS.RPM.TSS1Kb'])
            # enhancer positions & abc features
            self.enhancer_gene = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                          'K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated',
                                                          'Gene_enhancer_links',
                                                          'EnhancerPredictionsAllPutative.txt.gz'), sep='\t')
        elif self.cell_type == 'GM12878':
            self.promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                        'GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated',
                                                        'DNase_ENCFF020WZB_Neighborhoods',
                                                        'GeneList.txt'),
                                           sep='\t', index_col='symbol')
            self.promoter_df['PromoterActivity'] = np.sqrt(self.promoter_df['H3K27ac.RPM.TSS1Kb']*self.promoter_df['DHS.RPM.TSS1Kb'])
            # enhancer positions & abc features
            self.enhancer_gene = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                          'GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated',
                                                          'Gene_enhancer_links',
                                                          'EnhancerPredictionsAllPutative.txt.gz'), sep='\t')
        elif self.cell_type == 'H1':
            self.promoter_df = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                        'H1_ABC',
                                                        'Neighborhoods',
                                                        'GeneList.txt'),
                                           sep='\t', index_col='symbol')
            self.promoter_df['PromoterActivity'] = np.sqrt(
                self.promoter_df['H3K27ac.RPM.TSS1Kb'] * self.promoter_df['DHS.RPM.TSS1Kb'])
            self.enhancer_gene = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                          'H1_ABC',
                                                          'Predictions',
                                                          'EnhancerPredictionsAllPutative.txt.gz'), sep='\t')
            self.ensid_names = pd.read_csv(os.path.join(self.data_folder, 'promo_enhan_inter',
                                                        'leave_chrom_out_crossvalidation_split_18377genes.csv'))[
                ['ENSID', 'Gene name']]

        if self.cell_type == 'H1':
            self.enhancer_gene = self.enhancer_gene[
                (self.enhancer_gene['distance'] <= 100_000) & (self.enhancer_gene['distance'] > 1000)].reset_index()
            gene_tss = self.promoter_df[['name', 'chr', 'tss', 'strand']].merge(self.split_df, left_on='name', right_on='Gene name').drop(columns='name')
            self.enhancer_gene = self.enhancer_gene.merge(gene_tss, left_on='TargetGene', right_on='Gene name',
                                                          # how='right',
                                                          suffixes=['', '_gene']).reset_index()
        else:
            self.enhancer_gene = self.enhancer_gene[(self.enhancer_gene['distance'] <= 100_000) &
                                                    (self.enhancer_gene['distance'] > 1000)].reset_index()
            gene_tss = self.promoter_df[['name', 'chr', 'tss', 'strand']].merge(self.split_df[['ENSID', 'Gene name']],
                                                                                left_on='name', right_on='ENSID').drop(
                columns='name')
            self.enhancer_gene = self.enhancer_gene.merge(gene_tss, left_on='TargetGene', right_on='ENSID', how='right',
                                                          suffixes=['', '_gene']).reset_index()

        # blacklist of ENCODE project
        with open(os.path.join(self.data_folder, f'hg38_blacklist_dict_{self.blacklist_ver}.pkl'), "rb") as f:
            self.blacklist = pickle.load(f)
        # with gzip.open(os.path.join(self.data_folder, "hg38-blacklist.v2.bed.gz"), 'rt') as f:
        #     self.df_blacklist = pd.read_csv(f, sep='\t', header=None, names=['chrom', 'start', 'end', 'reason'])
        # include-list
        with open(os.path.join(self.data_folder, 'hg38_includelist_dict.pkl'), "rb") as f:
            self.includelist = pickle.load(f)

    def __len__(self):
        return len(self.data_h5['ensid'])

    def validate_promo_enhan_str(self, promo_info, enh_pos_info, seq_code):
        # validate promoter and enhancer string, same as processed saved h5 file
        promo_chrm, promo_start, promo_end = promo_info
        promo_str = self.genome[promo_chrm][promo_start:promo_end].seq
        promo_str_h5 = one_hot_to_dna_str(seq_code[0])
        assert promo_str == promo_str_h5

        for enh_idx, (enh_chrm, enh_start, enh_end) in enumerate(enh_pos_info[1]):
            enh_true_length = enh_end - enh_start
            enh_str = self.genome[enh_chrm][enh_start:enh_end].seq
            enh_str_h5 = one_hot_to_dna_str(seq_code[enh_idx+1])
            enh_start_ = int(len(enh_str_h5) / 2) - int(enh_true_length / 2)
            enh_str_h5 = enh_str_h5[enh_start_: enh_start_ + enh_true_length]
            assert enh_str == enh_str_h5

    def __getitem__(self, idx):
        sample_ensid = self.data_h5['ensid'][idx].decode()  # e.g. 'ENSG00000000003'
        promo_info, enh_pos_info, enh_masks, enh_feats = gene_pro_enh_pos(self.enhancer_gene, sample_ensid)
        # seq_code = self.data_h5['pe_code'][idx]  # 61, 2000, 4, this is the sequence
        # enhancer_distance = self.data_h5['distance'][idx,1:]  # 60
        # enhancer_intensity = self.data_h5['activity'][idx,1:]  # 60
        # enhancer_contact = self.data_h5['hic'][idx,1:]  # 60
        # # validate promoter and enhancer strings
        # self.validate_promo_enhan_str(promo_info, enh_pos_info, seq_code)

        # mask selected by biology
        true_select_mask, epinformer_select_mask = enh_masks[0], enh_masks[1]
        true_select_mask = torch.tensor(true_select_mask, dtype=torch.float32)
        epinformer_select_mask = torch.tensor(epinformer_select_mask, dtype=torch.float32)
        bio_masks = torch.concat((true_select_mask.unsqueeze(-1), epinformer_select_mask.unsqueeze(-1)), dim=-1)
        bio_masks = bio_masks[(len(bio_masks)-self.seq_range)//2: (len(bio_masks)+self.seq_range)//2]

        # extract the one-hot sequences
        if self.cell_type == 'H1':
            gene_name = self.ensid_names[self.ensid_names['ENSID'] == sample_ensid]['Gene name'].item()
            chrm, tss = self.promoter_df.loc[gene_name][['chr', 'tss']]
        else:
            chrm, tss = self.promoter_df.loc[sample_ensid][['chr', 'tss']]
        # promo_str = self.genome[chrm][tss-1000:tss+1000].seq
        # promo_str_h5 = one_hot_to_dna_str(seq_code[0])
        # assert promo_str == promo_str_h5
        each_total_lens = self.seq_range // 2

        seq_code_seqs = get_padded_seq(self.chr_lengths, self.genome, chrm, tss, each_total_lens, data_type='seq')
        # seq_code_seqs = self.genome[chrm][tss - each_total_lens: tss + each_total_lens].seq
        assert len(seq_code_seqs) == each_total_lens * 2
        if self.tokenizer_name == 'char':
            seq_code_tensor = self.tokenizer(seq_code_seqs, padding="max_length", max_length=self.seq_range, add_special_tokens=False)['input_ids']
            seq_code_tensor = torch.LongTensor(seq_code_tensor)
        elif self.tokenizer_name == 'one_hot':
            seq_code = dna_str_to_one_hot(seq_code_seqs).reshape(self.seq_range, 4)
            seq_code_tensor = torch.from_numpy(seq_code).float()

        # h3k27ac
        h3k27ac = get_padded_seq(self.chr_lengths, self.h3k27ac, chrm, tss, each_total_lens, data_type='signal')
        h3k27ac = torch.tensor(h3k27ac, dtype=torch.float32)
        h3k27ac = torch.nan_to_num(h3k27ac, nan=0.0)
        assert len(h3k27ac) == each_total_lens * 2
        # dnase
        dnase = get_padded_seq(self.chr_lengths, self.dnase, chrm, tss, each_total_lens, data_type='signal')
        dnase = torch.tensor(dnase, dtype=torch.float32)
        dnase = torch.nan_to_num(dnase, nan=0.0)
        assert len(dnase) == each_total_lens * 2
        # HiC
        hic = get_padded_seq(self.chr_lengths, self.hic_dict, chrm, tss, each_total_lens, data_type='hic')
        hic = torch.tensor(hic, dtype=torch.float32)
        hic = torch.nan_to_num(hic, nan=0.0)
        hic = torch.log10(hic + 1)
        # extract middle part
        hic = hic[(len(hic)-self.seq_range)//2: (len(hic)+self.seq_range)//2]
        assert len(hic) == each_total_lens * 2

        signal_tensor = torch.concat((h3k27ac.unsqueeze(-1), dnase.unsqueeze(-1), hic.unsqueeze(-1)), dim=-1)

        # blacklist & include list
        blacklist = get_padded_seq(self.chr_lengths, self.blacklist, chrm, tss, each_total_lens, data_type='mask')
        includelist = get_padded_seq(self.chr_lengths, self.includelist, chrm, tss, each_total_lens, data_type='mask')
        blacklist = torch.tensor(blacklist, dtype=torch.bool)
        includelist = torch.tensor(includelist, dtype=torch.bool)
        mask_regions = torch.concat((includelist.unsqueeze(-1), blacklist.unsqueeze(-1)), dim=-1)

        # DNase MACS3 select regions
        if self.dnase_peaks is not None:
            peak_mask = get_padded_seq(self.chr_lengths, self.dnase_peaks, chrm, tss, each_total_lens, data_type='signal')
            peak_mask = torch.tensor(peak_mask, dtype=torch.float32)
            peak_mask = torch.nan_to_num(peak_mask, nan=0.0)
            assert len(peak_mask) == each_total_lens * 2
        else:
            peak_mask = torch.zeros(each_total_lens * 2)

        # experimental results
        # validate_express = None
        # if sample_ensid in self.validated_gene_list:
        #     gene_rows = self.flowfish_df[self.flowfish_df['Gene stable ID'] == sample_ensid]
        #     validate_express = gene_rows[['chr', 'start', 'end', 'class', 'Fraction change in gene expr']]

        # rna features
        promoter_activity = self.promoter_df.loc[gene_name]['PromoterActivity'] if self.cell_type == 'H1' else \
        self.promoter_df.loc[sample_ensid]['PromoterActivity']
        rnaFeat = list(self.expr_df.loc[sample_ensid][['UTR5LEN_log10zscore','CDSLEN_log10zscore','INTRONLEN_log10zscore','UTR3LEN_log10zscore','UTR5GC','CDSGC','UTR3GC', 'ORFEXONDENSITY']].values.astype(float))
        rnaFeat = np.array(rnaFeat + [promoter_activity])  # len: 9
        rnaFeat_tensor = torch.from_numpy(rnaFeat).float()

        # labels
        if self.expr_type == 'CAGE':
            if self.cell_type == 'H1':
                cage_expr = get_padded_seq(self.chr_lengths, self.cage, chrm, tss, 192, data_type='signal')
                cage_expr = torch.tensor(cage_expr, dtype=torch.float32)
                cage_expr = torch.nan_to_num(cage_expr, nan=0.0)
                cage_expr = cage_expr.sum(dim=0, keepdim=True)
                expr_tensor = torch.log10(cage_expr+1).float()
            else:
                cage_expr = np.log10(self.expr_df.loc[sample_ensid][self.cell_type + '_CAGE_128*3_sum']+1)
                expr_tensor = torch.from_numpy(np.array([cage_expr])).float()
        elif self.expr_type == 'RNA':
            rna_expr = self.expr_df.loc[sample_ensid]['Actual_' + self.cell_type]
            expr_tensor = torch.from_numpy(np.array([rna_expr])).float()  # scalar
        else:
            assert False, 'label not exists!'
        assert not torch.isnan(seq_code_tensor).any()
        assert not torch.isnan(bio_masks).any()
        assert not torch.isnan(peak_mask).any()
        assert not torch.isnan(rnaFeat_tensor).any()
        assert not torch.isnan(signal_tensor).any()
        assert not torch.isnan(expr_tensor).any()

        return seq_code_tensor, bio_masks, peak_mask, rnaFeat_tensor, signal_tensor, mask_regions, expr_tensor, sample_ensid
