import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyfaidx
import json
import pickle
import gzip
import csv
import h5py

from src.dataloaders.utils.dna import dna_str_to_one_hot


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, chrom, start, end, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[chrom]
        start = max(start, 0)
        end = min(end, chromosome_length)
        # trimmed_interval = Interval(chrom,
        #                             max(start, 0),
        #                             min(end, chromosome_length),
        #                             )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(chrom,
                                          start + 1,
                                          end).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-start, 0)
        pad_downstream = 'N' * max(end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def one_hot_encode(sequence):
    return dna_str_to_one_hot(sequence).astype(np.float32)


def encode_promoter_enhancer_links(gene_enhancer_df, fasta_path = './data/hg38.fa', max_n_enhancer = 60, max_distanceToTSS = 100_000, max_seq_len=2000, add_flanking=False):
    fasta_extractor = FastaStringExtractor(fasta_path)
    gene_pe = gene_enhancer_df.sort_values(by='distance')
    row_0 = gene_pe.iloc[0]
    gene_name = row_0['TargetGene']
    gene_tss = row_0['TargetGeneTSS']
    chrom = row_0['chr']
    if row_0['TargetGeneTSS'] != row_0['TargetGeneTSS']:
        gene_tss = row_0['tss']
        gene_name = row_0['name_gene']
        chrom = row_0['chr']
    # target_interval = kipoiseq.Interval(chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))
    promoter_seq = fasta_extractor.extract(chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))
    promoter_code = one_hot_encode(promoter_seq)
    enhancers_code = np.zeros((max_n_enhancer, max_seq_len, 4))
    enhancer_activity = np.zeros(max_n_enhancer)
    enhancer_distance = np.zeros(max_n_enhancer)
    enhancer_contact = np.zeros(max_n_enhancer)
    # set distance threshold
    gene_pe = gene_pe[(gene_pe['distance'] > max_seq_len/2)&(gene_pe['distance'] <= max_distanceToTSS)]
    e_i = 0
    gene_element_pair = []
    for idx, row in gene_pe.iterrows():
        if row['TargetGene'] != row['TargetGene']:
            break
        if pd.isna(row['start']):
            continue
        if e_i >= max_n_enhancer:
            break
        enhancer_start = int(row['start'])
        enhancer_end = int(row['end'])
        enhancer_center = int((row['start'] + row['end'])/2)
        enhancer_len = enhancer_end - enhancer_start
        # put sequence at the center
        if add_flanking:
            # enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
            enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2)))
        else:
            # enhancers_signal = np.zeros((max_n_enhancer, max_seq_len))
            if enhancer_len > max_seq_len:
                # enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2))
                enhancers_code[e_i][:] = one_hot_encode(fasta_extractor.extract(chrom, enhancer_center-int(max_seq_len/2), enhancer_center+int(max_seq_len/2)))
            else:
                code_start = int(max_seq_len/2)-int(enhancer_len/2)
                # enhancer_target_interval = kipoiseq.Interval(chrom, enhancer_start, enhancer_end)
                enhancers_code[e_i][code_start:code_start+enhancer_len] = one_hot_encode(fasta_extractor.extract(chrom, enhancer_start, enhancer_end))
        # put sequence from the start
        enhancer_activity[e_i] = row['activity_base']
        enhancer_distance[e_i] = row['distance']
        enhancer_contact[e_i] = row['hic_contact']
        gene_element_pair.append([gene_name, row['name']])
        e_i += 1
    # print(promoter_signals.shape, enhancers_signal.shape)
    pe_code = np.concatenate([promoter_code[np.newaxis,:], enhancers_code], axis=0)
    gene_element_pair = pd.DataFrame(gene_element_pair, columns=['gene', 'element'])
    return pe_code, enhancer_activity, enhancer_distance, enhancer_contact, gene_name, gene_element_pair


def prepare_input(gene_enhancer_table, gene_list, cell, num_features=3,
                  root_folder=""):
    # enhancer_gene_k562_100kb[enhancer_gene_k562_100kb['#chr'] == 'chrX']['TargetGene'].unique()
    mRNA_feauture = pd.read_csv(os.path.join(root_folder, 'promo_enhan_inter', 'mRNA_halflife_features.csv'), index_col='gene_id')
    if cell == 'K562':
        promoter_signals = pd.read_csv('./data/K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
        promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    elif cell == 'GM12878':
        promoter_signals = pd.read_csv('./data/GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt', sep='\t', index_col='symbol')
        promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    elif cell == 'H1':
        promoter_signals = pd.read_csv(os.path.join(root_folder, 'promo_enhan_inter', 'H1_ABC/Neighborhoods/GeneList.txt'), sep='\t', index_col='symbol')
        promoter_signals['PromoterActivity'] = np.sqrt(promoter_signals['H3K27ac.RPM.TSS1Kb']*promoter_signals['DHS.RPM.TSS1Kb'])
    else:
        print(cell, 'not found!')
        return 0
    mRNA_feats = ['UTR5LEN_log10zscore',
       'CDSLEN_log10zscore', 'INTRONLEN_log10zscore', 'UTR3LEN_log10zscore',
       'UTR5GC', 'CDSGC', 'UTR3GC', 'ORFEXONDENSITY']
    PE_code_list = []
    PE_feat_list = []
    mRNA_promoter_list = []
    PE_links_list = []
    for gene in tqdm(gene_list):
        gene_df = gene_enhancer_table[gene_enhancer_table['ENSID'] == gene]
        (PE_code, activity_list, distance_list, contact_list, gene_name,
         PE_links) = encode_promoter_enhancer_links(gene_df,
                                                    fasta_path=os.path.join(root_folder, 'hg38/hg38.ml.fa'),
                                                    max_seq_len=2000, max_n_enhancer=60,
                                                    max_distanceToTSS=100_000, add_flanking=False)
        contact_list = np.concatenate([[0], contact_list])
        distance_list = np.concatenate([[0], distance_list/1000])
        activity_list = np.concatenate([[0], activity_list])
        # activity_list = np.log10(0.1+activity_list)
        contact_list = np.log10(1+contact_list)
        if cell == 'H1':
            mRNA_promoter_feat = np.array(
                list(mRNA_feauture.loc[gene, mRNA_feats].values) + [promoter_signals.loc[gene_name, 'PromoterActivity']])
        else:
            mRNA_promoter_feat = np.array(list(mRNA_feauture.loc[gene, mRNA_feats].values) + [promoter_signals.loc[gene, 'PromoterActivity']])
        if num_features == 1:
            PE_feat = distance_list[:,np.newaxis]
            mRNA_promoter_feat = np.array(list(mRNA_feauture.loc[gene, mRNA_feats].values) + [0])
        elif num_features == 2:
            PE_feat = np.concatenate([distance_list[:,np.newaxis], activity_list[:,np.newaxis], ],axis=-1)
        else:
            PE_feat = np.concatenate([distance_list[:,np.newaxis], contact_list[:,np.newaxis], activity_list[:,np.newaxis], ],axis=-1)
        # print(gene_name, PE_code.shape, PE_feat.shape, mRNA_promoter_feat.shape)
        PE_code_list.append(PE_code)
        PE_feat_list.append(PE_feat)
        mRNA_promoter_list.append(mRNA_promoter_feat)
        PE_links_list.append(PE_links)
    PE_links_df = pd.concat(PE_links_list)
    PE_code_list = np.array(PE_code_list)
    PE_feat_list = np.array(PE_feat_list)
    mRNA_promoter_list = np.array(mRNA_promoter_list)
    return PE_code_list, PE_feat_list, mRNA_promoter_list, PE_links_df


def gene_pro_enh_pos(gene_enhancer_table, ensid, max_n_enhancer=60, max_distanceToTSS=100_000, max_seq_len=2000):
    """
    find the promoter and enhancer position information given a gene name
    """
    gene_df = gene_enhancer_table[gene_enhancer_table['ENSID'] == ensid]
    # extract enhancers
    gene_pe = gene_df.sort_values(by='distance')
    row_0 = gene_pe.iloc[0]

    # promoter position
    gene_name = row_0['TargetGene']
    gene_tss = row_0['TargetGeneTSS']
    chrom = row_0['chr']
    if row_0['TargetGeneTSS'] != row_0['TargetGeneTSS']:  # nan
        gene_tss = row_0['tss']
        gene_name = row_0['Gene name']
        chrom = row_0['chr_gene']
    promo_tuple = (chrom, int(gene_tss-max_seq_len/2), int(gene_tss+max_seq_len/2))

    # total 200k region, slice_start can be negative to get correct enhancer position in this 200k region
    slice_start, slice_end = int(gene_tss-max_distanceToTSS), int(gene_tss+max_distanceToTSS)

    # enhancer positions
    enhancer_activity, enhancer_distance, enhancer_contact = [], [], []
    # set distance threshold
    gene_pe = gene_pe[(gene_pe['distance'] > max_seq_len/2) & (gene_pe['distance'] <= max_distanceToTSS)]
    # seq mask
    mask_enh = np.zeros(max_distanceToTSS * 2, dtype=bool)
    mask_enh_pad = np.zeros(max_distanceToTSS * 2, dtype=bool)
    # make promoter position = 1
    start_index = (max_distanceToTSS * 2 - max_seq_len) // 2
    end_index = start_index + max_seq_len
    mask_enh[start_index:end_index] = True
    mask_enh_pad[start_index:end_index] = True

    # enhancer position recording
    enh_pos, enh_pos_pad = [], []
    e_i = 0
    for idx, row in gene_pe.iterrows():
        if row['TargetGene'] != row['TargetGene']:
            break
        if pd.isna(row['start']):
            continue

        # record features
        enhancer_activity.append(row['activity_base'])
        enhancer_distance.append(row['distance'] / 1000)
        enhancer_contact.append(np.log10(1 + row['hic_contact']))
        # original enhancer positions and lengths
        # original number, not limited by max_n_enhancer
        enhancer_start = int(row['start'])
        enhancer_end = int(row['end'])
        enhancer_center = int((row['start'] + row['end'])/2)
        enhancer_len = enhancer_end - enhancer_start

        enh_pos.append([chrom, enhancer_start, enhancer_end])
        # project the enhancers position into current 200k mask sequence
        assert 0 <= enhancer_center - slice_start <= max_distanceToTSS * 2
        mask_enh_start = max(0, enhancer_start - slice_start)
        mask_enh_end = min(enhancer_end - slice_start, max_distanceToTSS * 2)
        mask_enh[mask_enh_start: mask_enh_end] = True

        # pad or truncate the enhancers
        if e_i >= max_n_enhancer:
            continue
        if enhancer_len > max_seq_len:
            pad_enh_start = enhancer_center - int(max_seq_len / 2)
            pad_enh_end = enhancer_center + int(max_seq_len / 2)
        else:
            pad_enh_start, pad_enh_end = enhancer_start, enhancer_end
            # code_start = int(max_seq_len / 2) - int(enhancer_len / 2)
            # enhancers_code[e_i][code_start:code_start + enhancer_len] = one_hot_encode(
            #     fasta_extractor.extract(chrom, enhancer_start, enhancer_end))
        enh_pos_pad.append([chrom, pad_enh_start, pad_enh_end])
        mask_enh_pad_start = max(0, pad_enh_start - slice_start)
        mask_enh_pad_end = min(pad_enh_end - slice_start, max_distanceToTSS * 2)
        mask_enh_pad[mask_enh_pad_start: mask_enh_pad_end] = True

        e_i += 1

    enhancer_activity = np.array(enhancer_activity, dtype=np.float64)
    enhancer_distance = np.array(enhancer_distance, dtype=np.float64)
    enhancer_contact = np.array(enhancer_contact, dtype=np.float64)

    return promo_tuple, (enh_pos, enh_pos_pad), (mask_enh, mask_enh_pad), (enhancer_distance, enhancer_activity, enhancer_contact)


def subseq_to_genome(df_bed, chr_lengths):
    """includelist & blacklist"""
    # hg38 len
    chr_lengths = chr_lengths['hg38']

    chromosome_masks = {}
    for chrm in chr_lengths:
        chromosome_masks[chrm] = np.zeros(chr_lengths[chrm], dtype=bool)

    # for entry in df_bed:
    for index, row in df_bed.iterrows():
        chrm = row['chrom']
        start = row['start']
        end = row['end']
        # chrm, start, end = entry[0], entry[1], entry[2]
        if 'alt' not in chrm:
            assert chrm in chromosome_masks
            chromosome_masks[chrm][start:end] = 1

    return chromosome_masks


def narrow_to_bedgraph(narrow_path, out_path):
    with open(narrow_path, 'r') as infile, open(out_path, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for line in tqdm(infile):
            fields = line.strip().split('\t')
            chr_name = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            writer.writerow([chr_name, start, end, 1])


def save_input_data(seq_one_hot, abc_feats, ensid_list, data_folder=""):
    save_path = os.path.join(data_folder, "H1_promoter_enhancer_encoding.h5")

    with h5py.File(save_path, 'w') as h5_file:
        h5_file.create_dataset('ensid', data=np.array(ensid_list, dtype='S'))
        h5_file.create_dataset('pe_code', data=seq_one_hot)
        h5_file.create_dataset('distance', data=abc_feats[:,:,0])
        h5_file.create_dataset('activity', data=abc_feats[:,:,2])
        h5_file.create_dataset('hic', data=abc_feats[:,:,1])


if __name__ == '__main__':
    # # convert include-list to dict
    # data_folder = ''
    # bed_file = os.path.join(data_folder, 'promoters', "RefSeqCurated.170308.bed.CollapsedGeneBounds.hg38.TSS500bp.bed")
    # columns = ["chrom", "start", "end", "name", "score", "strand"]
    # promo_positions = pd.read_csv(bed_file, sep='\t', names=columns, header=None)
    # with open(os.path.join(data_folder, 'chr_length.json'), 'r') as file:
    #     chr_lengths = json.load(file)
    # mask_dict = subseq_to_genome(promo_positions, chr_lengths)
    # with open(os.path.join(data_folder, 'hg38_includelist_dict.pkl'), "wb") as f:
    #     pickle.dump(mask_dict, f)
    #
    # # convert blacklist to dict
    # with gzip.open(os.path.join(data_folder, "hg38-blacklist.v2.bed.gz"), 'rt') as f:
    #     df_blacklist = pd.read_csv(f, sep='\t', header=None, names=['chrom', 'start', 'end', 'reason'])
    # blacklist_mask_dict = subseq_to_genome(df_blacklist, chr_lengths)
    # with open(os.path.join(data_folder, 'hg38_blacklist_dict_v2.pkl'), "wb") as f:
    #     pickle.dump(blacklist_mask_dict, f)
    #
    # # convert blacklist to dict
    # with gzip.open(os.path.join(data_folder, "hg38-blacklist.bed.gz"), 'rt') as f:
    #     df_blacklist = pd.read_csv(f, sep='\t', header=None, names=['chrom', 'start', 'end', 'reason'])
    # blacklist_mask_dict = subseq_to_genome(df_blacklist, chr_lengths)
    # with open(os.path.join(data_folder, 'hg38_blacklist_dict_v1.pkl'), "wb") as f:
    #     pickle.dump(blacklist_mask_dict, f)

    # convert narrowpeak to bedgraph file
    # data_folder = ''
    # cell_type = 'K562'
    # all_chrs = ['chr' + str(i) for i in list(range(1, 23)) + ['X']]
    # for chrm in all_chrs:
    #     print(f'current chrm {chrm}')
    #     narrow_path = os.path.join(data_folder, 'promo_enhan_inter/DNase-seq', cell_type, 'peaks', f'peaks_{chrm}.macs3_peaks.narrowPeak')
    #     bedgraph_path = os.path.join(data_folder, 'promo_enhan_inter/DNase-seq', cell_type, 'peaks', 'peaks_bedgraph', f'{chrm}.bedGraph')
    #     narrow_to_bedgraph(narrow_path, bedgraph_path)

    # H1 epinformer input data
    EnhancerPredictionsAllPutative_path = ""
    enhancer_gene_H1 = pd.read_csv(EnhancerPredictionsAllPutative_path, sep='\t')  # for H1, here name - symbol, not ensid. GM12878 & K562 make name -> ensid
    # Select the gene-enhancer links within 100kb to the TSS of target gene and remove the promoter element
    enhancer_gene_H1_100kb = enhancer_gene_H1[
        (enhancer_gene_H1['distance'] <= 100_000) & (enhancer_gene_H1['distance'] > 1000)].reset_index()
    # enhancer_gene_k562_100kb.to_csv('./data/K562_enhancer_gene_links_100kb.tsv', index=False, sep='\t')
    # # %%
    # enhancer_gene_k562_100kb = pd.read_csv('./data/K562_enhancer_gene_links_100kb.tsv', sep='\t')
    GeneList_path = ""
    gene_tss = pd.read_csv(
        GeneList_path,
        sep='\t')[['name', 'chr', 'tss', 'strand']]
    split_18377genes_path = ""
    data_split = pd.read_csv(split_18377genes_path)
    gene_tss = gene_tss.merge(data_split, left_on='name', right_on='Gene name').drop(columns='name')
    enhancer_gene_H1_100kb_includeNoEnhancerGene = enhancer_gene_H1_100kb.merge(gene_tss, left_on='TargetGene', right_on='Gene name',
                                                                                # how='right',
                                                                                suffixes=['', '_gene']).reset_index()
    # gene_info = gene_tss[gene_tss['fold_1'] == 'test'][['ENSID', 'Gene name']].head(16).reset_index(drop=True)
    gene_info = enhancer_gene_H1_100kb_includeNoEnhancerGene[['ENSID', 'Gene name']].drop_duplicates().reset_index(drop=True)
    # 11912 rows in total
    gene_list = list(gene_info['ENSID'])
    # encode gene-enhancer links for EPInformer
    # num_feature == 1: distance; num_feature == 2: distance + enhancer activity; num_feature == 3: distance + enhancer activity + hic contacts
    device = 'cpu'
    root_path = ""
    PE_codes, PE_feats, mRNA_feats, PE_pairs = prepare_input(enhancer_gene_H1_100kb_includeNoEnhancerGene,
                                                             gene_list, 'H1', num_features=3, root_folder=root_path)

    save_path = os.path.join(root_path, 'promo_enhan_inter')
    save_input_data(PE_codes, PE_feats, gene_list, data_folder=save_path)

    print('a')
