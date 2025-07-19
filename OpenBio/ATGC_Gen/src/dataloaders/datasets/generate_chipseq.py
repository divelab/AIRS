import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import esm
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
import os

from src.dataloaders.utils.selene_utils import MemmapGenome
from src.tasks.utils import index_mapping


protein_name_uniprot = {
    "RUNX3": "Q13761",
    "IKZF1": "Q13422",
    "BATF": "Q16520",
    "TEAD4": "Q15561",
    "TAL1": "P17542",
    "NFIC": "P08651",
    "HDAC1": "Q13547",
    "HDAC2": "Q92769",
    "ZBTB33": "Q86T24",
    "TCF12": "Q99081",
    "CBX3": "Q13185",
    "FOXA2": "Q9Y261",
    "ARID3A": "Q99856",
    "ATF2": "P15336",
    "BCL11A": "Q9H165",
    "STAT5A": "P42229",
    "RBBP5": "Q15291",
    "IRF4": "Q15306",
    "TRIM28": "Q13263",
    "MEF2A": "Q02078",
    "BCL3": "P20749",
    "POU2F2": "P09086",
    "ZBTB7A": "O95365",
    "PML": "P29590",
    "NFATC1": "O95644",
    "MTA3": "Q9BTC8",
    "GATA3": "P23771",
    "NR2F2": "P24468",
    "UBTF": "P17480",
    "BHLHE40": "O14503",
    "MEF2C": "Q06413",
    "ZNF217": "O75362",
    "TAF7": "Q15545",
    "FOSL1": "P15407",
    "SP1": "P08047",
    "FOSL2": "P15408",
    "HNF4G": "Q14541",
    "GATA2": "P23769",
    "SRF": "P11831",
    "TBP": "P20226",
    "SUZ12": "Q15022",
    "RXRA": "P19793",
    "SAP30": "O75446",
    "SIN3A": "Q96ST3",
    "E2F4": "Q16254",
    "SETDB1": "Q15047",
    "THAP1": "Q9NVV9",
    "ETS1": "P14921",
    "HNF4A": "P41235",
    "STAT3": "P40763",
    "EZH2": "Q15910",
    "STAT1": "P42224",
    "TAF1": "P21675",
    "ELF1": "P32519",
    "CEBPB": "P17676",
    "NANOG": "Q9H9S0",
    "ELK1": "P19419",
    "SIRT6": "Q8N6T7",
    "PHF8": "Q9UPP1",
    "HDAC6": "Q9UBN7",
    "SMC3": "Q9UQE7",
    "USF2": "Q15853",
    "RFX5": "P48382",
    "YY1": "P25490",
    "ZZZ3": "Q8IYH5",
    "CHD1": "O14646",
    "ATF3": "P18847",
    "BRCA1": "P38398",
    "SIX5": "Q8N196",
    "CTCF": "P49711",
    "ZNF274": "Q96GC6",
}


def get_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        fasta_data = response.text
        sequence = "".join(fasta_data.split("\n")[1:])
        return sequence
    else:
        print(f"Error: Unable to fetch sequence for {uniprot_id}")
        return None


class ChIPSeqData(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            tokenizer_name='char',
            split='train',
            max_length=1024,
            protein_len=1000,
            load_prob=False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.protein_len = protein_len
        self.load_prob = load_prob

        if split == 'train':
            self.df = pd.read_csv(f'../../../data/chip_seq/gm12878_500_train_df.csv')
        elif split == 'valid':
            self.df = pd.read_csv(f'../../../data/chip_seq/gm12878_500_val_df.csv')
        elif split == 'test':
            self.df = pd.read_csv(f'../../../data/chip_seq/gm12878_500_test_df.csv')
        else:
            raise NotImplementedError()
        tokens = ["GM12878"]
        self.cell_to_idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx_to_cell = {idx: token for idx, token in enumerate(tokens)}

        # protein embed
        self.protein_embed = torch.load("../../../data/chip_seq/protein_seq_embed.pth")

        self.init_worker()

    def init_worker(self):
        self.genome = MemmapGenome(
            input_path='../../../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
            memmapfile='../../../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap',
            blacklist_regions='hg38'
        )

        self.chr_lens = self.genome.get_chr_lens()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        info_row = self.df.iloc[item]
        chrm = info_row['chrom']
        start = int(info_row['start'])
        end = int(info_row['end'])
        protein_name = info_row['TF']
        cell_type = info_row['Cell_Type']
        cell_type_idx = self.cell_to_idx[cell_type]

        # protein embedding
        protein_embed = self.protein_embed[protein_name]
        protein_embed = F.pad(protein_embed, (0, 0, self.protein_len - protein_embed.shape[0], 0))

        if self.load_prob:
            # use concat token + signal embedding
            seq = self.genome.get_encoding_from_coords(chrm, start, end, "+")  # 1024 * 4
            input_embed = torch.tensor(seq, dtype=torch.float32)

            # sequence tensor - target
            seq_str = self.genome.get_str_seq(chrm, start, end, "+")
            if self.tokenizer_name == 'char':
                seq = self.tokenizer(seq_str, padding="max_length",
                                     max_length=self.max_length, add_special_tokens=False)['input_ids']
            seq = torch.LongTensor(seq)
            seq = torch.LongTensor([index_mapping[int(label)] for label in seq])
            return input_embed, protein_embed, seq
        else:
            seq_str = self.genome.get_str_seq(chrm, start, end, "+")
            if self.tokenizer_name == 'char':
                seq = self.tokenizer(seq_str, padding="max_length",
                                     max_length=self.max_length + 2, add_special_tokens=True)['input_ids']

            seq = torch.LongTensor(seq)

            data = seq[:-1].clone()
            target = seq[1:].clone()

            assert len(data) == self.max_length + 1
            assert len(target) == self.max_length + 1

            return {
                "data": data,
                "target": target,
                "condition": protein_embed,
                "protein_name": protein_name,
                "cell_type": cell_type_idx,
            }
            # return data, target, protein_embed


if __name__ == '__main__':
    root_path = "/data/xsu2/DNA_Gen"
    # binding score filter
    # df = pd.read_csv(f"{root_path}/data/chip_seq/encRegTfbsClusteredWithCells.hg38.bed.gz",
    #                  sep="\t", header=None, names=["chrom", "start", "end", "TF", "score", "cell_line"], compression="gzip")
    # df = df[df["score"] >= 1000]
    # # df.to_csv(f"{root_path}/data/chip_seq/high_bind.csv")
    device = torch.device("cpu")

    df = pd.read_csv(f"{root_path}/data/chip_seq/high_bind.csv", index_col=0)

    """chr filter"""
    # only keep valid chrom
    valid_chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    df = df[df['chrom'].isin(valid_chromosomes)]  # 1056550

    # explode cell type
    df['cell_line'] = df['cell_line'].str.split(',')
    df = df.explode('cell_line').reset_index(drop=True)  # 5752737

    """sei protein overlap"""
    sei_path = f"{root_path}/data/promoter_design/target.sei.names"
    data = []
    with open(sei_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' | ')
            if len(parts) == 3:
                cell_type, protein, identifier = parts
                identifier = identifier.replace('ID:', '')
                data.append([cell_type, protein, identifier])

    df_sei = pd.DataFrame(data, columns=['Cell_Type', 'Protein', 'ID'])
    df_sei = df_sei[df_sei['ID'] == 'ENCODE']
    df_sei = df_sei.drop_duplicates(keep=False)

    # merge
    result_df = pd.merge(df, df_sei, left_on=['TF', 'cell_line'], right_on=['Protein', 'Cell_Type'],
                         how='outer', indicator=True)
    unmerged_rows = result_df[result_df['_merge'] == 'left_only']
    # 4914496
    merged_rows = result_df[result_df['_merge'] == 'both']
    # 838241

    # short TF length, < 1000, 62 proteins
    TF_short_list = ['CTCF', 'TCF12', 'FOSL1', 'MEF2C', 'TAF7', 'EZH2', 'FOXA2', 'RFX5', 'ZZZ3', 'ZBTB7A', 'PML', 'ZNF274', 'HDAC1',
     'CBX3', 'ETS1', 'ZBTB33', 'STAT3', 'TAL1', 'ELK1', 'HNF4G', 'STAT1', 'USF2', 'HNF4A', 'BHLHE40', 'TBP', 'CEBPB',
     'YY1', 'ATF2', 'FOSL2', 'TRIM28', 'SIX5', 'RBBP5', 'NANOG', 'SP1', 'MTA3', 'ARID3A', 'THAP1', 'SIRT6', 'POU2F2',
     'SUZ12', 'STAT5A', 'NFIC', 'IRF4', 'HDAC2', 'IKZF1', 'ELF1', 'NR2F2', 'GATA3', 'MEF2A', 'ATF3', 'SAP30', 'GATA2',
     'SRF', 'BCL11A', 'TEAD4', 'BATF', 'RXRA', 'RUNX3', 'E2F4', 'NFATC1', 'BCL3', 'UBTF']
    merged_rows = merged_rows[merged_rows['TF'].isin(TF_short_list)]

    """protein embedding"""
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    protein_embed = {}
    for protein_name in tqdm(merged_rows['TF'].unique()):
        uniprot_id = protein_name_uniprot[protein_name]
        seq = get_uniprot_sequence(uniprot_id)

        fasta_file = os.path.join(f"{root_path}/data/chip_seq", f'{protein_name}.fasta')
        if os.path.exists(fasta_file):
            pass
        else:
            header = f">{protein_name}"
            with open(fasta_file, 'w') as f:
                f.write(f"{header}\n")
                f.write(f"{seq}\n")

        # esm forward
        esm_data = [("protein1", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(esm_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].cpu())

        assert len(sequence_representations) == 1
        assert sequence_representations[0].shape[0] == len(seq)
        protein_embed[protein_name] = sequence_representations[0]  # length * dim
    torch.save(protein_embed, f"{root_path}/data/chip_seq/protein_seq_embed.pth")

    """cell type"""
    df_gm12878 = merged_rows[merged_rows['Cell_Type'] == "GM12878"]  # 81337

    """length control"""
    df_gm12878 = df_gm12878[(df_gm12878['end'] - df_gm12878['start']) <= 500]  # 55830

    df_train = df_gm12878[~df_gm12878['chrom'].isin(['chr20', 'chr21', 'chr22', 'chrX'])]  # 51800
    df_val = df_gm12878[df_gm12878['chrom'].isin(['chr20', 'chr21'])]  # 2181
    df_test = df_gm12878[df_gm12878['chrom'].isin(['chr22', 'chrX'])]  # 1849

    df_train.to_csv(f"{root_path}/data/chip_seq/gm12878_500_train_df.csv")
    df_val.to_csv(f"{root_path}/data/chip_seq/gm12878_500_val_df.csv")
    df_test.to_csv(f"{root_path}/data/chip_seq/gm12878_500_test_df.csv")

