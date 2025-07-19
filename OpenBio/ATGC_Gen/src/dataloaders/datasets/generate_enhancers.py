import copy
import pickle
import pyfaidx

import torch, random, os, json
import numpy as np

from src.tasks.utils import index_mapping


cur_index_mapping = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T',
}


class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_name,
            tokenizer_name='char',
            split='train',
            max_length=1024,
            load_prob=False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.split = split
        self.max_length = max_length
        self.load_prob = load_prob

        if dataset_name == 'mel_enhancer':
            name = "MEL2"
        elif dataset_name == 'flybrain_enhancer':
            name = 'FlyBrain'
        self.all_data = pickle.load(open(f'../../../data/the_code/General/data/Deep'
                                         f'{name}_data.pkl', 'rb'))
        # self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
        self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(self.all_data[f'y_{split}'])), dim=-1)

        self.num_cls = self.all_data[f'y_{split}'].shape[-1]
        # self.alphabet_size = 4

        self.one_hot_to_str()

    def init_worker(self):
        pass

    def one_hot_to_str(self):
        one_hot = self.all_data[f'{self.split}_data']
        indices = np.argmax(one_hot, axis=-1)
        map_func = np.vectorize(cur_index_mapping.get)
        mapped_array = map_func(indices)
        self.joined_strings = [''.join(row) for row in mapped_array]

    def __len__(self):
        return len(self.joined_strings)

    def __getitem__(self, idx):
        if self.load_prob:
            # bert input data. leading + sequence
            one_hot = self.all_data[f'{self.split}_data'][idx]
            one_hot = torch.tensor(one_hot, dtype=torch.float32)
            condition = self.clss[idx]
            condition_expanded = condition.reshape(1, 1).expand(one_hot.shape[0], 1)
            one_hot = torch.cat([one_hot, condition_expanded], dim=1)

            seq_str = self.joined_strings[idx]
            if self.tokenizer_name == 'char':
                seq = self.tokenizer(seq_str, padding="max_length",
                                     max_length=self.max_length, add_special_tokens=False)['input_ids']
            seq = torch.LongTensor(seq)
            seq = torch.LongTensor([index_mapping[int(label)] for label in seq])
            return one_hot, torch.tensor([]), seq

        seq_str = self.joined_strings[idx]
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq_str, padding="max_length",
                                 max_length=self.max_length, add_special_tokens=True)['input_ids']
        seq = torch.LongTensor(seq)
        data = seq[:-1].clone()
        target = seq[1:].clone()
        condition = self.clss[idx]
        assert len(data) == self.max_length + 1
        assert len(target) == self.max_length + 1
        return {
            "data": data,
            "target": target,
            "condition": condition,
        }
        # return data, target, condition
