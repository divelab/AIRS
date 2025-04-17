import torch
from torch.utils.data import Dataset
import numpy as np
import re
import json
import lmdb
from functools import lru_cache
import pickle
import gzip
import os
import random


class NewMol3DDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, conditions=None, conditions_split_id=None, db_path=None):
        self.texts = texts
        self.conditions = conditions  # New addition
        self.conditions_split_id = conditions_split_id  # New addition
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

        if self.env.begin().get(key=int(0).to_bytes(4, byteorder="big")) is None:
            self.idx_offset = 1
        else:
            self.idx_offset = 0
        # import pdb; pdb.set_trace()

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self.texts)
    
    
    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        # if len(self._keys) > 1000: # this is training lmdb
        key = int(idx+self.idx_offset).to_bytes(4, byteorder="big")
        # else:
        #     key = int(idx).to_bytes(4, byteorder="big")

        datapoint_pickled_compressed = self.env.begin().get(key=key)
        protein_embedding_dict = pickle.loads(gzip.decompress(datapoint_pickled_compressed))
        
        protein_embedding_mask = protein_embedding_dict['mask']
        protein_padded_embedding = protein_embedding_dict['padded_embedding']
        
        # import pdb; pdb.set_trace()
        text = self.texts[idx].strip()
        if self.conditions is not None:
            # Concatenate condition and text
            condition = self.conditions[idx].strip()
            full_text = condition + " " + text
        else:
            full_text = text
        if self.conditions_split_id is not None:
            condition_split_id = int(self.conditions_split_id[idx].strip())
        elif self.conditions is not None:
            condition_split_id = len(condition.split())
        else:
            condition_split_id = 0
        encoded_text = self.tokenizer.batch_encode_plus([full_text])
        raw_input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.long).squeeze()
        # if self.conditions is not None:
        #     raw_input_ids = raw_input_ids[1:]  # Remove the first token (<s>)
        input_ids = raw_input_ids[:-1]
        targets = raw_input_ids[1:]
        # import pdb; pdb.set_trace()

        if protein_padded_embedding.requires_grad:
            protein_padded_embedding.requires_grad_(False)
        
        return input_ids, targets, condition_split_id, protein_padded_embedding, protein_embedding_mask


class Mol3DDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, conditions=None, conditions_split_id=None):
        self.texts = texts
        self.conditions = conditions  # New addition
        self.conditions_split_id = conditions_split_id  # New addition
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        text = self.texts[idx].strip()
        if self.conditions is not None:
            # Concatenate condition and text
            condition = self.conditions[idx].strip()
            full_text = condition + " " + text
        else:
            full_text = text
        if self.conditions_split_id is not None:
            condition_split_id = int(self.conditions_split_id[idx].strip())
        elif self.conditions is not None:
            condition_split_id = len(condition.split())
        else:
            condition_split_id = 0
        encoded_text = self.tokenizer.batch_encode_plus([full_text])
        raw_input_ids = torch.tensor(encoded_text["input_ids"], dtype=torch.long).squeeze()
        # if self.conditions is not None:
        #     raw_input_ids = raw_input_ids[1:]  # Remove the first token (<s>)
        input_ids = raw_input_ids[:-1]
        targets = raw_input_ids[1:]
        return input_ids, targets, condition_split_id


class SimpleTokenizer:
    def __init__(self, max_length, support_rag=False):
        if support_rag:
            self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "<sep>": 4}
            self.count = 5
        else:
            self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
            self.count = 4
        self.max_length = max_length
        
    def fit_on_lmdb(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = lmdb.open(
            self.db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
        
            for key in keys:
                datapoint_pickled_compressed = txn.get(key=key)
                retrieved_list = pickle.loads(gzip.decompress(datapoint_pickled_compressed))
                for line in retrieved_list:
                    self.fit_on_text(line.strip())

    def fit_on_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.fit_on_text(line.strip())

    def fit_on_text(self, text):
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.count
                self.count += 1

    def encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence + [self.vocab["</s>"]]
        padding_length = self.max_length - len(sequence)

        if padding_length > 0:
            sequence.extend([self.vocab["<pad>"]] * padding_length)

        return sequence[:self.max_length]

    # def decode(self, token_ids):
    #     reverse_vocab = {v: k for k, v in self.vocab.items()}
    #     return ' '.join(reverse_vocab.get(token_id, "<unk>") for token_id in token_ids if
    #                     token_id not in [self.vocab["<pad>"], self.vocab["<s>"], self.vocab["</s>"]])
    def decode(self, token_ids):
        # --- Remove any characters after the <pad> and </s> ---
        # import pdb; pdb.set_trace()
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        # --- Remove the <s> token ---
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (token_ids == self.vocab["</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"

        # reverse_vocab = {v: k for k, v in self.vocab.items()}
        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        # for token_id in token_ids:
        #     decoded_tokens.append(reverse_vocab.get(token_id, "<unk>"))

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence  # Do not add the ending token for generation
        return sequence

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file)

    def token_decode(self, token_id):
        return self.reverse_vocab.get(token_id, "<unk>")

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            self.vocab = json.load(file)
            self.count = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_decoder_func = np.vectorize(self.token_decode)

    def batch_encode_plus(self, texts):
        encodings = self.encode_batch(texts)
        attention_masks = [[float(token != self.vocab["<pad>"]) for token in encoding] for encoding in encodings]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks
        }


class SubChTokenizer:
    def __init__(self, max_length):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "+": 4, "-": 5, ".": 6, "°": 7}
        self.count = 8  # Start counting after the special tokens
        self.max_length = max_length

    def fit_on_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.fit_on_text(line.strip())

    def fit_on_text(self, text):
        for word in self.split_text(text):
            if word not in self.vocab:
                self.vocab[word] = self.count
                self.count += 1

    def split_text(self, text):
        # Split by space and then further split numbers
        words = []
        for word in text.split():
            # Check if the word is a number or a number followed by °
            if re.match(r'^[+-]?\d+(\.\d+)?°?$', word):
                # Split into tokens
                num_parts = re.findall(r'[+-]|\d+|\.|°', word)
                words.extend(num_parts)
            else:
                words.append(word)
        return words

    def encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in self.split_text(text)]
        sequence = [self.vocab["<s>"]] + sequence + [self.vocab["</s>"]]
        padding_length = self.max_length - len(sequence)

        if padding_length > 0:
            sequence.extend([self.vocab["<pad>"]] * padding_length)

        return sequence[:self.max_length]

    # def decode(self, token_ids):
    #     reverse_vocab = {v: k for k, v in self.vocab.items()}
    #     return ' '.join(reverse_vocab.get(token_id, "<unk>") for token_id in token_ids if
    #                     token_id not in [self.vocab["<pad>"], self.vocab["<s>"], self.vocab["</s>"]])

    def decode(self, token_ids):
        # --- Remove any characters after the <pad> and </s> ---
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        # --- Remove the <s> token ---
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (
                    token_ids == self.vocab[
                "</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"

        # reverse_vocab = {v: k for k, v in self.vocab.items()}
        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        # for token_id in token_ids:
        #     decoded_tokens.append(reverse_vocab.get(token_id, "<unk>"))

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in self.split_text(text)]
        sequence = [self.vocab["<s>"]] + sequence  # Do not add the ending token for generation
        return sequence

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            self.vocab = json.load(file)
            self.count = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_decoder_func = np.vectorize(lambda token_id: self.reverse_vocab.get(token_id, "<unk>"))

    def batch_encode_plus(self, texts):
        encodings = self.encode_batch(texts)
        attention_masks = [[float(token != self.vocab["<pad>"]) for token in encoding] for encoding in encodings]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks
        }


