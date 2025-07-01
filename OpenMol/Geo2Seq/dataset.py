import torch
from torch.utils.data import Dataset
import numpy as np
import re
import json


class Mol3DDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, conditions=None, conditions_split_id=None):
        self.texts = texts
        self.conditions = conditions
        self.conditions_split_id = conditions_split_id
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].strip()
        if self.conditions is not None:
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
        input_ids = raw_input_ids[:-1]
        targets = raw_input_ids[1:]
        return input_ids, targets, condition_split_id


class SimpleTokenizer:
    def __init__(self, max_length):
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.count = 4
        self.max_length = max_length

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

    def decode(self, token_ids):
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (token_ids == self.vocab["</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"
        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]
        sequence = [self.vocab["<s>"]] + sequence
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
        words = []
        for word in text.split():
            if re.match(r'^[+-]?\d+(\.\d+)?°?$', word):
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


    def decode(self, token_ids):
        end_ids = torch.nonzero((token_ids == self.vocab["<pad>"]) | (token_ids == self.vocab["</s>"]))
        end = end_ids.min() if len(end_ids) > 0 else len(token_ids)
        token_ids = token_ids[:end]
        token_ids = token_ids[token_ids != self.vocab["<s>"]]
        assert (token_ids == self.vocab["<pad>"]).sum() + (token_ids == self.vocab["<s>"]).sum() + (
                    token_ids == self.vocab[
                "</s>"]).sum() == 0, "There are still <s>, <pad>, or </s> tokens in the decoded sequence"

        decoded_tokens = self.token_decoder_func(token_ids.cpu())

        return ' '.join(decoded_tokens)

    def generation_encode(self, text):
        sequence = [self.vocab.get(word, self.vocab["<unk>"]) for word in self.split_text(text)]
        sequence = [self.vocab["<s>"]] + sequence
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


