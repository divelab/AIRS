import os
import pickle
import numpy as np
import torch
from torchvision import transforms as tf
import scipy.io as sio
import torch.nn as nn


def train_valid_test_split(x, y, train_ratio=0.8, valid_ratio=0.1):

    split_train_index = int(x.shape[0] * train_ratio)
    split_valid_index = int(x.shape[0] * valid_ratio) + split_train_index
    x_train = x[:split_train_index]
    y_train = y[:split_train_index]
    x_valid = x[split_train_index:split_valid_index]
    y_valid = y[split_train_index:split_valid_index]
    x_test = x[split_valid_index:]
    y_test = y[split_valid_index:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, training=False, sw=3, batch_couple=False, subset=False, data_frac="1/16"):
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        # self.data = (data / 255.0).float()
        self.training = training
        
        self.data = data
        self.sw = sw
        self.batch_couple = batch_couple
        self.num_seqs = self.data.shape[0]
        if self.training:
            if batch_couple:
                self.num_data_per_seq = self.data.shape[1] - (self.sw + 1 - 1)
            else:
                self.num_data_per_seq = self.data.shape[1] - (self.sw - 1)
        else:
            self.num_data_per_seq = self.data.shape[1] // self.sw
        self.mean = data.mean()
        self.std = data.std()

        if self.training:
            self.transform = tf.Compose([
                    # tf.RandomCrop((32,32), 4, padding_mode='edge'),
                    # tf.RandomHorizontalFlip(),
                    # tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    tf.Normalize((self.mean), (self.std)),
                    # tf.RandomRotation(15),
                    # tf.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.05, 20), value=0, inplace=False),
                    ])
        else:
        #     # self.transform = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            self.transform = tf.Normalize((self.mean), (self.std))

    def __len__(self):

        return self.num_seqs * self.num_data_per_seq

    def __getitem__(self, index):
        # print(index.shape)
        seq_idx = (index // self.num_data_per_seq)
        data_idx = (index % self.num_data_per_seq)

        if self.training:
            if self.batch_couple:
                x = self.data[seq_idx, data_idx:(data_idx+self.sw+1), ...]
            else:
                x = self.data[seq_idx, data_idx:(data_idx+self.sw), ...]
        else:
            x = self.data[seq_idx, (data_idx*self.sw):((data_idx+1)*self.sw), ...]

        # x = self.transform(x) 

        if self.labels is not None:
            if self.training:
                if self.batch_couple:
                    y = self.labels[seq_idx, data_idx:(data_idx+self.sw+1), ...]
                else:
                    y = self.labels[seq_idx, data_idx:(data_idx+self.sw), ...]
            else:
                y = self.labels[seq_idx, (data_idx*self.sw):((data_idx+1)*self.sw), ...]
            return x, y
        else:
            return x