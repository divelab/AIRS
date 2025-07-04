import jsonargparse
from torch.utils.data import Dataset
from CEL.utils.register import register
from CEL.data.datasets import *
from torch.utils.data import DataLoader
from typing import List, Dict
from CEL.data.datasets.meta_dataset import DatasetRegistry
# from CEL.data.datasets.pendulum import DampledPendulum, IntervenableDampledPendulum, Dataset
# from CEL.data.datasets.series_dataset import LotkaVolterraDataset
import os
def load_dataset(ds_config):
    dataset_name = ds_config.name
    os.makedirs(ds_config.dataset_root / dataset_name, exist_ok=True)
    train_dataset = register.datasets[dataset_name](ds_config.dataset_root / dataset_name / 'train', group='train', **ds_config[dataset_name])
    test_dataset = register.datasets[dataset_name](ds_config.dataset_root / dataset_name/ 'test', group='test', **ds_config[dataset_name])
    return {'train': train_dataset, 'test': test_dataset}

def load_dataloader(dataset, ds_config):
    dataloader = {'train': DataLoader(dataset['train'], batch_size=ds_config.batch_size['train'], shuffle=True, num_workers=ds_config.num_workers, pin_memory=True, drop_last=False),
                  'test': DataLoader(dataset['test'], batch_size=ds_config.batch_size['test'], shuffle=False, num_workers=ds_config.num_workers, pin_memory=True, drop_last=False)}
    return dataloader

class MyDataset:
    def __init__(self, train: Dataset, test: Dataset):
        self.train = train
        self.test = test

class MyDataLoader:

    def __init__(self):
        super().__init__()


class AphynityDataLoader(MyDataLoader):
    def __init__(self, dataset: MyDataset, batch_size: dict, num_workers: int, pin_memory: bool, drop_last: bool, shuffle: dict):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.train = DataLoader(self.dataset.train, batch_size=self.batch_size['train'], num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, shuffle=self.shuffle['train'])
        self.test = DataLoader(self.dataset.test, batch_size=self.batch_size['test'], num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, shuffle=self.shuffle['test'])

class SeriesDataLoader(MyDataLoader):
    def __init__(self, dataset: Dataset, i_fold: int, batch_size: dict, num_workers: int, pin_memory: bool, drop_last: bool):
        super().__init__()
        self.dataset = dataset[i_fold]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train = DataLoader(self.dataset[0], batch_size=self.batch_size['train'], num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, shuffle=True)
        self.id_test = DataLoader(self.dataset[1], batch_size=self.batch_size['test'], num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, shuffle=False)
        self.ood_test = DataLoader(self.dataset[2], batch_size=self.batch_size['test'], num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, shuffle=False)

from dataclasses import dataclass
from jsonargparse import set_docstring_parse_options
from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse import class_from_function


class Calendar:
    def __init__(self, firstweekday: int):
        self.firstweekday = firstweekday

    @classmethod
    def from_str(cls, firstweekday:str):
        if firstweekday == "Monday":
            return cls(1)

CalendarFromStr = class_from_function(Calendar.from_str, Calendar, name="CalendarFromStr")
class TestDataset:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset


if __name__ == '__main__':
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_class_arguments(TestDataset, 'test')
    cfg = parser.parse_path('../../configs/test.yaml')
    print(cfg.as_dict())
    cfg = parser.instantiate_classes(cfg)
    print(cfg.as_dict())
    # loader = cfg.loader
    # print(list(loader.train))
