import glob
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *

# from .rd import ReactionDiffusion
# from .wave import DampedWaveEquation
# from .pendulum import DampledPendulum, IntervenableDampledPendulum
#
#
# import torch
# import math
# from torch.utils.data import DataLoader
#
# def param_rd(buffer_filepath, batch_size=64):
#     dataset_train_params = {
#         'path': buffer_filepath+'_train',
#         'group': 'train',
#         'num_seq': 1600,
#         'size': 32,
#         'time_horizon': 3,
#         'dt': 0.1,
#     }
#
#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params['num_seq'] = 320
#     dataset_test_params['group'] = 'test'
#     dataset_test_params['path'] = buffer_filepath+'_test'
#
#     dataset_train = ReactionDiffusion(**dataset_train_params)
#     dataset_test  = ReactionDiffusion(**dataset_test_params)
#
#     dataloader_train_params = {
#         'dataset'    : dataset_train,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : True,
#     }
#
#     dataloader_test_params = {
#         'dataset'    : dataset_test,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : False,
#     }
#
#     dataloader_train = DataLoader(**dataloader_train_params)
#     dataloader_test  = DataLoader(**dataloader_test_params)
#
#     return dataloader_train, dataloader_test
#
# def param_wave(buffer_filepath, batch_size=64):
#     dataset_train_params = {
#         'path': buffer_filepath+'_train',
#         'dt': 0.001,
#         'size': 64,
#         'seq_len': 25,
#         'dt': 1e-3,
#         'group': 'train',
#         'num_seq': 200,
#         'intervention': 'delta',
#     }
#
#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params['num_seq'] = 50
#     dataset_test_params['group'] = 'test'
#     dataset_test_params['path'] = buffer_filepath+'_test'
#
#     dataset_train = DampedWaveEquation(**dataset_train_params)
#     dataset_test  = DampedWaveEquation(**dataset_test_params)
#
#     dataloader_train_params = {
#         'dataset'    : dataset_train,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : True,
#     }
#
#     dataloader_test_params = {
#         'dataset'    : dataset_test,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : False,
#     }
#     dataloader_train = DataLoader(**dataloader_train_params)
#     dataloader_test  = DataLoader(**dataloader_test_params)
#
#     return dataloader_train, dataloader_test
#
# def param_pendulum(buffer_filepath, batch_size=25):
#     dataset_train_params = {
#         'num_seq': 25,
#         'time_horizon': 20,
#         'dt': 0.5,
#         'group': 'train',
#         'path': buffer_filepath+'_train',
#     }
#
#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params['num_seq'] = 25
#     dataset_test_params['group'] = 'test'
#     dataset_test_params['path'] = buffer_filepath+'_test'
#
#     dataset_train = DampledPendulum(**dataset_train_params)
#     dataset_test  = DampledPendulum(**dataset_test_params)
#
#     dataloader_train_params = {
#         'dataset'    : dataset_train,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : True,
#     }
#
#     dataloader_test_params = {
#         'dataset'    : dataset_test,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : False,
#     }
#     dataloader_train = DataLoader(**dataloader_train_params)
#     dataloader_test  = DataLoader(**dataloader_test_params)
#
#     return dataloader_train, dataloader_test
#
#
# def param_intervenable_pendulum(buffer_filepath, batch_size=25):
#     dataset_train_params = {
#         'num_seq': 25,
#         'time_horizon': 20,
#         'dt': 0.5,
#         'group': 'train',
#         'path': buffer_filepath+'_train',
#         'intervention': 'delta',
#     }
#
#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params['num_seq'] = 25
#     dataset_test_params['group'] = 'test'
#     dataset_test_params['path'] = buffer_filepath+'_test'
#
#     dataset_train = IntervenableDampledPendulum(**dataset_train_params)
#     dataset_test = IntervenableDampledPendulum(**dataset_test_params)
#
#     dataloader_train_params = {
#         'dataset'    : dataset_train,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : True,
#     }
#
#     dataloader_test_params = {
#         'dataset'    : dataset_test,
#         'batch_size' : batch_size,
#         'num_workers': 0,
#         'pin_memory' : True,
#         'drop_last'  : False,
#         'shuffle'    : False,
#     }
#     dataloader_train = DataLoader(**dataloader_train_params)
#     dataloader_test  = DataLoader(**dataloader_test_params)
#
#     return dataloader_train, dataloader_test
#
#
# def init_dataloaders(dataset, buffer_filepath=None):
#     assert buffer_filepath is not None
#     if dataset == 'rd':
#         return param_rd(buffer_filepath)
#     elif dataset == 'wave':
#         return param_wave(buffer_filepath)
#     elif dataset == 'pendulum':
#         return param_pendulum(buffer_filepath)
#     elif dataset == 'intervenable_pendulum':
#         return param_intervenable_pendulum(buffer_filepath)