import os
import argparse
from runner import Runner

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='result/', help='The directory for storing training outputs')
parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')

args = parser.parse_args()

result_path = args.result_path
if not os.path.isdir(result_path):
    os.mkdir(result_path)

assert args.dataset in ['perov_5', 'carbon_24', 'mp_20'], "Not supported dataset"

train_data_path = os.path.join('data', args.dataset, 'train.pt')
if not os.path.isfile(train_data_path):
    train_data_path = os.path.join('data', args.dataset, 'train.csv')

val_data_path = os.path.join('data', args.dataset, 'val.pt')
if not os.path.isfile(val_data_path):
    val_data_path = os.path.join('data', args.dataset, 'val.csv')

score_norm_path = os.path.join('data', args.dataset, 'score_norm.txt')

if args.dataset == 'perov_5':
    from config.perov_5_config_dict import conf
elif args.dataset == 'carbon_24':
    from config.carbon_24_config_dict import conf
else:
    from config.mp_20_config_dict import conf

runner = Runner(conf, score_norm_path)
runner.train(train_data_path, val_data_path, result_path)