import os
import argparse
import torch
from runner import Runner
from dataset import MatDataset
from torch_geometric.data import DataLoader
from utils import smact_validity, compute_elem_type_num_wdist, get_structure, compute_density_wdist, structure_validity

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='The directory for storing training outputs')
parser.add_argument('--dataset', type=str, default='perov_5', help='Dataset name, must be perov_5, carbon_24, or mp_20')
parser.add_argument('--num_gen', type=int, default=100, help='Number of materials to generate')

args = parser.parse_args()

assert args.dataset in ['perov_5', 'carbon_24', 'mp_20'], "Not supported dataset"

train_data_path = os.path.join('data', args.dataset, 'train.pt')
if not os.path.isfile(train_data_path):
    train_data_path = os.path.join('data', args.dataset, 'train.csv')

test_data_path = os.path.join('data', args.dataset, 'test.pt')
if not os.path.isfile(test_data_path):
    train_data_path = os.path.join('data', args.dataset, 'test.csv')

score_norm_path = os.path.join('data', args.dataset, 'score_norm.txt')

if args.dataset == 'perov_5':
    from config import perov_5_conf as conf
elif args.dataset == 'carbon_24':
    from config import carbon_24_conf as conf
else:
    from config import mp_20_config_dict as conf

dataset = MatDataset(test_data_path, prop_name=conf['data']['prop_name'])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

gt_atom_types_list, gt_lengths_list, gt_angles_list, gt_frac_coords_list = [], [], [], []
for iter_num, data_batch in enumerate(loader):
    atom_types, lengths, angles, frac_coords = data_batch.atom_types.numpy(), data_batch.lengths.numpy().reshape(-1), \
        data_batch.angles.numpy().reshape(-1), data_batch.frac_coords.numpy()
    gt_atom_types_list.append(atom_types)
    gt_lengths_list.append(lengths)
    gt_angles_list.append(angles)
    gt_frac_coords_list.append(frac_coords)
gt_structure_list = get_structure(gt_atom_types_list, gt_lengths_list, gt_angles_list, gt_frac_coords_list)

runner = Runner(conf, score_norm_path)
runner.model.load_state_dict(torch.load(args.model_path))

gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list = runner.generate(args.num_gen, train_data_path)

is_valid, validity = smact_validity(gen_atom_types_list)
print("composition validity: {}".format(validity))

elem_type_num_wdist = compute_elem_type_num_wdist(gen_atom_types_list, gt_atom_types_list)
print("element EMD: {}".format(elem_type_num_wdist))

gen_structure_list = get_structure(gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list)

is_valid, structure_validity = structure_validity(gen_atom_types_list, gen_lengths_list, gen_angles_list, gen_frac_coords_list, gen_structure_list)
print("structure validity: {}".format(structure_validity))

density_wdist = compute_density_wdist(gen_structure_list, gt_structure_list)
print("density EMD: {}".format(density_wdist))