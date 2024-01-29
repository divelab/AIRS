# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from protein.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from os.path import join
import os
from protein.sampling import sample
import protein.losses as losses
from configs.datasets_config import pdb_protein


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def generate_and_save_protein(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_pt=False, change_scale=False, lambda_0=2, psi=2, suffix=None):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'h': [], 'x': [], 'node_mask': []}
    h_list = []
    x_list = []
    start_time = time.time()
    for i in range(int(n_samples/batch_size)):
        # nodesxsample = nodes_dist.sample(batch_size)
        nodesxsample = torch.ones((batch_size, ), dtype=torch.int32) * 32
        h, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample, lambda_0=lambda_0, psi=psi)

        molecules['h'].append(h.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        h_list.append(h)
        x_list.append(x)

    print(f"Total time for generating {n_samples} samples: {time.time() - start_time}")


    h = torch.cat(h_list, dim=0)
    x = torch.cat(x_list, dim=0)
    if change_scale:
        x = x * 10
    if save_to_pt:
        save_dir = join(eval_args.model_path + '_eval', 'sampled_molecules', suffix)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({'h': h, 'x': x}, join(save_dir, 'latent_data.pt'))

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    parser.add_argument('--save_to_pt', type=eval, default=False,
                        help='Should save samples to pt files.')
    parser.add_argument('--change_scale', type=eval, default=False,
                        help='')
    parser.add_argument('--eval_epoch', type=int, default=-1,
                        help='epoch for evaluation')
    parser.add_argument('--lambda_0', type=float, default=2, help='parameter for low temperature sampling')
    parser.add_argument('--psi', type=int, default=2, help='parameter for low temperature sampling')
    parser.add_argument('--suffix', type=str, default="",
                        help='Specify model path')
    parser.add_argument('--diffusion_steps', type=int, default=1000)

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.diffusion_steps = eval_args.diffusion_steps

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    # utils.create_folders(args)
    print(args)

    
    dataloaders = None

    # Load model
    dataset_info = pdb_protein
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataloaders, None, hidden_dim=32)

    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    if eval_args.eval_epoch > 0 and not (args.ema_decay > 0):
        fn = f'generative_model_{eval_args.eval_epoch}.npy'

    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    num_params = sum(p.numel() for p in generative_model.parameters())
    print(f"\n# Params: {num_params}")

    generate_and_save_protein(
        args, eval_args, device, generative_model, nodes_dist,
        None, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_pt=eval_args.save_to_pt, change_scale=eval_args.change_scale,
        lambda_0=eval_args.lambda_0, psi=eval_args.psi, suffix=eval_args.suffix)


if __name__ == "__main__":
    main()
