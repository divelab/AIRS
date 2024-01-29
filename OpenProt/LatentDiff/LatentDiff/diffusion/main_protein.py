# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_dataset
from configs.datasets_config import pdb_protein
import copy
import utils
import argparse
import wandb
from os.path import join
from protein.models import get_optim, get_model
from equivariant_diffusion import en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle

import train_test


parser = argparse.ArgumentParser(description='latent_diffusion')
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--model', type=str, default='egnn_dynamics', help='')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')


parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='', help='dataset name')
parser.add_argument('--n_report_steps', type=int, default=10)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='')
parser.add_argument("--conditioning", nargs='+', default=[], help='') # not used in latentdiff
parser.add_argument('--resume', type=str, default=None, help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--ema_decay', type=float, default=0,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10], help='')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not') # used in EDM, always set to False for latentdiff
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')
parser.add_argument('--include_index', type=eval, default=True, help='include sequence index or not')

parser.add_argument('--latent_dataname', type=str, default='', help='dataset name for latent diffusion')
parser.add_argument('--change_scale', type=eval, default=False, help='change scale of coordinate or not')

args = parser.parse_args()
#

data_file = f'../data/{args.latent_dataname}_train.pt'
val_file = f'../data/{args.latent_dataname}_val.pt'
test_file = f'../data/{args.latent_dataname}_test.pt'

def load_data(train_file, val_file, test_file):
    def convert(file):
        datalist = []
        dataset = torch.load(file)
        for i in range(len(dataset)):
            if args.change_scale:
                data = torch.cat([dataset[i].coords / 10, dataset[i].h], dim=1)
            else:
                data = torch.cat([dataset[i].coords, dataset[i].h], dim=1)
            datalist.append(data)
        return datalist

    train_data = convert(train_file)
    val_data = convert(val_file)
    test_data = convert(test_file)

    return train_data, val_data, test_data

split_data = load_data(data_file, val_file, test_file)


if args.remove_h:
    raise NotImplementedError()
else:
    dataset_info = pdb_protein

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

transform = build_dataset.ProteinLatentTransform(dataset_info, args.include_charges, args.include_index, device, args.sequential, group_size=1)

dataloaders = {}
for key, data_list in zip(['train', 'val', 'test'], split_data):
    dataset = build_dataset.ProteinDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_dataset.ProteinDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle)
del split_data


# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

if args.resume is not None:
    exp_name = args.exp_name  + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    print(args)

utils.create_folders(args)
print(args)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'latent_diffusion_protein', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

data_dummy = next(iter(dataloaders['train']))


context_node_nf = 0
property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow

model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader_train=dataloaders['train'], hidden_dim=32)
model = model.to(device)
optim = get_optim(args, model)
# print(model)


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def main():
    if args.resume is not None:

        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))

        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_test.train_epoch(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                               property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                               prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0 and not (epoch == 0):
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                      property_norms, nodes_dist, partition='Val')
            nll_test = train_test.test(args, dataloaders['test'], epoch, model_ema_dp, device, dtype,
                                       property_norms, nodes_dist, partition='Test')

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, '../diffusion/outputs/%s/optim.npy' % args.exp_name)
                    checkpt_best = {'model': model.state_dict(), 'epoch': epoch}
                    torch.save(checkpt_best, f'../diffusion/outputs/{args.exp_name}/best_checkpoint.pt')
                    utils.save_model(model, f'../diffusion/outputs/{args.exp_name}/generative_model.npy')
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, '../diffusion/outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('../diffusion/outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, '../diffusion/outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, '../diffusion/outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))

                utils.save_model(optim, '../diffusion/outputs/%s/optim_last.npy' % (args.exp_name))
                utils.save_model(model, '../diffusion/outputs/%s/generative_model_last.npy' % (args.exp_name))

                if args.ema_decay > 0:
                    utils.save_model(model_ema, '../diffusion/outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('../diffusion/outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
                    
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
