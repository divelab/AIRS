import os
import sys
import torch
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data, Batch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import StepLR
from model import ProAuto
from matplotlib import pyplot as plt
from utils import RMSD, KabschRMSD
import re
# from analyze_plot import analyze, get_ground_truth, get_reconstructed, plot_histogram

# global variable initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reg_criterion = torch.nn.L1Loss()
multi_class_criterion = torch.nn.CrossEntropyLoss()
binary_class_criterion = torch.nn.BCELoss()

max_length = 128


def eval(args, model, loader, length=None):
    """
    Evaluate the model
    Args:
        model: model to evaluate
        data: valid_loader or test_loader
    Returns:
        Model error on data
    """
    model.eval()

    total_mse = 0
    total_acc_residue = 0
    total_acc_padding = 0
    total_kl_x = 0
    total_kl_h = 0

    pred_torsion = []
    pred_dist = []
    pred_coord = []
    pred_aa_type = []
    pred_pad_type = []

    true_torsion = []
    true_dist = []
    true_coord = []
    true_aa_type = []
    true_pad_type = []

    count = 0

    rmsd_criterion = RMSD()

    # loop over minibatches
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        count += 1

        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)

        with torch.no_grad():
            pred_coords_ca, pred_residue, pred_pad, kl_x, kl_h = model(batch)


        total_kl_x += kl_x
        total_kl_h += kl_h

        pred_coords_ca_split = torch.split(pred_coords_ca, length)
        protein_mask_split = torch.split(batch.protein_mask, length)
        for pred_coords_ca, protein_mask in zip(pred_coords_ca_split, protein_mask_split):
            pred_coord.append(pred_coords_ca[protein_mask].detach().cpu())
            pred_dist.append((pred_coords_ca[protein_mask][0:-1] - pred_coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt().detach().cpu())
            pred_torsion.append(torsion_angle(pred_coords_ca[protein_mask]))

        pred_aa_type.append(pred_residue[batch.protein_mask].detach().cpu())
        pred_pad_type.append(pred_pad.detach().cpu())

        coords_ca_split = torch.split(batch.coords_ca, length)
        for coords_ca, protein_mask in zip(coords_ca_split, protein_mask_split):
            true_coord.append(coords_ca[protein_mask].detach().cpu())
            true_dist.append((coords_ca[protein_mask][0:-1] - coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt().detach().cpu())
            true_torsion.append(torsion_angle(coords_ca[protein_mask]))

        true_aa_type.append(batch.x[batch.protein_mask].detach().cpu())
        true_pad_type.append(batch.padding.detach().cpu())

    # accuracy for residue type prediction
    preds = torch.argmax(torch.cat(pred_aa_type, dim=0), dim=1)
    acc_residue = torch.sum(preds == torch.cat(true_aa_type, dim=0).squeeze(1)) / preds.shape[0]

    # accuracy for padding type prediction
    preds = (torch.cat(pred_pad_type, dim=0) > 0.5).to(torch.int)
    acc_padding = torch.sum(preds.squeeze(1) == torch.cat(true_pad_type, dim=0)) / preds.shape[0]

    # MAE for atom position reconstruction
    mae = reg_criterion(torch.cat(pred_coord, dim=0), torch.cat(true_coord, dim=0))

    # calculate rmsd, note: this calculation doesn't use alignment, if use krmsd, batch size need to be set to 1
    rmsd = rmsd_criterion(pred_coord, true_coord)

    # MAE for edge distance
    edge_mae = reg_criterion(torch.cat(pred_dist, dim=0), torch.cat(true_dist, dim=0))
    pred_dist = torch.cat(pred_dist, dim=0)
    stable = np.logical_and((pred_dist.cpu().numpy() > 3.65), (pred_dist.cpu().numpy() < 3.95))
    edge_stable = stable.sum() / stable.size

    # MAE for torsion angle
    torsion_mae = reg_criterion(torch.cat(pred_torsion, dim=0), torch.cat(true_torsion, dim=0))

    return torsion_mae, edge_mae, edge_stable, mae, rmsd, acc_residue, acc_padding, total_kl_x / (step + 1), total_kl_h / (step + 1), pred_coord, true_coord, pred_aa_type, true_aa_type, pred_pad_type, true_pad_type

def torsion_angle(coords_ca):

    v1 = coords_ca[1:-2] - coords_ca[0:-3]  # r_ji
    v2 = coords_ca[2:-1] - coords_ca[1:-2]  # r_kj
    v3 = coords_ca[3:] - coords_ca[2:-1]  # r_lk
    v1 = v1 / v1.norm(dim=1, keepdim=True)
    v2 = v2 / v2.norm(dim=1, keepdim=True)
    v3 = v3 / v3.norm(dim=1, keepdim=True)
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)
    a = (n1 * n2).sum(dim=-1)
    b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))

    torsion_angle = torch.nan_to_num(torch.atan2(b, a))

    return torsion_angle

def train(args, model, loader, optimizer, working_dir, loss_term='all', length=None, edgeloss_weight=0.5, kl_weight=0, torsionloss_weight=0):
    """
    Train the model for one epoch
    Args:
        model: model to train
        loader: DataLoader
        optimizer: torch.optim
    Returns:
        Training error
    """

    # set model(s) to training mode and init misc. variables for training
    model.train()
    total_loss = 0
    total_mae = 0
    total_res_loss = 0
    total_pad_loss = 0
    total_edge_mae = 0
    total_torsion_mae = 0

    total_kl_x = 0
    total_kl_h = 0

    # loop over minibatches
    t = tqdm(loader, desc="Iteration")

    fig = plt.figure()

    for step, batch in enumerate(t):
        # count total number of training steps and move the minibatch to device

        batch.coords_ca = batch.coords_ca.double()
        batch = batch.to(device)

        pred_coords_ca, pred_residue, pred_pad, kl_x, kl_h = model(batch)

        assert torch.isnan(pred_coords_ca).sum() == 0
        assert torch.isnan(pred_residue).sum() == 0
        assert torch.isnan(pred_pad).sum() == 0


        # MAE loss
        loss_coords_ca = reg_criterion(pred_coords_ca[batch.protein_mask], batch.coords_ca[batch.protein_mask])
        # cross entropy loss for residue type prediction
        loss_multi_classify = multi_class_criterion(pred_residue[batch.protein_mask], batch.x[batch.protein_mask].squeeze(1).to(torch.long))
        # cross entropy loss for padding type prediction (binary type)
        loss_binary_classify = binary_class_criterion(pred_pad.float(), batch.padding.unsqueeze(1).to(torch.float))

        # edge distance loss
        edge_dist_loss = 0
        pred_coords_ca_split = torch.split(pred_coords_ca, length)
        coords_ca_split = torch.split(batch.coords_ca, length)
        protein_mask_split = torch.split(batch.protein_mask, length)
        count = 0
        for pred_coords_ca, coords_ca, protein_mask in zip(pred_coords_ca_split, coords_ca_split, protein_mask_split):
            count += 1
            pred_dist = (pred_coords_ca[protein_mask][0:-1] - pred_coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt()
            true_dist = (coords_ca[protein_mask][0:-1] - coords_ca[protein_mask][1:]).pow(2).sum(-1).sqrt()
            edge_dist_loss += reg_criterion(pred_dist, true_dist)
        edge_dist_loss = edge_dist_loss / count

        # torsion angle loss
        torsion_loss = 0

        for pred_coords_ca, coords_ca, protein_mask in zip(pred_coords_ca_split, coords_ca_split, protein_mask_split):
            count += 1

            pred_torsion = torsion_angle(pred_coords_ca[protein_mask])
            true_torsion = torsion_angle(coords_ca[protein_mask])

            torsion_loss += reg_criterion(pred_torsion, true_torsion)
        torsion_loss = torsion_loss / count

        loss = loss_coords_ca + loss_multi_classify + loss_binary_classify + 0.1 * kl_x + kl_weight * kl_h + edgeloss_weight * edge_dist_loss + torsionloss_weight * torsion_loss

        # reset accumlated gradient from previous backprop and back prop
        optimizer.zero_grad()

        loss.backward()

        # append description for tqdm progress bar
        t.set_description(f"loss_dist {edge_dist_loss:.3f}, "
                          f"loss_coords {loss_coords_ca:.3f}, "
                          f"loss_res {loss_multi_classify:.3f}, "
                          f"loss_pad {loss_binary_classify:.3f}, ")


        optimizer.step()

        total_loss += loss.detach().cpu()
        total_mae += loss_coords_ca.detach().cpu()
        total_pad_loss += loss_binary_classify.detach().cpu()
        total_res_loss += loss_multi_classify.detach().cpu()
        total_edge_mae += edge_dist_loss.detach().cpu()
        total_torsion_mae += torsion_loss.detach().cpu()

        if type(kl_x) != int:
            total_kl_x += kl_x.detach().cpu()
        if type(kl_h) != int:
            total_kl_h += kl_h.detach().cpu()

    # return the mean loss across all minibatches
    return total_loss / (step + 1), total_mae / (step + 1), total_res_loss / (step + 1), total_pad_loss / (step + 1), \
           total_edge_mae / (step + 1), total_torsion_mae / (step + 1), total_kl_x / (step + 1), total_kl_h / (step + 1)

    
def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Protein generation")

    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

    # model hyperparameters
    parser.add_argument("--mp_steps", type=int, default=4, help="number of steps of message passing for equivariant network")
    parser.add_argument("--emb_dim", type=int, default=32, help="dimensionality of hidden layers in GNN")
    parser.add_argument("--layers", type=int, default=3, help="number of layers in encoder and decoder")
    parser.add_argument('--pooling', type=str, default='True', help='pooling or not')
    parser.add_argument('--up_mlp', action='store_true', default=False, help='mlp after copy latent embedding')
    parser.add_argument('--residual', type=str, default='True', help='residual connection or not')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to input position')
    parser.add_argument('--transpose', action='store_true', default=False, help='decoder in transpose conv way')
    parser.add_argument('--attn', action='store_true', default=False, help='')
    parser.add_argument('--loss', type=str, default='all', help='')
    parser.add_argument('--output_res_dim', type=int, default=20, help='how many kinds of residue')


    # training
    parser.add_argument("--mode", type=str, default="train", help="")
    parser.add_argument("--lr_init", type=float, default = 1e-3, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size for training")
    parser.add_argument("--edgeloss_weight", type=float, default=0.5, help="weight for edge distance loss")
    parser.add_argument("--kl_weight", type=float, default=0, help="weight for kl divergence loss")
    parser.add_argument("--torsionloss_weight", type=float, default=0.5, help="weight for torsion angle loss")

    # data
    parser.add_argument("--dataname", type=str, default="EC_data_256", help="data")
    parser.add_argument("--num_workers", type=int, default=0, help="num of data loader workers")
    parser.add_argument("--max_length", type=int, default=128, help="max amino acid sequence length")
    parser.add_argument("--data_path", type=str, default="../data/", help="path to data")

    # directory
    parser.add_argument("--working_dir", type=str, default="../", help="working directory for logs, saved models, etc.")
    parser.add_argument("--suffix", type=str, default="", help="optional suffix added to working_dir")
    parser.add_argument("--log_dir", type=str, default="../log", help="tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="trained_models", help="directory to save checkpoint in working directory")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="directory with checkpoint.pt")

    # parse args and display; get time
    args = parser.parse_args()

    args.pooling = True if (args.pooling == 'True') else False
    args.residual = True if (args.residual == 'True') else False


    print(args)
    cur_time = time.strftime("%Y%m%d_%H%M")

    # create the working dir
    if args.debug:
        args.working_dir = os.path.join(args.working_dir, args.suffix+"_debug")
    else:
        args.working_dir = os.path.join(args.working_dir, cur_time + args.suffix)
    if args.mode == 'train':
        os.makedirs(args.working_dir, exist_ok=True)

        # write arguments to txt
        with open(os.path.join(args.working_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
    
    # set seeds and init dictionaries for model definition
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    global max_length
    max_length = args.max_length

    params = {
        "mp_steps": args.mp_steps,
        "layers": args.layers,
        "num_types": 27,
        "type_dim": 32,
        "hidden_dim": args.emb_dim,
        "out_node_dim": 32,
        "in_edge_dim":32,
        "output_pad_dim": 1,
        "output_res_dim": args.output_res_dim,
        "pooling": args.pooling,
        "up_mlp": args.up_mlp,
        "residual": args.residual,
        "noise": args.noise,
        "transpose": args.transpose,
        "attn": args.attn,
    }

    # load data
    if args.dataname == "AFPDB_data_128_complete":

        train_set = torch.load(os.path.join(args.data_path, 'AFPDB_data_128_Train_complete.pt')) if args.mode == 'train' else None
        valid_set = torch.load(os.path.join(args.data_path, 'PDB_data_128_Val_complete.pt'))
    else:
        ValueError("Invalid dataname!")


    length = int(re.findall(r"([0-9]+)", args.dataname)[0])

    # initialize data loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if args.mode == 'train' else None
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # init model; display the number of parameters
    model = ProAuto(**params).double()

    print(f"Training with {torch.cuda.device_count()} GPUs!")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n# Params: {num_params}")

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=2e-4)

    # initialize scheduler
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    if args.log_dir != '' and args.mode == 'train':
        log_dir = os.path.join(args.log_dir, cur_time + args.suffix)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    best_valid_rmsd = 1000
    best_res_acc = 0
    best_edge_stable = 0
    best_valid_rmsd_epoch = 0
    best_res_acc_epoch = 0
    best_edge_stable_epoch = 0

    if args.mode == 'train':

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            print("=====Epoch {}".format(epoch))
            print('Training...')
            total_loss, train_mae, res_loss, pad_loss, train_edge_mae, train_torsion_mae, train_kl_x, train_kl_h = train(args, model,
                                                                                                                      train_loader,
                                                                                                                      optimizer,
                                                                                                                      args.working_dir,
                                                                                                                      args.loss,
                                                                                                                      length,
                                                                                                                      edgeloss_weight=args.edgeloss_weight,
                                                                                                                      kl_weight=args.kl_weight,
                                                                                                                      torsionloss_weight=args.torsionloss_weight)

            print('Evaluating...')
            valid_torsion_mae, valid_edge_mae, edge_stable, valid_mae, rmsd, res_acc, pad_acc, kl_x, kl_h, pred_coord, true_coord, pred_aa_type, true_aa_type, pred_pad_type, true_pad_type = eval(args, model, valid_loader, length)


            print("Epoch {:d}, valid_edge_mae: {:.5f}, edge_stable: {:.5f}, Train_mae: {:.5f}, Validation_mae: {:.5f}, Validation_rmsd: {:.5f}, res_acc: {:.2f}, pad_acc: {:.2f}, kl_x: {:.2f}, kl_h: {:.2f}, elapse: {:.5f}".
                  format(epoch, valid_edge_mae, edge_stable, train_mae, valid_mae, rmsd, res_acc, pad_acc, kl_x, kl_h, time.time() - start))

            if args.log_dir != '':
                writer.add_scalar('valid/torsion_mae', valid_torsion_mae, epoch)
                writer.add_scalar('valid/edge_mae', valid_edge_mae, epoch)
                writer.add_scalar('valid/edge_stable', edge_stable, epoch)
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('valid/acc_res', res_acc, epoch)
                writer.add_scalar('valid/acc_pad', pad_acc, epoch)
                writer.add_scalar('valid/rmsd', rmsd, epoch)
                writer.add_scalar('valid/kl_x', kl_x, epoch)
                writer.add_scalar('valid/kl_h', kl_h, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)
                writer.add_scalar('train/loss_res', res_loss, epoch)
                writer.add_scalar('train/loss_pad', pad_loss, epoch)
                writer.add_scalar('train/kl_x', train_kl_x, epoch)
                writer.add_scalar('train/kl_h', train_kl_h, epoch)
                writer.add_scalar('train/edge_mae', train_edge_mae, epoch)
                writer.add_scalar('train/torsion_mae', train_torsion_mae, epoch)

            if rmsd < best_valid_rmsd:
                best_valid_rmsd = rmsd
                best_valid_rmsd_epoch = epoch
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                                  'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
                    checkpoint_dir = os.path.join(args.working_dir, args.checkpoint_dir)
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_rmsd.pt'))

                print('Analyzing...')
                savedir = os.path.join(args.working_dir, 'reconstruction')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

            if res_acc > best_res_acc:
                best_res_acc = res_acc
                best_res_acc_epoch = epoch
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                                  'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
                    checkpoint_dir = os.path.join(args.working_dir, args.checkpoint_dir)
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_rec_acc.pt'))

            if edge_stable > best_edge_stable:
                best_edge_stable = edge_stable
                best_edge_stable_epoch = epoch
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                                  'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc, 'pad_acc': pad_acc, 'edge_stable': edge_stable, 'torsion_mae': valid_torsion_mae}
                    checkpoint_dir = os.path.join(args.working_dir, args.checkpoint_dir)
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_edge_stable.pt'))


            scheduler.step()
            print(f'Best validation RMSD so far: {best_valid_rmsd}, epoch: {best_valid_rmsd_epoch}')
            print(f'Best validation rec acc so far: {best_res_acc}, epoch: {best_res_acc_epoch}')
            print(f'Best validation edge stable so far: {best_edge_stable}, epoch: {best_edge_stable_epoch}')

    elif args.mode == 'valid':
        print("Loading checkpoint ...")
        checkpoint = torch.load(os.path.join(args.saved_model_dir, 'checkpoint_bst_rmsd.pt'))
        print("Loading successfully, epoch: ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.double()
        print('Evaluating on validation dataset...')
        valid_torsion_mae, valid_edge_mae, edge_stable, valid_mae, rmsd, res_acc, pad_acc, kl_x, kl_h, pred_coord, true_coord, pred_aa_type, true_aa_type, pred_pad_type, true_pad_type = eval(args, model, valid_loader, length)
        print("Validation_torsion_mae: {:.5f}, Validation_edge_stable: {:.5f}, Validation_rmsd: {:.5f}, res_acc: {:.2f}, pad_acc: {:.2f}".
            format(valid_torsion_mae, edge_stable, rmsd, res_acc, pad_acc))
        

if __name__ == "__main__":
    main()