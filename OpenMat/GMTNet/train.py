import argparse
import torch
from torch import nn
import numpy as np
from data import get_dataset
import pandas as pd
import pickle as pk
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from e3nn.io import CartesianTensor
from pandarallel import pandarallel
from data import get_symmetry_dataset
pandarallel.initialize(progress_bar=False)

from graphs import atoms2graphs, atoms2graphs_etgnn, GraphDataset
from utils import get_id_train_val_test
from gmtnet import GMTNet
from megnet import MEGNET
from mace_models import MACE
from ecomformer import EComformerEquivariant
from etgnn import DimeNetPlusPlusWrap
import matplotlib.pyplot as plt
from e3nn import o3
import pdb
# torch config
torch.set_default_dtype(torch.float32)
import torch
import numpy as np
import random
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Set the random seed for Python, NumPy, and PyTorch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Ensuring CUDA's determinism
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
    # Configure PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

adptor = JarvisAtomsAdaptor()

diagonal = [0, 4, 8]
off_diagonal = [1, 2, 3, 5, 6, 7]
converter = CartesianTensor("ij")
irreps_output = o3.Irreps('1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o')

def structure_to_graphs(
    df: pd.DataFrame,
    use_corrected_structure: bool = False,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(p_input):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(p_input["structure"])
        return atoms2graphs(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            reduce=reduce_cell,
            equivalent_atoms=p_input['equivalent_atoms'],
            use_canonize=True,
        )
    graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    # graphs = df["p_input"].apply(atoms_to_graph).values
    return graphs

def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total size: {total_params}")
    print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_params

# def structure_to_graphs( # etgnn and mace model
#     df: pd.DataFrame,
#     use_corrected_structure: bool = False,
#     reduce_cell: bool = False,
#     cutoff: float = 5.0, # 6.0 for etgnn 5.0 for MACE
#     max_neighbors: int = 16
# ):
#     def atoms_to_graph(p_input):
#         """Convert structure dict to DGLGraph."""
#         structure = adptor.get_atoms(p_input["structure"])
#         return atoms2graphs_etgnn(
#             structure,
#             cutoff=cutoff,
#         )
#     graphs = df["p_input"].parallel_apply(atoms_to_graph).values
#     # graphs = df["p_input"].apply(atoms_to_graph).values
#     return graphs

class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def get_pyg_dataset(data, target, reduce_cell=False):
    df_dataset = pd.DataFrame(data)
    g_dataset = structure_to_graphs(df_dataset, reduce_cell=reduce_cell)
    pyg_dataset = GraphDataset(df=df_dataset,graphs=g_dataset, target=target)
    return pyg_dataset

def train(model, args):
    # load the dataset
    if args.load_preprocessed:
        print("load preprocessed dataset ...")
    dataset_sym = get_dataset(dataset_name=args.target,use_corrected_structure=args.use_corrected_structure,load_preprocessed=args.load_preprocessed)
    # pdb.set_trace()
    # preprocess the dataset and random split
    id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dataset_sym),
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            keep_data_order=False,
        )
    dataset_train = [dataset_sym[x] for x in id_train]
    dataset_val = [dataset_sym[x] for x in id_val]
    dataset_test = [dataset_sym[x] for x in id_test]
    
    pyg_dataset_train = get_pyg_dataset(dataset_train, args.target, args.reduce_cell)
    pyg_dataset_val = get_pyg_dataset(dataset_val, args.target, args.reduce_cell)
    pyg_dataset_test = get_pyg_dataset(dataset_test, args.target, args.reduce_cell)

    # form dataloaders
    collate_fn = pyg_dataset_train.collate
    train_loader = DataLoader(
        pyg_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        pyg_dataset_val,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        pyg_dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    count_parameters(model)
    # set up training configs
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_iter = steps_per_epoch * args.epochs
    scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=args.learning_rate, end_lr=0.00001, power=1)
    from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "huber": nn.HuberLoss(),
    }
    criterion = criteria[args.loss]
    MAE = nn.L1Loss()
 
    # training epoch
    wandb.login()
    wandb.init(project="crys")
    best_score = 10000
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit='batch') as pbar:
            for data in train_loader:
                structure, mask, equality, labels, rot_list = data
                structure, mask, equality, labels = structure.to(device), mask.to(device), equality.to(device), labels.to(device)
                optimizer.zero_grad()

                if args.model == "gmtnet":
                    outputs = model(structure, mask, equality)
                    loss = criterion(outputs, labels)
                elif args.model == "megnet":
                    outputs = model(structure).view(-1, 3, 3)
                    # ablation for frame average
                    out_list = []
                    for bi in range(len(rot_list)):
                        out = outputs[bi]
                        R = rot_list[bi].to(device)
                        RT = R.transpose(1, 2)
                        out = out.repeat(R.shape[0], 1, 1)
                        RM = torch.matmul(R, out)
                        res = torch.matmul(RM, RT).mean(dim=0)
                        out_list.append(res)
                    loss = criterion(torch.stack(out_list), labels)
                elif args.model == "mace" or args.model == "ecomformer":
                    outputs = model(structure).view(-1, 3, 3)
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(structure)
                    # ablation for frame average
                    out_list = []
                    for bi in range(len(rot_list)):
                        out = outputs[bi]
                        R = rot_list[bi].to(device)
                        RT = R.transpose(1, 2)
                        out = out.repeat(R.shape[0], 1, 1)
                        RM = torch.matmul(R, out)
                        res = torch.matmul(RM, RT).mean(dim=0)
                        out_list.append(res)
                    loss = criterion(torch.stack(out_list), labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'training_loss': running_loss / (pbar.n + 1)})
                pbar.update(1)
                scheduler.step()

        average_train_loss = running_loss / len(train_loader)
        wandb.log({"Train Loss": average_train_loss})

        # Validation
        model.eval()
        running_loss = 0.0
        label_list = []
        output_list = []
        
        for data in val_loader:
            structure, mask, _, labels, rot_list = data
            structure, mask, labels = structure.to(device), mask.to(device), labels.to(device)
            if args.model == "gmtnet":
                outputs = model(structure, mask, _).detach()
            else:
                outputs = model(structure).detach()
                if outputs.shape[-1] > 3:
                    outputs = outputs.view(-1, 3, 3)

            output_list.append(outputs.reshape(-1, 9))

            label_list.append(labels.reshape(-1, 9))

        
        outputs = torch.stack(output_list).reshape(-1, 9)
        labels = torch.stack(label_list).reshape(-1, 9)
        mae = abs(outputs - labels).mean(dim=-1).mean()
        
        if mae < best_score and epoch > 100:
            best_score = mae
            torch.save(model.state_dict(), "runs/%s/model_best_%s_%d.pt"%(args.name, args.model, epoch + 1))

        print("Validation mae ", mae)
        wandb.log({"Validation MAE": mae})

    torch.save(model.state_dict(), "runs/%s/final_model_test_corrected%s.pt"%(args.name, args.model))

    wandb.finish()

    return

def rotation_matrix_x_axis(theta):
    theta = np.radians(theta)
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return R

def rotation_matrix_y_axis(theta):
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), 0, -np.sin(theta)], 
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]])
    return R

def rotation_matrix_z_axis(theta):
    theta = np.radians(theta)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    return R

def new_structure_transformed(data, trans):
    data_new = {}
    data_new['equivalent_atoms'] = data['equivalent_atoms']
    data_new['sym_dataset'] = data['sym_dataset']
    

def test_augment(dataset, args):
    from data import is_group, rm_duplicates, find_almost_equal_entries
    # None, XZ_exchange, Xrotate, Yrotate, Zrotate
    theta = 45
    if args.test_augment == "None":
        return dataset
    if args.test_augment == "XZ_exchange":
        R = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    elif args.test_augment == "Xrotate":
        R = rotation_matrix_x_axis(theta)
    elif args.test_augment == "Yrotate":
        R = rotation_matrix_y_axis(theta)
    elif args.test_augment == "Zrotate":
        R = rotation_matrix_z_axis(theta)
    print("applying test augmentation R", R)
    for i in tqdm(range(len(dataset))):
        structure = dataset[i]['structure']
        Lat = dataset[i]['structure'].lattice.matrix.T
        Lat_new = np.matmul(R, Lat).T
        dataset[i]['structure'] = Structure(lattice=Lat_new, species=structure.atomic_numbers, coords=structure.frac_coords)
        target_tmp = np.array(dataset[i]['dielectric'])
        dataset[i]['dielectric'] = np.matmul(R, np.matmul(target_tmp, R.T))
        sym_dataset = get_symmetry_dataset(dataset[i]['structure'], 1e-5)
        dataset[i]['equivalent_atoms'] = sym_dataset['equivalent_atoms']
        dataset[i]['sym_dataset'] = sym_dataset

        mask = (torch.arange(32)+10.)
        mask[8:] *= 100
        rots = np.array(sym_dataset['rotations'])
        rots = rm_duplicates(rots)
        Lat = dataset[i]['structure'].lattice.matrix.T
        L_inv = np.linalg.inv(Lat)
        D_x = torch.zeros(32, 32)
        tmp_rot = np.matmul(Lat, np.matmul(rots, L_inv))
        assert is_group(tmp_rot), ("Found non_group rots", tmp_rot)
        D_tmp = irreps_output.D_from_matrix(torch.Tensor(tmp_rot))
        assert (((abs(D_tmp[:,5:8,5:8] - tmp_rot)).sum(dim=-1).sum(dim=-1) > 1e-2).sum() < 1e-5), (abs(D_tmp[:,5:8,5:8] - tmp_rot).sum(dim=-1).sum(dim=-1))
        D_x = D_tmp.sum(dim=0)
        feature_mask = torch.matmul(D_x, mask)
        mask_total = feature_mask[[0, 2, 3, 4, 8, 9, 10, 11, 12]]
        ideal_matrix = converter.to_cartesian(mask_total)
        dataset[i]['ideal_matrix'] = ideal_matrix
        D_x = D_x / D_tmp.shape[0]
        zero_mask = (D_x > 1e-5).float()
        D_x *= zero_mask
        dataset[i]['feature_mask'] = D_x
        dataset[i]['feature_mask_ori'] = feature_mask
        dataset[i]['reduce_rotations'] = None
        dataset[i]['wigner_D_per_atom'] = None
        dataset[i]['wigner_D_num'] = None
        dataset[i]['p_input'] = {}
        dataset[i]['p_input']['structure'] = dataset[i]['structure']
        dataset[i]['p_input']['equivalent_atoms'] = dataset[i]['equivalent_atoms']
        dataset[i]['matrix_equal'] = find_almost_equal_entries(dataset[i]['ideal_matrix'])
    
    return dataset


def test(model, args):
    # load the dataset
    if args.load_preprocessed:
        print("load preprocessed dataset ...")
    dataset_sym = get_dataset(dataset_name=args.target,use_corrected_structure=args.use_corrected_structure,load_preprocessed=args.load_preprocessed)
    count_parameters(model)
    id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dataset_sym),
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            keep_data_order=False,
        )
    dataset_train = [dataset_sym[x] for x in id_train]
    seen_ele=np.zeros([120])
    for itm in dataset_train:
        elems = itm['structure'].atomic_numbers
        for je in range(len(elems)):
            if seen_ele[elems[je]] < 1e-5:
                seen_ele[elems[je]] = 1.0
    
    unseen_list = []
    for i in range(120):
        if seen_ele[i] < 1e-5:
            unseen_list.append(i)
    print("unseen elements:", unseen_list)
    dataset_test = [dataset_sym[x] for x in id_test]
    dataset_test = test_augment(dataset_test, args)
    
    pyg_dataset_test = get_pyg_dataset(dataset_test, args.target)

    # form dataloaders
    collate_fn = pyg_dataset_test.collate

    test_loader = DataLoader(
        pyg_dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    print("n_test:", len(test_loader.dataset))

    # set up training configs
    model.to(device)
    MAE = nn.L1Loss()

    # evaluation and store the model
    model.eval()
    # store the label and prediction pairs
    cubic_label = [] # space group 195 <= i <= 230
    cubic_output = []
    cubic_ideal = []

    hexa_label = [] # space group 143 <= i <= 194
    hexa_output = []
    hexa_ideal = []

    tetr_label = [] # space group 75 <= i <= 142
    tetr_output = []
    tetr_ideal = []
    tetr_feat = []

    orth_label = [] # space group 16 <= i <= 74
    orth_output = []
    orth_ideal = []

    mono_label = [] # space group 3 <= i <= 15
    mono_output = []
    mono_ideal = []

    tric_label = [] # space group 1 <= i <= 2
    tric_output = []
    tric_ideal = []
    tric_feat = []

    i = 0
    mae_list =[]
    frob_list = []
    percen_list = []
    out_list = []
    error_eT = []

    for data in tqdm(test_loader):
        structure, mask, equality, labels, rot_list = data
        structure, mask, equality, labels = structure.to(device), mask.to(device), equality.to(device), labels.to(device)
        
        if args.model == "gmtnet":
            outputs = model(structure, mask, equality) # 3 * 3
            outputs = outputs.view(3, 3).cpu().detach()
            tmpo = outputs
            outputs = (outputs + outputs.T) / 2
            error_eT.append((abs(outputs.view(-1) - tmpo.view(-1))).mean())
        elif args.model == "megnet":
            outputs = model(structure,test=True).view(1, 3, 3).cpu().detach() # 3 * 3
        else:
            outputs = model(structure).view(1, 3, 3).cpu().detach() # 3 * 3
        
        out_list.append(outputs)

        labels = labels.cpu()
        mae_list.append(abs(outputs - labels).view(-1).mean())
        frob_ = ((labels.view(-1) - outputs.view(-1)) ** 2).sum() ** 0.5
        frob_norm = (labels.view(-1) ** 2).sum() ** 0.5
        frob_list.append(frob_)
        percen_list.append(frob_/frob_norm)
        space_g = dataset_test[i]['sym_dataset']['number']
        if space_g >= 195:
            cubic_label.append(labels.view(3, 3))
            cubic_output.append(outputs)
            cubic_ideal.append(dataset_test[i]['ideal_matrix'])
        elif space_g >= 143:
            hexa_label.append(labels.view(3, 3))
            hexa_output.append(outputs)
            hexa_ideal.append(dataset_test[i]['ideal_matrix'])
        elif space_g >= 75:
            tetr_label.append(labels.view(3, 3))
            tetr_output.append(outputs)
            tetr_ideal.append(dataset_test[i]['ideal_matrix'])
            tetr_feat.append(dataset_test[i]['feature_mask'])
        elif space_g >= 16:
            orth_label.append(labels.view(3, 3))
            orth_output.append(outputs)
            orth_ideal.append(dataset_test[i]['ideal_matrix'])
        elif space_g >= 3:
            mono_label.append(labels.view(3, 3))
            mono_output.append(outputs)
            mono_ideal.append(dataset_test[i]['ideal_matrix'])
        else:
            tric_label.append(labels.view(3, 3))
            tric_output.append(outputs)
            tric_ideal.append(dataset_test[i]['ideal_matrix'])
            tric_feat.append(dataset_test[i]['feature_mask'])

        i += 1
    
    # with open('ours_test_res.pkl', 'wb') as f:
    #     pk.dump(out_list, f)
    
    # with open('dataset_test.pkl', 'wb') as f:
    #     pk.dump(dataset_test, f)
    print("diff in eT", np.mean(error_eT))
    print("MAE ", np.mean(mae_list))
    print("M_Frob", np.mean(frob_list))
    percen_list = np.array(percen_list)
    print("EwT 25", np.sum(percen_list < 0.25) / percen_list.shape[0])
    print("EwT 10", np.sum(percen_list < 0.1) / percen_list.shape[0])
    print("EwT 5", np.sum(percen_list < 0.05) / percen_list.shape[0])
    print("EwT 2", np.sum(percen_list < 0.02) / percen_list.shape[0])
    # evaluation for cubic system
    print("total number of cubic systems", len(cubic_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(cubic_label)):
        label = cubic_label[i].view(9)
        pred = cubic_output[i].view(9)
        ideal = cubic_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1
        
        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1


    # CUBIC label errors
    print("CUBIC systems: Label symmetry error - Zero Error", label_sym_error, "Equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "Equality Error", pred_equi_error, "Fnorm", F_error/len(cubic_label))

    # evaluation for Tetragonal system
    print("total number of Tetragonal systems", len(tetr_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(tetr_label)):
        label = tetr_label[i].view(9)
        pred = tetr_output[i].view(9)
        ideal = tetr_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1
        
        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1
        

    # label errors
    print("Tetragonal systems: Label symmetry error - Zero Error", label_sym_error, "Equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "Equality Error", pred_equi_error, "Fnorm", F_error/len(tetr_label))

    # evaluation for hexagonal system
    print("total number of hexagonal systems", len(hexa_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(hexa_label)):
        label = hexa_label[i].view(9)
        pred = hexa_output[i].view(9)
        ideal = hexa_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1
        
        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1
        

    # label errors
    print("Hexagonal systems: Label symmetry error - Zero Error", label_sym_error, "Equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "Equality Error", pred_equi_error, "Fnorm", F_error/len(hexa_label))

    # evaluation for Orthorhombic system
    print("total number of Orthorhombic systems", len(orth_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(orth_label)):
        label = orth_label[i].view(9)
        pred = orth_output[i].view(9)
        ideal = orth_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1
        
        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    print("yes")
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    print("yes")
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1
        

    # label errors
    print("Orthorhombic systems: Label symmetry error - Zero Error", label_sym_error, "equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "equality Error", pred_equi_error, "Fnorm", F_error/len(orth_label))

    # evaluation for Orthorhombic system
    print("total number of Monoclinic systems", len(mono_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(mono_label)):
        label = mono_label[i].view(9)
        pred = mono_output[i].view(9)
        ideal = mono_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1
        
        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    print("yesmono")
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    print("yesmono")
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1
    

    # label errors
    print("Monoclinic systems: Label symmetry error - Zero Error", label_sym_error, "Equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "Equality Error", pred_equi_error, "Fnorm", F_error/len(mono_label))

    # evaluation for Triclinic system
    print("total number of Triclinic systems", len(tric_label))
    label_sym_error = 0
    label_equi_error = 0
    F_error = 0
    pred_sym_error = 0
    pred_equi_error = 0
    for i in range(len(tric_label)):
        label = tric_label[i].view(9)
        pred = tric_output[i].view(9)
        ideal = tric_ideal[i].view(9)

        F_error += ((label - pred) ** 2).sum() ** 0.5

        zero_entries = abs(ideal) < 1.0
        if (abs(label[zero_entries]) > 1e-5).any():
            label_sym_error += 1
        if (abs(pred[zero_entries]) > 1e-5).any():
            pred_sym_error += 1

        # equality analysis
        label_mask = abs(ideal) > 1.0
        label = label[label_mask]
        pred = pred[label_mask]
        ideal = ideal[label_mask]
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(label[px] - label[py]) > 1e-4:
                        flag = True
                        break
        
        if flag: label_equi_error += 1
        flag = False
        for px in range(label.shape[0] - 1):
            for py in range(px + 1, label.shape[0]):
                if abs(ideal[px] / ideal[py] - 1.0) < 1e-5:
                    if abs(pred[px] - pred[py]) > 1e-4:
                        flag = True
                        break
        if flag: pred_equi_error += 1
    

    # label errors
    print("Triclinic systems: Label symmetry error - Zero Error", label_sym_error, "Equality Error", label_equi_error)
    # Prediction errors
    print("Prediction error - Zero Error", pred_sym_error, "Equality Error", pred_equi_error, "Fnorm", F_error/len(tric_label))
    
    return



def main():
    parser = argparse.ArgumentParser(description='Training script')

    # Define command-line arguments
    # training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and evaluating')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay')
    parser.add_argument('--loss', type=str, default='huber', help='mse or l1 or huber')
    parser.add_argument('--model', type=str, default='gmtnet', help='gmtnet, ecomformer or megnet')
    parser.add_argument('--load_model', type=bool, default=False, help='load pretrained model or not')
    parser.add_argument('--project', type=str, default='test', help='name of project for wandb visualization')
    parser.add_argument('--name', type=str, default='test', help='name of project for storage')
    parser.add_argument('--reduce_cell', type=bool, default=False, help='reduce the cell into irreducible atom sets, not used')
    parser.add_argument('--use_mask', type=bool, default=True, help='symmetry correction module introduced in the paper')
    # dataset parameters
    parser.add_argument('--split_seed', type=int, default=32, help='the random seed of spliting data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training ratio used in data split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='evaluate ratio used in data split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio used in data split')
    parser.add_argument('--target', type=str, default='dielectric', help='dielectric, piezoelectric, or elastic')
    parser.add_argument('--test_augment', type=str, default='None', help='None, XZ_exchange, Xrotate, Yrotate, Zrotate')
    parser.add_argument('--threshold', type=float, default=100., help='threshold to remove samples')
    parser.add_argument('--use_corrected_structure', type=bool, default=False, help='correct input structure or not')
    parser.add_argument('--load_preprocessed', type=bool, default=False, help='load previous processed dataset')

    args = parser.parse_args()

    print('Training settings:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Learning rate: {args.learning_rate}')
    print(args)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)
    # load the model
    if args.model == "gmtnet":
        model = GMTNet(args)
    elif args.model == "megnet":
        model = MEGNET()
    elif args.model == "mace":
        model = MACE(avg_num_neighbors=34)
    elif args.model == "ecomformer":
        model = EComformerEquivariant(args)
    else:
        model = DimeNetPlusPlusWrap()

    if not os.path.exists('runs/' + args.name):
        # Create the directory
        os.makedirs('runs/' + args.name)
        
    if args.load_model:
        if args.model == "gmtnet":
            saved_model_path = "yourpath/model_final.pt"
        state_dict = torch.load(saved_model_path)
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

    train(model, args)
    # test(model, args)

if __name__ == "__main__":
    main()
