import argparse
from os.path import join
import torch
import pickle
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.property_prediction import main_qm9_prop
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import numpy as np
from qm9.property_prediction.prop_utils import get_adj_matrix

def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


def spherical_seq_for_edm_eval(input_path, remove_h = False, max_num_atoms = 100):
    if remove_h:
        dict = {'C': 0, 'C@': 0,'C@@': 0, 'N': 1, 'O': 2, 'F': 3}
    else:
        dict = {'H': 0, 'C': 1, 'C@': 1,'C@@': 1, 'N': 2, 'O': 3, 'F': 4}
    
    with open(input_path, 'r') as file:
        num_samples = len(file.readlines())
    file.close()

    all_len = []
    max_num_atoms = max_num_atoms
    num_type = max(dict.values()) + 1
    count_invalid_len = 0
    count_invalid_seq = 0
    count_invalid_coords = 0
    one_hot = torch.zeros((num_samples, max_num_atoms, num_type), dtype=float)
    x = torch.zeros((num_samples, max_num_atoms, 3), dtype=float)
    node_mask = torch.zeros((num_samples, max_num_atoms), dtype=float)
    y = torch.zeros((num_samples), dtype=float)

    idx = 0
    with open(input_path, 'r') as file:
        for num_line, line in enumerate(tqdm(file)):
            if num_line >= num_samples:
                break
            
            split = np.array(line.split())
            prop = float(split[0][1:-1])
            mol = split[1:]
            try:
                mol = mol.reshape(-1,4)
            except:
                for cut_idx in range(int(len(mol)/4)-1):
                    vals = mol[4 * cut_idx:4 * cut_idx + 4]
                    if vals[2][-1] != '°' or vals[3][-1] != '°':
                        mol = mol[:4 * cut_idx].reshape(-1,4)
                        break
                    else:
                        try:
                            dict[vals[0]]
                            vals[1].astype(float)
                            np.str_(vals[2][:-1]).astype(float)
                            np.str_(vals[3][:-1]).astype(float)
                        except:
                            mol = mol[:4 * cut_idx].reshape(-1,4)
                            break
                if cut_idx == int(len(mol)/4)-2:
                    mol = mol[:4 * cut_idx + 4].reshape(-1,4)
                # print('invalid length')
                count_invalid_len += 1
                # continue
            if len(mol.shape) == 1:
                count_invalid_seq += 1
                continue
            seq = mol[:,0]

            try:
                one_hot_emb = torch.nn.functional.one_hot(torch.tensor([dict[key] for key in seq]), num_type)
            except:
                # print('invalid seq')
                count_invalid_seq += 1
                continue
            try:
                spherical_coords = mol[:,1:]
                d = spherical_coords[:,0].astype(float)
                theta = np.array([s[:-1] for s in spherical_coords[:,1]]).astype(float)
                phi = np.array([s[:-1] for s in spherical_coords[:,2]]).astype(float)
                invariant_coords = np.stack((d * np.sin(theta) * np.cos(phi), d * np.sin(theta) * np.sin(phi), d * np.cos(theta))).T
            except:
                # print('invalid coords')
                count_invalid_coords += 1
                continue   
            
            num_nodes = len(seq)
            if num_nodes > max_num_atoms:
                continue
            all_len.append(num_nodes)
            one_hot[idx, :num_nodes] = one_hot_emb
            x[idx,:num_nodes] = torch.tensor(invariant_coords)
            node_mask[idx, :num_nodes] = 1.
            y[idx] = prop
            idx += 1
    one_hot, x, node_mask, y = one_hot[:idx], x[:idx], node_mask[:idx], y[:idx]
    print('max_num_atoms', 0 if len(all_len) == 0 else max(all_len))
    frequency_mol_len = {}
    for element in all_len:
        frequency_mol_len[element] = frequency_mol_len.get(element, 0) + 1
    molecules = {'one_hot': one_hot, 'positions': x, 'atom_mask': node_mask, 'y':y}
    # torch.save(molecules, write_path)
    print('invalid: 1. length is not a multiple of 4; 2. invalid atom type; 3. invalid coords:\n', 
          count_invalid_len, count_invalid_seq, count_invalid_coords)
    print('done')
    return molecules


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class CondMol(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, max_num_atoms=40):
        """
        """
        self.data = spherical_seq_for_edm_eval(txt_file, max_num_atoms=max_num_atoms)
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


class PreprocessQM9:
    def __init__(self, load_charges=False):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        to_keep = (batch['atom_mask'].sum(0) > 0)

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['atom_mask']

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch


def eval(model, loader, mean, mad, device, log_interval=20):
    loss_l1 = nn.L1Loss()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        model.eval()
        batch_size, n_nodes = data['atom_mask'].size()
        # # atom_mask = data['atom_mask']
        # # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        # # #mask diagonal
        # # diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        # # edge_mask *= diag_mask

        # # #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        # # data['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)
        #charges = data['charges'].to(device, dtype).squeeze(2)
        #nodes = prop_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = get_adj_matrix(n_nodes, batch_size, device)
        label = data['y'].to(device, torch.float32)

        pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        
        loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""

        if i % log_interval == 0:
            print(prefix + "Iteration %d \t loss %.4f" % (i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))

    return res['loss'] / res['counter']


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen) # second half data
    property_norms = compute_mean_mad(dataloaders, [args.property], args_gen.dataset)
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    dataset = CondMol(txt_file = args.generated_path, max_num_atoms=args.max_num_atoms)
    preprocess = PreprocessQM9()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=preprocess.collate_fn)
    print("Language model: We evaluate the classifier on our generated samples")
    loss = eval(classifier, dataloader, mean, mad, args.device, args.log_interval)
    print("Loss classifier on Generated samples: %.4f" % loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generated_path', type=str, default='generation/cond_alpha_example.txt')
    parser.add_argument('--property', type=str, default='alpha', help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--max_num_atoms', type=int, default=30, help='break point or not')
    parser.add_argument('--batch_size', type=int, default=64, help='break point or not')
    parser.add_argument('--task', type=str, default='ours', help='naive, edm, ours, qm9_second_half, qualitative')
    parser.add_argument('--device', type=int, default=9, help='Device to use')
    parser.add_argument('--log_interval', type=int, default=5, help='break point or not')

    args = parser.parse_args()
    args.generators_path = 'e3_diffusion_for_molecules/outputs/exp_35_conditional_nf192_9l_alpha'
    args.classifiers_path = 'data/geom/checkpoints/QM9/Property_Classifiers/exp_class_' + args.property
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    args.device = device

    main_quantitative(args)
