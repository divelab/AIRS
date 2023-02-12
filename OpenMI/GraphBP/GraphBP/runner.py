import torch
from torch.utils.data import DataLoader, dataset
import os
import time
import pandas as pd
import numpy as np
from model import GraphBP
from dataset_from_scratch import CrossDocked2020_SBDD, collate_mols
import torch.optim as optim
from torch_scatter import scatter

from Bio.PDB import PDBParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem

atomic_num_to_type = {5:0, 6:1, 7:2, 8:3, 9:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 21:11, 23:12, 26:13, 29:14, 30:15, 33:16, 34:17, 35:18, 39:19, 42:20, 44:21, 45:22, 51:23, 53:24, 74:25, 79:26}

atomic_element_to_type = {'C':27, 'N':28, 'O':29, 'NA':30, 'MG':31, 'P':32, 'S':33, 'CL':34, 'K':35, 'CA':36, 'MN':37, 'CO':38, 'CU':39, 'ZN':40, 'SE':41, 'CD':42, 'I':43, 'CS':44, 'HG':45}

class Runner():
    def __init__(self, conf, out_path=None):
        self.conf = conf
        if conf['gen_model'] == 'GraphBP':
            self.model = GraphBP(**conf['model'])
        else:
            print('Please give correct gen_model name!')
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])
        self.focus_ce = torch.nn.BCELoss()
        self.contact_ce = torch.nn.BCELoss()
        self.out_path = out_path
    

    def _train_epoch(self, loader):
        self.model.train()
        total_ll_node, total_ll_dist, total_ll_angle, total_ll_torsion, total_focus_ce, total_contact_ce = 0, 0, 0, 0, 0, 0
        skip_batch_num = 0
        for iter_num, data_batch in enumerate(loader):
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            
            if self.conf['model']['use_gpu']:
                for key in data_batch:
                    data_batch[key] = data_batch[key].to('cuda')

            if data_batch['atom_type'].size(0) > 600000:
                skip_batch_num += 1
                print("Skip batch to avoid OOM!")
                continue
            node_out, focus_score, contact_score, dist_out, angle_out, torsion_out = self.model(data_batch)
            cannot_focus = data_batch['cannot_focus']
            cannot_contact = data_batch['cannot_contact']

            ll_node = torch.mean(1/2 * (node_out[0] ** 2) - node_out[1])
            ll_dist = torch.mean(1/2 * (dist_out[0] ** 2) - dist_out[1])
            ll_angle = torch.mean(1/2 * (angle_out[0] ** 2) - angle_out[1])
            ll_torsion = torch.mean(1/2 * (torsion_out[0] ** 2) - torsion_out[1])
            focus_ce = self.focus_ce(focus_score, cannot_focus)
            contact_ce = self.contact_ce(contact_score, cannot_contact)

            loss = ll_node + ll_dist + ll_angle + ll_torsion + focus_ce + contact_ce

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ll_node += ll_node.to('cpu').item()
            total_ll_dist += ll_dist.to('cpu').item()
            total_ll_angle += ll_angle.to('cpu').item()
            total_ll_torsion += ll_torsion.to('cpu').item()
            total_focus_ce += focus_ce.to('cpu').item()
            total_contact_ce += contact_ce.to('cpu').item()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            
            duration = t_end - t_start

            if iter_num % self.conf['verbose'] == 0:
                print('Training iteration {} | loss node {:.6f} dist {:.6f} angle {:.6f} torsion {:.6f} focus {:.6f} contact {:.6f} duration {:.6f}'.format(iter_num, ll_node.to('cpu').item(), 
                    ll_dist.to('cpu').item(), ll_angle.to('cpu').item(), ll_torsion.to('cpu').item(), focus_ce.to('cpu').item(), contact_ce.to('cpu').item(), duration))
        
        iter_num += 1
        iter_num -= skip_batch_num
        return total_ll_node / iter_num, total_ll_dist / iter_num, total_ll_angle / iter_num, total_ll_torsion / iter_num, total_focus_ce / iter_num, total_contact_ce / iter_num, skip_batch_num


    def train(self, binding_site_range):
        dataset = CrossDocked2020_SBDD(binding_site_range=binding_site_range)
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=True, collate_fn=collate_mols, num_workers=self.conf['num_workers'])


        epochs = self.conf['epochs']
        for epoch in range(epochs):
            avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce, avg_contact_ce, skip_batch_num = self._train_epoch(loader)
            print('=============================================')
            print('Training | Average loss node {:.6f} dist {:.6f} angle {:.6f} torsion {:.6f} focus {:.6f} contact {:.6f}'.format(avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce, avg_contact_ce))
            print('Skip batch nums:', skip_batch_num)
            print('=============================================')
            if self.out_path is not None:
                torch.save(self.model.state_dict(), os.path.join(self.out_path, 'model_{}.pth'.format(epoch)))
                file_obj = open(os.path.join(self.out_path, 'record.txt'), 'a')
                file_obj.write('Training | Average loss node {:.6f} dist {:.6f} angle {:.6f} torsion {:.6f} focus {:.6f} contact {:.6f}\n'.format(avg_ll_node, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce, avg_contact_ce))
                file_obj.close()
    


    def generate(self, num_gen, temperature=[1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5, contact_th=0.5, add_final=False, contact_prob=False, data_root='./data/crossdock2020', data_file='./data/crossdock2020/selected_test_targets.types', atomic_num_to_type=atomic_num_to_type, atomic_element_to_type = atomic_element_to_type, known_binding_site=False, binding_site_range=15.0):
        data_cols = [
            'low_rmsd',
            'true_aff',
            'xtal_rmsd',
            'rec_src',
            'lig_src',
            'vina_aff'
        ]
        data_lines = pd.read_csv(
            data_file, sep=' ', names=data_cols, index_col=False
        )
        pdb_parser = PDBParser()
            
        
        all_mol_dicts = {}
        
        for index in range(len(data_lines)):
            example = data_lines.iloc[index]
            rec_src = example.rec_src
            lig_src = example.lig_src.rsplit('.', 1)[0]
            print(rec_src)
            print(lig_src)
            print("=============")
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PDBConstructionWarning)
                rec_structure = pdb_parser.get_structure('', os.path.join(data_root, rec_src))
                
            rec_atom_type = [atomic_element_to_type[atom.element] for atom in rec_structure.get_atoms() if atom.element!='H']
            rec_position = np.stack([atom.coord for atom in rec_structure.get_atoms() if atom.element!='H'], axis=0)
            rec_atom_type = torch.tensor(rec_atom_type)
            rec_position = torch.tensor(rec_position)
            # print(rec_atom_type.shape)
            # print(rec_position.shape)
            
            if known_binding_site:
                supp = Chem.SDMolSupplier()
                print("Generate molecules with given binding site infomation...")
                sdf_file = os.path.join(data_root, lig_src)
                supp.SetData(open(sdf_file).read(), removeHs=False, sanitize=False)
                lig_mol = Chem.rdmolops.RemoveAllHs(supp[0], sanitize=False)
                lig_n_atoms = lig_mol.GetNumAtoms()
                lig_pos = supp.GetItemText(0).split('\n')[4:4+lig_n_atoms]
                lig_position = np.array([[float(x) for x in line.split()[:3]] for line in lig_pos], dtype=np.float32)
                lig_position = torch.tensor(lig_position)
                lig_center = torch.mean(lig_position, dim=0)
                rec_atom_dist_to_lig_center = torch.sqrt(torch.sum(torch.square(rec_position - lig_center), dim=-1))
                # print(lig_position)
                # print(rec_position)
                # print(rec_atom_dist_to_lig_center)
                selected_mask = rec_atom_dist_to_lig_center <= binding_site_range
                assert torch.sum(selected_mask) > 0
                rec_atom_type = rec_atom_type[selected_mask]
                rec_position = rec_position[selected_mask]
                # print(rec_atom_type.shape)
                # print(rec_position.shape)
                # print(lig_position)
                # print(rec_position)
                del supp
            
            
            num_remain = num_gen
            one_time_gen = self.conf['chunk_size']
            type_to_atomic_number_dict = {atomic_num_to_type[k]:k for k in atomic_num_to_type}
            type_to_atomic_number = np.zeros([max(type_to_atomic_number_dict.keys())+1], dtype=int)
            for k in type_to_atomic_number_dict:
                type_to_atomic_number[k] = type_to_atomic_number_dict[k]
            mol_dicts = {}

            self.model.eval()
            while num_remain > 0:
                if num_remain > one_time_gen:
                    mols = self.model.generate(type_to_atomic_number, rec_atom_type, rec_position, one_time_gen, temperature, min_atoms, max_atoms, focus_th, contact_th, add_final, contact_prob)
                else:
                    mols = self.model.generate(type_to_atomic_number, rec_atom_type, rec_position, num_remain, temperature, min_atoms, max_atoms, focus_th, contact_th, add_final, contact_prob)

                for num_atom in mols:
                    if not num_atom in mol_dicts.keys():
                        mol_dicts[num_atom] = mols[num_atom]
                    else:
                        mol_dicts[num_atom]['_atomic_numbers'] = np.concatenate((mol_dicts[num_atom]['_atomic_numbers'], mols[num_atom]['_atomic_numbers']), axis=0)
                        mol_dicts[num_atom]['_positions'] = np.concatenate((mol_dicts[num_atom]['_positions'], mols[num_atom]['_positions']), axis=0)
                        mol_dicts[num_atom]['_focus'] = np.concatenate((mol_dicts[num_atom]['_focus'], mols[num_atom]['_focus']), axis=0)
                    num_mol = len(mols[num_atom]['_atomic_numbers'])
                    num_remain -= num_mol
            
                print('{} molecules are generated!'.format(num_gen-num_remain))
            mol_dicts['rec_src'] = rec_src
            mol_dicts['lig_src'] = lig_src
            all_mol_dicts[index] = mol_dicts
        
        return all_mol_dicts
            
            