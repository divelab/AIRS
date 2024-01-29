import os
import torch
from torch_geometric.data import DataLoader
import numpy as np
from model import MatGen
from dataset import MatDataset
from utils import StandardScalerTorch


class Runner():
    def __init__(self, conf, score_norm_path):
        self.conf = conf
        score_norm = np.loadtxt(score_norm_path)
        self.model = MatGen(**conf['model'], score_norm=score_norm)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])
    

    def _get_normalizer(self, dataset):
        normalizer = StandardScalerTorch()
        lengths = torch.tensor([data['graph_arrays'][2] for data in dataset.data_dict_list])
        angles = torch.tensor([data['graph_arrays'][4] for data in dataset.data_dict_list])
        length_angles = torch.cat((lengths, angles), dim=-1)
        normalizer.fit(length_angles)
        normalizer.means, normalizer.stds = normalizer.means.to('cuda'), normalizer.stds.to('cuda')
        return normalizer


    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_elem_type_num_loss, total_elem_type_loss, total_elem_num_loss, total_lattice_loss, total_coord_loss = 0, 0, 0, 0, 0
        total_kld_loss, total_kld_loss1, total_kld_loss2, total_kld_loss3 = 0.0, 0.0, 0.0, 0.0
        total_loss = 0

        for iter_num, data_batch in enumerate(loader):
            data_batch = data_batch.to('cuda')
            loss_dict = self.model(data_batch, temp=self.conf['train_temp'])

            kld_loss, elem_type_num_loss, elem_type_loss, elem_num_loss, lattice_loss, coord_loss = loss_dict['kld_loss'], loss_dict['elem_type_num_loss'], \
                loss_dict['elem_type_loss'], loss_dict['elem_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
            loss = self.conf['kld_weight'] * kld_loss + self.conf['elem_type_num_weight'] * elem_type_num_loss + \
                self.conf['elem_type_weight'] * elem_type_loss + self.conf['elem_num_weight'] * elem_num_loss \
                + self.conf['lattice_weight'] * lattice_loss + self.conf['coord_weight'] * coord_loss
            
            if epoch > 10 and (loss < 0.1 or loss > 100):
                self.optimizer.zero_grad()
                return None
                
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                return None

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.conf['max_grad_value'])
            self.optimizer.step()
            
            total_loss += loss.to('cpu').item()
            total_kld_loss += kld_loss.to('cpu').item()
            total_elem_type_num_loss += elem_type_num_loss.to('cpu').item()
            total_elem_type_loss += elem_type_loss.to('cpu').item()
            total_elem_num_loss += elem_num_loss.to('cpu').item()
            total_lattice_loss += lattice_loss.to('cpu').item()
            total_coord_loss += coord_loss.to('cpu').item()
            
            if 'kld_loss1' in loss_dict and 'kld_loss2' in loss_dict and 'kld_loss3' in loss_dict:
                kld_loss1, kld_loss2, kld_loss3 = loss_dict['kld_loss1'].to('cpu').item(), loss_dict['kld_loss2'].to('cpu').item(), loss_dict['kld_loss3'].to('cpu').item()
                total_kld_loss1 += kld_loss1
                total_kld_loss2 += kld_loss2
                total_kld_loss3 += kld_loss3
            else:
                kld_loss1, kld_loss2, kld_loss3 = 0.0, 0.0, 0.0

            if iter_num % self.conf['verbose'] == 0:
                print('Training iteration {} | loss kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f}'.format(iter_num, kld_loss.to('cpu').item(), 
                    kld_loss1, kld_loss2, kld_loss3, elem_type_num_loss.to('cpu').item(), elem_type_loss.to('cpu').item(), elem_num_loss.to('cpu').item(), lattice_loss.to('cpu').item(), coord_loss.to('cpu').item()))
        
        iter_num += 1
        return total_loss / iter_num, total_kld_loss / iter_num, total_kld_loss1 / iter_num, total_kld_loss2 / iter_num, total_kld_loss3 / iter_num, \
            total_elem_type_num_loss / iter_num, total_elem_type_loss / iter_num, total_elem_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num
    

    def train(self, data_path, val_data_path, out_path):
        torch.nn.init.constant_(self.model.fc_var.weight, 1e-10)
        torch.nn.init.constant_(self.model.fc_var.bias, 0.)
        torch.nn.init.constant_(self.model.fc_lattice_log_var[-1].weight, 1e-10)
        torch.nn.init.constant_(self.model.fc_lattice_log_var[-1].bias, 0.)

        dataset = MatDataset(data_path, **self.conf['data'])
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=True)
        normalizer = self._get_normalizer(dataset)
        self.model.lattice_normalizer = normalizer

        val_dataset = MatDataset(val_data_path, **self.conf['data'])
        val_loader = DataLoader(val_dataset, batch_size=self.conf['batch_size'], shuffle=False)

        end_epoch = self.conf['end_epoch']
        for epoch in range(self.conf['start_epoch'], end_epoch+1):
            if epoch == self.conf['start_epoch']:
                last_optim_dict = self.optimizer.state_dict().copy()
                last_model_dict = self.model.state_dict().copy()
                last_last_optim_dict, last_last_model_dict = last_optim_dict, last_model_dict
            else:
                last_last_optim_dict, last_last_model_dict = last_optim_dict, last_model_dict
                last_optim_dict = self.optimizer.state_dict().copy()
                last_model_dict = self.model.state_dict().copy()

            train_returns = self._train_epoch(loader, epoch)
            
            retry_num = 0
            while train_returns is None and retry_num <= 3:
                retry_num += 1
                self.optimizer.load_state_dict(last_optim_dict)
                self.model.load_state_dict(last_model_dict)
                train_returns = self._train_epoch(loader, epoch)
            
            if train_returns is None:
                retry_num = 0
                while train_returns is None and retry_num <= 3:
                    retry_num += 1
                    self.optimizer.load_state_dict(last_last_optim_dict)
                    self.model.load_state_dict(last_last_model_dict)
                    train_returns = self._train_epoch(loader, epoch)
                if train_returns is None:
                    exit()

            avg_loss, avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss = train_returns
            
            _, _, _, _, _, _, _, _, _, _, elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match = self.valid(loader)

            avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2, avg_val_kld_loss3, avg_val_elem_type_num_loss, avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss,\
                avg_val_coord_loss, val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match = self.valid(val_loader)
            print('Training | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} \
                elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f}'.format(avg_loss, 
                avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss, \
                elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match))
            print('Validation | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} \
                elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f}'.format(avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2, \
                avg_val_kld_loss3, avg_val_elem_type_num_loss, avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss, avg_val_coord_loss, \
                val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match))
            
            if out_path is not None:
                if (epoch + 1) % self.conf['save_interval'] == 0:
                    torch.save(self.model.state_dict(), os.path.join(out_path, 'model_{}.pth'.format(epoch)))
                
                file_obj = open(os.path.join(out_path, 'train.txt'), 'a')
                file_obj.write('Training | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} '
                    'elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f}\n'.format(avg_loss, 
                    avg_kld_loss, avg_kld_loss1, avg_kld_loss2, avg_kld_loss3, avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss, \
                    elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match))
                file_obj.close()

                file_obj = open(os.path.join(out_path, 'val.txt'), 'a')
                file_obj.write('Validation | Average loss {:.4f} kld {:.4f} kld1 {:.4f} kld2 {:.4f} kld3 {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} elem_type_num_acc {:.4f} '
                    'elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f}\n'.format(avg_val_loss, avg_val_kld_loss, avg_val_kld_loss1, avg_val_kld_loss2, avg_val_kld_loss3, avg_val_elem_type_num_loss, \
                    avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss, avg_coord_loss, val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match))
                file_obj.close()
    

    def valid(self, loader):
        self.model.eval()
        total_kld_loss, total_elem_type_num_loss, total_elem_type_loss, total_elem_num_loss, total_lattice_loss, total_coord_loss = 0, 0, 0, 0, 0, 0
        total_kld_loss1, total_kld_loss2, total_kld_loss3 = 0.0, 0.0, 0.0
        total_loss = 0
        total_elem_type_num_num, total_elem_type_num_correct = 0, 0
        total_elem_type_num, total_pos_elem_type_num, total_elem_num_num = 0, 0, 0
        total_elem_type_correct, total_pos_elem_type_correct, total_elem_num_correct = 0, 0, 0
        total_elem_type_match, total_elem_num_match, total_comp_match = 0, 0, 0

        with torch.no_grad():
            for iter_num, data_batch in enumerate(loader):
                data_batch = data_batch.to('cuda')
                loss_dict = self.model(data_batch, temp=self.conf['val_temp'], eval=True)

                kld_loss, elem_type_num_loss, elem_type_loss, elem_num_loss, lattice_loss, coord_loss = loss_dict['kld_loss'], loss_dict['elem_type_num_loss'], \
                    loss_dict['elem_type_loss'], loss_dict['elem_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
                loss = self.conf['kld_weight'] * kld_loss + self.conf['elem_type_num_weight'] * elem_type_num_loss + self.conf['elem_type_weight'] * elem_type_loss \
                    + self.conf['elem_num_weight'] * elem_num_loss + self.conf['lattice_weight'] * lattice_loss \
                    + self.conf['coord_weight'] * coord_loss
                
                total_loss += loss.to('cpu').item()
                total_kld_loss += kld_loss.to('cpu').item()
                total_elem_type_num_loss += elem_type_num_loss.to('cpu').item()
                total_elem_type_loss += elem_type_loss.to('cpu').item()
                total_elem_num_loss += elem_num_loss.to('cpu').item()
                total_lattice_loss += lattice_loss.to('cpu').item()
                total_coord_loss += coord_loss.to('cpu').item()

                if 'kld_loss1' in loss_dict and 'kld_loss2' in loss_dict and 'kld_loss3' in loss_dict:
                    kld_loss1, kld_loss2, kld_loss3 = loss_dict['kld_loss1'].to('cpu').item(), loss_dict['kld_loss2'].to('cpu').item(), loss_dict['kld_loss3'].to('cpu').item()
                    total_kld_loss1 += kld_loss1
                    total_kld_loss2 += kld_loss2
                    total_kld_loss3 += kld_loss3
                else:
                    kld_loss1, kld_loss2, kld_loss3 = 0.0, 0.0, 0.0
                
                total_elem_type_num_num += loss_dict['total_elem_type_num_num']
                total_elem_type_num_correct += loss_dict['elem_type_num_correct']
                total_elem_type_num += loss_dict['total_elem_type_num']
                total_pos_elem_type_num += loss_dict['pos_elem_type_num']
                total_elem_num_num += loss_dict['elem_num_num']
                total_elem_type_correct += loss_dict['elem_type_correct']
                total_pos_elem_type_correct += loss_dict['pos_elem_type_correct']
                total_elem_num_correct += loss_dict['elem_num_correct']
                total_elem_type_match += loss_dict['composition_correct'][0]
                total_elem_num_match += loss_dict['composition_correct'][1]
                total_comp_match += loss_dict['composition_correct'][2]
        
        iter_num += 1
        return total_loss / iter_num, total_kld_loss / iter_num, total_kld_loss1 / iter_num, total_kld_loss2 / iter_num, total_kld_loss3 / iter_num, \
            total_elem_type_num_loss / iter_num, total_elem_type_loss / iter_num, total_elem_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num, \
            total_elem_type_num_correct / total_elem_type_num_num, total_elem_type_correct / total_elem_type_num, total_pos_elem_type_correct / total_pos_elem_type_num, \
            total_elem_num_correct / total_elem_num_num, total_elem_type_match / total_elem_type_num_num, total_elem_num_match / total_elem_type_num_num, total_comp_match / total_elem_type_num_num


    def generate(self, num_gen, data_path, coord_num_langevin_steps=100, coord_step_rate=1e-4):
        dataset = MatDataset(data_path, **self.conf['data'])
        normalizer = self._get_normalizer(dataset)
        self.model.lattice_normalizer = normalizer
        
        num_atoms_list, atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], [], []
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        temperature = self.conf['gen_temp']
        coord_noise_start = self.conf['model']['noise_start']
        coord_noise_end = self.conf['model']['noise_end']
        coord_num_diff_steps = self.conf['model']['num_time_steps']
        
        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mat_arrays = self.model.generate(one_time_gen, temperature, coord_noise_start, coord_noise_end, coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate)
            else:
                mat_arrays = self.model.generate(num_remain, temperature, coord_noise_start, coord_noise_end, coord_num_diff_steps, coord_num_langevin_steps, coord_step_rate)
            
            num_atoms_list.append(mat_arrays[0].detach().cpu())
            atom_types_list.append(mat_arrays[1].detach().cpu())
            lengths_list.append(mat_arrays[2].detach().cpu())
            angles_list.append(mat_arrays[3].detach().cpu())
            frac_coords_list.append(mat_arrays[4].detach().cpu())
            
            num_mat = len(mat_arrays[0])
            num_remain -= num_mat
            print('{} materials are generated!'.format(num_gen - num_remain))
        
        all_num_atoms = torch.cat(num_atoms_list, dim=0)
        all_atom_types = torch.cat(atom_types_list, dim=0)
        all_lengths = torch.cat(lengths_list, dim=0)
        all_angles = torch.cat(angles_list, dim=0)
        all_frac_coords = torch.cat(frac_coords_list, dim=0)

        atom_types_list, lengths_list, angles_list, frac_coords_list = [], [], [], []
        start_idx = 0
        for idx, num_atom in enumerate(all_num_atoms.tolist()):
            atom_types = all_atom_types.narrow(0, start_idx, num_atom).numpy()
            lengths = all_lengths[idx].numpy()
            angles = all_angles[idx].numpy()
            frac_coords = all_frac_coords.narrow(0, start_idx, num_atom).numpy()

            atom_types_list.append(atom_types)
            lengths_list.append(lengths)
            angles_list.append(angles)
            frac_coords_list.append(frac_coords)

            start_idx += num_atom
        
        return atom_types_list, lengths_list, angles_list, frac_coords_list