import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .spherenet import SphereNetEncoder
from .modules import build_mlp
from .coordgen import CoordGen


class MatGen(torch.nn.Module):
    def __init__(self, enc_backbone_params, dec_backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, max_num_atoms, max_atomic_num,
        num_time_steps, noise_start, noise_end, cutoff, max_num_neighbors, logvar_clip=6.0, mu_clip=14.0, 
        use_gpu=True, lattice_scale=True, pred_prop=False, use_multi_latent=False, coord_loss_type='per_node', score_norm=None):

        super(MatGen, self).__init__()
        self.lattice_scale = lattice_scale
        self.encoder_backbone = SphereNetEncoder(**enc_backbone_params)
        
        latent_in_dim = enc_backbone_params['out_channels']
        latent_out_dim = 2 * latent_dim if use_multi_latent else latent_dim
        self.fc_mu = nn.Linear(latent_in_dim, latent_out_dim)
        self.fc_var = nn.Linear(latent_in_dim, latent_out_dim)
        self.fc_elem_type = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_atomic_num)
        self.fc_elem_num = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_num_atoms)
        self.fc_lattice_mu = build_mlp(6, fc_hidden_dim, num_fc_hidden_layers, latent_dim)
        self.fc_lattice_log_var = build_mlp(6, fc_hidden_dim, num_fc_hidden_layers, latent_dim)
        self.fc_lattice = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, 6)
        self.fc_elem_type_num = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_num_atoms)
        
        self.coordgen = CoordGen(dec_backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps,
            noise_start, noise_end, cutoff, max_num_neighbors, loss_type=coord_loss_type, use_gpu=use_gpu, score_norm=score_norm)
        
        if use_gpu:
            self.encoder_backbone = self.encoder_backbone.to('cuda')
            self.fc_mu = self.fc_mu.to('cuda')
            self.fc_var = self.fc_var.to('cuda')
            self.fc_elem_type = self.fc_elem_type.to('cuda')
            self.fc_elem_num = self.fc_elem_num.to('cuda')
            self.fc_lattice_mu = self.fc_lattice_mu.to('cuda')
            self.fc_lattice_log_var = self.fc_lattice_log_var.to('cuda')
            self.fc_lattice = self.fc_lattice.to('cuda')
            self.fc_elem_type_num = self.fc_elem_type_num.to('cuda')
        
        self.elem_emb = self.encoder_backbone.init_e.emb
        
        if pred_prop:
            self.fc_prop = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, 1)
            if use_gpu:
                self.fc_prop = self.fc_prop.to('cuda')
        
        self.max_atomic_num = max_atomic_num
        self.latent_dim = latent_dim
        self.logvar_clip = logvar_clip
        self.mu_clip = mu_clip
        self.use_gpu = use_gpu
        self.prop_normalizer = None
        self.lattice_normalizer = None
        self.use_multi_latent = use_multi_latent

    
    def encode(self, data_batch, temp=[0.5, 0.5, 0.5]):
        hidden = self.encoder_backbone(data_batch)
        mu, log_var = self.fc_mu(hidden), self.fc_var(hidden)
        mu.clip_(min=-self.mu_clip, max=self.mu_clip)
        log_var.clip_(max=self.logvar_clip)

        if not self.use_multi_latent:
            std = torch.exp(0.5 * log_var) * temp[0]
            latent = torch.randn_like(std) * std + mu
            latent_comp, latent_pos = latent, latent
        else:
            std_comp = torch.exp(0.5 * log_var[:, :self.latent_dim]) * temp[0]
            latent_comp = torch.randn_like(std_comp) * std_comp + mu[:, :self.latent_dim]
            std_pos = torch.exp(0.5 * log_var[:, self.latent_dim:]) * temp[1]
            latent_pos = torch.randn_like(std_pos) * std_pos + mu[:, self.latent_dim:]
        
        lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.transform(lengths_angles)
        mu_lattice, log_var_lattice = self.fc_lattice_mu(lengths_angles), self.fc_lattice_log_var(lengths_angles)
        mu_lattice.clip_(min=-self.mu_clip, max=self.mu_clip)
        log_var_lattice.clip_(max=self.logvar_clip)
        std_lattice = torch.exp(0.5 * log_var_lattice) * temp[2]
        latent_lattice = torch.randn_like(std_lattice) * std_lattice + mu_lattice
        
        return mu, log_var, mu_lattice, log_var_lattice, latent_comp, latent_pos, latent_lattice


    def __topk_mask(self, input, k):
        sorted, _ = torch.sort(input, dim=-1, descending=True)
        thres = sorted[torch.arange(input.shape[0], device=input.device), k-1].view(-1, 1)
        return (input >= thres).long().to(input.device)
    

    def __match_composition(self, elem_type_topk, target_elem_type, elem_num_pred, target_elem_num, num_elem_per_mat):
        idx = 0
        elem_type_match_num, elem_num_match_num, match_num = 0, 0, 0
        for i in range(len(num_elem_per_mat)):
            idx2 = idx + num_elem_per_mat[i].long()
            elem_type_match = (elem_type_topk[i] == target_elem_type[i]).min(-1).values
            elem_num_match = (elem_num_pred[idx : idx2] == target_elem_num[idx : idx2]).min(-1).values
            elem_type_match_num += elem_type_match
            elem_num_match_num += elem_num_match
            match_num += elem_type_match * elem_num_match
            idx = idx2
        return elem_type_match_num, elem_num_match_num, match_num


    def forward(self, data_batch, temp=[0.5, 0.5, 0.5], eval=False):
        mu, log_var, mu_lattice, log_var_lattice, latent_comp, latent_pos, latent_lattice = self.encode(data_batch, temp)
        
        pred_elem_type_num = self.fc_elem_type_num(latent_comp) # [256, 20] 
        elem_type_num = pred_elem_type_num.argmax(dim=-1) + 1 # [256]
        pred_elem_type = self.fc_elem_type(latent_comp) # [256, 100]
        elem_type_topk = self.__topk_mask(pred_elem_type, elem_type_num).to('cuda') # select top k elem type -> [256, 100] 1 or 0

        elem_types = data_batch.atom_types - 1
        elem_types_one_hot = F.one_hot(elem_types, num_classes=self.max_atomic_num)
        elem_num = scatter(elem_types_one_hot, data_batch.batch, dim=0, reduce='sum')
        nonzero_mask = elem_num > 0
        target_elem_type = nonzero_mask.long()

        num_elem_per_mat = target_elem_type.sum(dim=-1)
        latent_repeat = torch.repeat_interleave(latent_comp, num_elem_per_mat, dim=0)
        elem_type_index = torch.nonzero(elem_num)
        batch_index, elem_type_batch = elem_type_index[:, 0], elem_type_index[:, 1]
        pred_elem_num = self.fc_elem_num(latent_repeat * self.elem_emb(elem_type_batch))
        elem_num_pred = pred_elem_num.argmax(dim=-1)
        target_elem_num = torch.masked_select(elem_num, nonzero_mask) - 1

        pred_lengths_angles = self.fc_lattice(latent_lattice)
        
        loss_dict = {}
        
        loss_dict['kld_loss'] = torch.mean(-0.5 * torch.sum(1.0 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0) + \
            torch.mean(-0.5 * torch.sum(1.0 + log_var_lattice - mu_lattice ** 2 - log_var_lattice.exp(), dim=1), dim=0)
        if self.use_multi_latent:
            loss_dict['kld_loss1'] = torch.mean(-0.5 * torch.sum(1.0 + log_var[:, :self.latent_dim] - mu[:, :self.latent_dim] ** 2 - log_var[:, :self.latent_dim].exp(), dim=1), dim=0)
            loss_dict['kld_loss2'] = torch.mean(-0.5 * torch.sum(1.0 + log_var[:, self.latent_dim:] - mu[:, self.latent_dim:] ** 2 - log_var[:, self.latent_dim:].exp(), dim=1), dim=0)
            loss_dict['kld_loss3'] = torch.mean(-0.5 * torch.sum(1.0 + log_var_lattice - mu_lattice ** 2 - log_var_lattice.exp(), dim=1), dim=0)
        
        elem_type_num_loss = F.cross_entropy(pred_elem_type_num, num_elem_per_mat - 1, reduction='mean')
        elem_type_loss = F.binary_cross_entropy_with_logits(pred_elem_type, target_elem_type.float(), reduction='mean')
        elem_num_loss = scatter(F.cross_entropy(pred_elem_num, target_elem_num, reduction='none'), batch_index, reduce='mean').mean()
        loss_dict['elem_type_num_loss'] = elem_type_num_loss
        loss_dict['elem_type_loss'] = elem_type_loss
        loss_dict['elem_num_loss'] = elem_num_loss

        if eval:
            loss_dict['total_elem_type_num_num'] = len(elem_type_num)
            loss_dict['elem_type_num_correct'] = elem_type_num.eq(num_elem_per_mat).sum().cpu().detach().item()

            loss_dict['total_elem_type_num'] = target_elem_type.shape[0] * target_elem_type.shape[1]
            loss_dict['pos_elem_type_num'] = nonzero_mask.sum().cpu().detach().item()
            loss_dict['elem_type_correct'] = elem_type_topk.eq(target_elem_type).sum().cpu().detach().item()
            loss_dict['pos_elem_type_correct'] = torch.logical_and(elem_type_topk.eq(target_elem_type), nonzero_mask).sum().cpu().detach().item()

            loss_dict['elem_num_num'] = len(target_elem_num)
            loss_dict['elem_num_correct'] = pred_elem_num.argmax(dim=-1).eq(target_elem_num).sum().cpu().detach().item()

            loss_dict['composition_correct'] = self.__match_composition(elem_type_topk, target_elem_type, elem_num_pred, target_elem_num, num_elem_per_mat)       
        
        target_lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            target_lengths_angles = self.lattice_normalizer.transform(target_lengths_angles)
        loss_dict['lattice_loss'] = F.mse_loss(pred_lengths_angles, target_lengths_angles)
        
        loss_dict['coord_loss'] = self.coordgen(latent_pos, data_batch.num_atoms, data_batch.atom_types-1, data_batch.frac_coords, data_batch.lengths, data_batch.angles, data_batch.batch)
        
        return loss_dict


    @torch.no_grad()
    def generate(self, num_gen, temperature=[0.5, 0.5, 0.5, 0.01], coord_noise_start=0.01, coord_noise_end=10, coord_num_diff_steps=50, coord_num_langevin_steps=100, coord_step_rate=1e-4):
        if not self.use_multi_latent:
            if self.use_gpu:
                prior = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[0] * torch.ones([self.latent_dim]).cuda())
            else:
                prior = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[0] * torch.ones([self.latent_dim]))
            
            latent = prior.sample([num_gen])
            latent_comp, latent_pos, latent_lattice = latent, latent, latent
        else:
            if self.use_gpu:
                prior1 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[0] * torch.ones([self.latent_dim]).cuda())
                prior2 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[1] * torch.ones([self.latent_dim]).cuda())
                prior3 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]).cuda(), temperature[2] * torch.ones([self.latent_dim]).cuda())
            else:
                prior1 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[0] * torch.ones([self.latent_dim]))
                prior2 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[1] * torch.ones([self.latent_dim]))
                prior3 = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), temperature[2] * torch.ones([self.latent_dim]))
            latent_comp, latent_pos, latent_lattice = prior1.sample([num_gen]), prior2.sample([num_gen]), prior3.sample([num_gen])
        
        pred_elem_type_num = self.fc_elem_type_num(latent_comp)
        elem_type_num = pred_elem_type_num.argmax(dim=-1) + 1
        pred_elem_type = self.fc_elem_type(latent_comp)
        elem_type_topk = self.__topk_mask(pred_elem_type, elem_type_num).to('cuda')
        num_elem_per_mat = elem_type_topk.sum(dim=-1)
        
        latent_repeat = torch.repeat_interleave(latent_comp, num_elem_per_mat, dim=0)
        elem_type_index = torch.nonzero(elem_type_topk)
        batch_index, elem_type_batch = elem_type_index[:, 0], elem_type_index[:, 1]
        pred_elem_num = self.fc_elem_num(latent_repeat * self.elem_emb(elem_type_batch)).argmax(dim=-1) + 1
        
        atom_types = torch.repeat_interleave(elem_type_batch + 1, pred_elem_num, dim=0)
        num_atoms = scatter(pred_elem_num, batch_index, dim=0, reduce='sum')
        
        lengths_angles = self.fc_lattice(latent_lattice)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.inverse_transform(lengths_angles)
        lengths, angles = lengths_angles[:, :3], lengths_angles[:, 3:]
        if self.lattice_scale:
            lengths = lengths * num_atoms.view(-1, 1).float()**(1/3)
        
        frac_coords = self.coordgen.generate(latent_pos, num_atoms, atom_types - 1, lengths, angles, coord_noise_start, coord_noise_end, coord_num_diff_steps, 
          coord_num_langevin_steps, temperature[-1], coord_step_rate)

        return num_atoms, atom_types, lengths, angles, frac_coords
        