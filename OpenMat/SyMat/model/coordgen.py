import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch_scatter import scatter
from tqdm import tqdm
from .spherenet_light import SphereNetLightDecoder
from .modules import build_mlp
import sys
sys.path.append("..")
from utils import get_pbc_cutoff_graphs, frac_to_cart_coords, cart_to_frac_coords, correct_cart_coords, get_pbc_distances, align_gt_cart_coords


class CoordGen(torch.nn.Module):
    def __init__(self, backbone_params, latent_dim, num_fc_hidden_layers, fc_hidden_dim, num_time_steps, noise_start, noise_end, cutoff, max_num_neighbors, loss_type='per_node', score_upper_bound=None, use_gpu=True, score_norm=None):
        super(CoordGen, self).__init__()
        
        self.backbone = SphereNetLightDecoder(**backbone_params)
        self.fc_score = build_mlp(latent_dim + backbone_params['hidden_channels'], fc_hidden_dim, num_fc_hidden_layers, 1)
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.use_gpu = use_gpu
        if score_norm is not None:
            self.score_norms = torch.from_numpy(score_norm).float()
        
        self._get_noise_params(num_time_steps, noise_start, noise_end)
        self.num_time_steps = num_time_steps
        self.pool = mp.Pool(16)
        self.score_upper_bound = score_upper_bound
        self.loss_type = loss_type
        if use_gpu:
            self.backbone = self.backbone.to('cuda')
            self.fc_score = self.fc_score.to('cuda')
            self.sigmas = self.sigmas.to('cuda')
            self.score_norms = self.score_norms.to('cuda')


    def _get_noise_params(self, num_time_steps, noise_start, noise_end):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_time_steps)
        sigmas = np.exp(log_sigmas)
        self.sigmas = torch.from_numpy(sigmas).float()
        self.sigmas.requires_grad = False


    def _center_coords(self, coords, batch):
        coord_center = scatter(coords, batch, reduce='mean', dim=0)
        return coords - coord_center[batch]


    def forward(self, latents, num_atoms, atom_types, gt_frac_coords, lengths, angles, batch, edge_index=None, to_jimages=None, num_bonds=None):
        num_graphs = batch[-1].item() + 1
        time_steps = torch.randint(0, self.num_time_steps, size=(num_graphs,), device=atom_types.device)
        
        sigmas_per_graph = self.sigmas.index_select(0, time_steps)
        sigmas_per_node = sigmas_per_graph.index_select(0, batch).view(-1,1)
        gt_cart_coords = frac_to_cart_coords(gt_frac_coords, lengths, angles, num_atoms)
        cart_coords_noise = torch.randn_like(gt_cart_coords)
        cart_coords_perturbed = gt_cart_coords + sigmas_per_node * cart_coords_noise
        cart_coords_perturbed = correct_cart_coords(cart_coords_perturbed, lengths, angles, num_atoms, batch)

        if edge_index is None or to_jimages is None or num_bonds is None:
            edge_index, distance_vectors, pbc_offsets = get_pbc_cutoff_graphs(cart_coords_perturbed, lengths, angles, num_atoms, self.cutoff, self.max_num_neighbors)
        else:
            _, distance_vectors, _ = get_pbc_distances(cart_coords_perturbed, edge_index, lengths, angles, to_jimages, num_atoms, num_bonds, True)
        edge_features = self.backbone(atom_types, edge_index, distance_vectors)
        num_multi_edge_per_graph = scatter(torch.ones(size=(edge_index.shape[1],), device=edge_index.device).long(), batch[edge_index[0]], dim_size=num_graphs, reduce='sum')
        latents_per_multi_edge = latents.repeat_interleave(num_multi_edge_per_graph, dim=0)
        edge_features = torch.cat((edge_features, latents_per_multi_edge), dim=1)

        j, i = edge_index
        no_iden_mask = (i != j)
        j, i, edge_features, distance_vectors = j[no_iden_mask], i[no_iden_mask], edge_features[no_iden_mask], distance_vectors[no_iden_mask]
        scores_per_multi_edge = self.fc_score(edge_features)
        
        if edge_index is None or to_jimages is None or num_bonds is None:
            pbc_offsets = pbc_offsets[no_iden_mask]
            aligned_gt_cart_coords = align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms)
            gt_distance_vectors = aligned_gt_cart_coords[i] - aligned_gt_cart_coords[j] - pbc_offsets
            gt_dists_per_multi_edge = torch.linalg.norm(gt_distance_vectors, dim=-1, keepdim=True)
        
        else:
            aligned_gt_cart_coords = align_gt_cart_coords(gt_cart_coords, cart_coords_perturbed, lengths, angles, num_atoms)
            _, gt_distance_vectors, _ = get_pbc_distances(aligned_gt_cart_coords, edge_index, lengths, angles, to_jimages, num_atoms, num_bonds, True)
            gt_distance_vectors = gt_distance_vectors[no_iden_mask]
            gt_dists_per_multi_edge = torch.linalg.norm(gt_distance_vectors, dim=-1, keepdim=True)
        
        perturb_dists_per_multi_edge = torch.linalg.norm(distance_vectors, dim=-1, keepdim=True)
        gt_scores_per_multi_edge = gt_dists_per_multi_edge - perturb_dists_per_multi_edge
        
        score_norms_per_graph = self.score_norms.index_select(0, time_steps)
        score_norms_per_node = score_norms_per_graph.index_select(0, batch).view(-1,1)
        score_norms_per_multi_edge = score_norms_per_node.index_select(0, i).view(-1,1)

        if self.score_upper_bound is not None:
            upper_bound_mask = (gt_scores_per_multi_edge <= self.score_upper_bound * score_norms_per_multi_edge).view(-1)
            j, i = j[upper_bound_mask], i[upper_bound_mask]
            scores_per_multi_edge = scores_per_multi_edge[upper_bound_mask]
            gt_scores_per_multi_edge = gt_scores_per_multi_edge[upper_bound_mask]
            score_norms_per_multi_edge = score_norms_per_multi_edge[upper_bound_mask]
            distance_vectors = distance_vectors[upper_bound_mask]
            perturb_dists_per_multi_edge = perturb_dists_per_multi_edge[upper_bound_mask]
        
        if self.loss_type == 'per_edge':
            score_loss = F.mse_loss(scores_per_multi_edge, gt_scores_per_multi_edge / score_norms_per_multi_edge, reduction='none')
            edge_to_graph = batch[i]
            score_loss = scatter(score_loss, edge_to_graph, dim=0, reduce='mean').mean()
        
        elif self.loss_type == 'per_node':
            num_multi_edges = len(i)
            new_edge_start_mask = torch.logical_or(i[:-1] != i[1:], j[:-1] != j[1:])
            new_edge_start_id = torch.nonzero(new_edge_start_mask).view(-1) + 1
            num_multi_edges_per_edge = torch.cat([new_edge_start_id[0:1], new_edge_start_id[1:] - new_edge_start_id[:-1], num_multi_edges - new_edge_start_id[-1:]])
            multi_edge_to_edge_idx = torch.repeat_interleave(torch.arange(len(num_multi_edges_per_edge), device=num_multi_edges_per_edge.device), num_multi_edges_per_edge)

            scores_per_multi_edge = scores_per_multi_edge * distance_vectors / perturb_dists_per_multi_edge
            scores_per_edge = scatter(scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
            gt_scores_per_multi_edge = gt_scores_per_multi_edge * distance_vectors / perturb_dists_per_multi_edge
            gt_scores_per_edge = scatter(gt_scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
            unique_edge_receiver_index = scatter(i, multi_edge_to_edge_idx, dim=0, reduce='mean').long()
            scores_per_node_pos = scatter(scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')
            gt_scores_per_node_pos = scatter(gt_scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')

            score_loss = F.mse_loss(scores_per_node_pos, gt_scores_per_node_pos / score_norms_per_node, reduction='none')
            score_loss = scatter(score_loss, batch, dim=0, reduce='mean').mean()

        return score_loss


    def get_score_norm(self, sigma):
        sigma_min, sigma_max = self.sigmas[0], self.sigmas[-1]
        sigma_index = (torch.log(sigma) - torch.log(sigma_min)) / (torch.log(sigma_max) - torch.log(sigma_min)) * (len(self.sigmas) - 1)
        sigma_index = torch.round(torch.clip(sigma_index, 0, len(self.sigmas)-1)).long()
        return self.score_norms[sigma_index]
    

    def predict_pos_score(self, latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigma):
        edge_index, distance_vectors, _ = get_pbc_cutoff_graphs(cart_coords, lengths, angles, num_atoms, self.cutoff, self.max_num_neighbors)
        edge_features = self.backbone(atom_types, edge_index, distance_vectors)
        num_multi_edge_per_graph = scatter(torch.ones(size=(edge_index.shape[1],), device=edge_index.device).long(), batch[edge_index[0]], reduce='sum')
        latents_per_multi_edge = latents.repeat_interleave(num_multi_edge_per_graph, dim=0)
        edge_features = torch.cat((edge_features, latents_per_multi_edge), dim=1)

        j, i = edge_index
        no_iden_mask = (i != j)
        j, i, edge_features, distance_vectors = j[no_iden_mask], i[no_iden_mask], edge_features[no_iden_mask], distance_vectors[no_iden_mask]
        dists_per_multi_edge = torch.linalg.norm(distance_vectors, dim=-1, keepdim=True)
        scores_per_multi_edge = self.fc_score(edge_features) * distance_vectors / dists_per_multi_edge

        num_multi_edges = len(i)
        new_edge_start_mask = torch.logical_or(i[:-1] != i[1:], j[:-1] != j[1:])
        new_edge_start_id = torch.nonzero(new_edge_start_mask).view(-1) + 1
        num_multi_edges_per_edge = torch.cat([new_edge_start_id[0:1], new_edge_start_id[1:] - new_edge_start_id[:-1], num_multi_edges - new_edge_start_id[-1:]])
        multi_edge_to_edge_idx = torch.repeat_interleave(torch.arange(len(num_multi_edges_per_edge), device=num_multi_edges_per_edge.device), num_multi_edges_per_edge)
        scores_per_edge = scatter(scores_per_multi_edge, multi_edge_to_edge_idx, dim=0, reduce='mean')
        unique_edge_receiver_index = scatter(i, multi_edge_to_edge_idx, dim=0, reduce='mean').long()
        scores_per_node_pos = scatter(scores_per_edge, unique_edge_receiver_index, dim=0, dim_size=len(batch), reduce='sum')

        score_norm = self.get_score_norm(sigma)
        return scores_per_node_pos / score_norm


    @torch.no_grad()
    def generate(self, latents, num_atoms, atom_types, lengths, angles, noise_start, noise_end, num_gen_steps=50, num_langevin_steps=100, coord_temp=0.01, step_rate=1e-4):
        log_sigmas = np.linspace(np.log(noise_start), np.log(noise_end), num_gen_steps)
        sigmas = np.exp(log_sigmas)
        sigmas = torch.from_numpy(sigmas).float()
        sigmas = torch.cat([torch.zeros([1], device=sigmas.device), sigmas])
        sigmas.requires_grad = False
        
        batch = torch.repeat_interleave(torch.arange(len(num_atoms), device=num_atoms.device), num_atoms)
        frac_coords_init = torch.rand(size=(batch.shape[0], 3), device=lengths.device) - 0.5
        cart_coords_init = frac_to_cart_coords(frac_coords_init, lengths, angles, num_atoms)
        
        cart_coords = cart_coords_init
        for t in tqdm(range(num_gen_steps, 0, -1)):
            current_alpha = step_rate * (sigmas[t] / sigmas[1]) ** 2
            for _ in range(num_langevin_steps):
                scores_per_node_pos = self.predict_pos_score(latents, num_atoms, atom_types, lengths, angles, cart_coords, batch, sigmas[t])
                cart_coords += current_alpha * scores_per_node_pos + (2 * current_alpha).sqrt() * (coord_temp * torch.randn_like(cart_coords))
                cart_coords = correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch)
        
        frac_coords = cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)
        return frac_coords
