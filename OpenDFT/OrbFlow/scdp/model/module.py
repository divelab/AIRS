import math

import torch
from lightning import LightningModule
from hydra.utils import instantiate
from torch_ema import ExponentialMovingAverage

from scdp.common.utils import scatter
from scdp.model.basis_set import build_transformed_basis_set, aug_etb_for_basis
from scdp.model.gtos import GTOs
from scdp.model.coeff_flow_matching import (
    coeff_norm_denominator,
    num_graphs_in_batch,
    resolve_flow_target_mode,
)
from scdp.model.density_synthesis import apply_gto_exponent_scale_for_mode, resolve_density_synthesis_mode
from scdp.model.hybrid_density import (
    hybrid_density_at_probes,
    parse_grid_shape,
    resolve_density_eval_mode,
)
from scdp.model.utils import get_nmape


class ChgLightningModule(LightningModule):
    """
    Charge density prediction with the probe point method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.construct_orbitals()
        num_neighbors = self.hparams.metadata["avg_num_neighbors"]
        
        self.model = instantiate(
            self.hparams.model, 
            num_neighbors=num_neighbors, 
            expo_trainable=self.hparams.expo_trainable,
            max_n_Ls=self.max_n_Ls,
            max_n_orbitals_per_L=self.max_n_orbitals_per_L
        )
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.train.ema.decay
        )        
        self.distributed = (
            self.hparams.train.trainer.strategy in ("ddp", "ddp_find_unused_parameters_true")
            and self.hparams.train.trainer.devices > 1
        )
        self.register_buffer("scale", torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt())
        self.target_mode = resolve_flow_target_mode(getattr(self.hparams, "target_mode", None))
        self.density_synthesis_mode = resolve_density_synthesis_mode(
            getattr(self.hparams, "density_synthesis_mode", None)
        )
        # Density engine: realspace (default) | hybrid_fft (soft FT+IFFT + sharp RS).
        self.density_eval_mode = resolve_density_eval_mode(
            getattr(self.hparams, "density_eval_mode", None)
        )
        self.hybrid_fft_zeta_max = float(
            getattr(self.hparams, "hybrid_fft_zeta_max", 5.0)
        )
        self.hybrid_fft_phase_sign = float(
            getattr(self.hparams, "hybrid_fft_phase_sign", -1.0)
        )
        self.hybrid_fft_chunk_G = int(getattr(self.hparams, "hybrid_fft_chunk_G", 65536))
        self._sad_provider = None
        # GTO density kernels. Default False so training / val keep the original path.
        self.gto_triton_coeff_eval = False

    def construct_orbitals(self):
        # construct GTOs
        unique_atom_types = self.hparams.metadata['unique_atom_types']
        basis_set = build_transformed_basis_set(
            self.hparams.dft_basis_set,
            required_atom_types=unique_atom_types,
        )
        if self.hparams.dft_wt_aug:
            basis_set = aug_etb_for_basis(
                basis_set, 
                beta=self.hparams.beta, 
                lmax_restriction=self.hparams.lmax_restriction,
                lmax_relax=self.hparams.lmax_relax if 'lmax_relax' in self.hparams else 0,
            )

        synth_mode = resolve_density_synthesis_mode(
            getattr(self.hparams, "density_synthesis_mode", None)
        )
        basis_set = apply_gto_exponent_scale_for_mode(basis_set, synth_mode)

        vbasis = basis_set[self.hparams.vnode_elem]
                    
        if self.hparams.uncontracted:
            for v in basis_set.values():
                v['contraction'] = None
            vbasis['contraction'] = None
            
        gto_dict = {}
        for elem in unique_atom_types:
            # atomic number 0 for virtual nodes.
            if elem == 0:
                gto_dict['0'] = GTOs(**vbasis, cutoff=self.hparams.orb_cutoff)
            else:
                gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=self.hparams.orb_cutoff)
        
        self.register_buffer('unique_atom_types', torch.tensor(unique_atom_types))
        self.register_buffer('n_Ls', torch.tensor([len(gto_dict[str(i)].Ls) for i in unique_atom_types]))
        self.register_buffer('n_orbitals', torch.tensor([gto_dict[str(i)].outdim for i in unique_atom_types]))
        self.gto_dict = torch.nn.ModuleDict(gto_dict)
        
        self.Lmax = max([gto.Lmax for gto in self.gto_dict.values()])
        self.max_n_Ls = max([len(gto.Ls) for gto in self.gto_dict.values()])
        self.max_n_orbitals_per_L = torch.stack(
            [x.n_orbitals_per_L for x in self.gto_dict.values()]).max(dim=0)[0]
        self.max_outdim_per_L = self.max_n_orbitals_per_L * (2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1)
        self.max_outdim = int(self.max_outdim_per_L.sum())
        
        orb_index = torch.zeros(max(unique_atom_types)+1, self.max_outdim, dtype=torch.bool)
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.max_outdim_per_L, dim=0)])        
        L_index = torch.zeros(max(unique_atom_types)+1, self.max_n_Ls, dtype=torch.bool)
        for k, v in gto_dict.items():
            index = torch.cat(
                [torch.arange(offsets[l], offsets[l]+v.outdim_per_L[l]) for l in range(self.Lmax+1)])
            orb_index[int(k), index] = True
            L_index[int(k), :len(v.Ls)] = True
        self.register_buffer('orb_index', orb_index)
        self.register_buffer('L_index', L_index)
            
        self.pbc = self.hparams.pbc

    def _sad_provider_lazy(self):
        if self._sad_provider is None:
            from scdp.data.sad_coeffs_provider import SadCoeffsProvider

            self._sad_provider = SadCoeffsProvider.from_model_hparams(
                self.hparams, self.hparams.metadata
            )
        return self._sad_provider

    def _ensure_sad_coeffs(self, batch) -> None:
        if "sad_coeffs" in batch:
            return
        if bool(getattr(self.hparams, "sad_on_the_fly", False)):
            batch.sad_coeffs = self._sad_provider_lazy().compute_batch(batch)
            return
        raise KeyError(
            "target_mode='res' requires batch['sad_coeffs'] from data.dataset.sad_coeffs_path "
            "or model.sad_on_the_fly=true with model.atomic_chgcar_dir set."
        )

    def _normalized_sad(self, batch, norm: torch.Tensor) -> torch.Tensor:
        sad_raw = batch["sad_coeffs"].to(device=batch.coords.device, dtype=batch.coords.dtype)
        if sad_raw.shape[1] != int(self.max_outdim):
            raise ValueError(
                f"sad_coeffs.shape[1]={sad_raw.shape[1]} but max_outdim={int(self.max_outdim)}"
            )
        return sad_raw / norm

    def predict_coeffs(self, batch):
        """
        Predict GTO coefficients.

        ``target_mode=gt``: full normalized coefficients (default direct baseline).
        ``target_mode=res``: backbone predicts residual Δc; return c_sad + Δc (normalized).
        """
        delta_raw, expo_scaling = self.model(batch)

        n_orbs = self.n_orbitals[
            (batch.atom_types.repeat(len(self.unique_atom_types), 1).T ==
             self.unique_atom_types).nonzero()[:, 1]]
        ng = num_graphs_in_batch(batch)
        batch_n_orbs = scatter(n_orbs, batch.batch, ng)
        delta = delta_raw / batch_n_orbs[batch.batch].sqrt().view(-1, 1).clamp(min=1e-8)

        if expo_scaling is not None:
            # range from 0.5 to 2.0
            expo_scaling = 1.5 / (1 + torch.exp(-expo_scaling + math.log(2))) + 0.5

        if self.target_mode == "res":
            self._ensure_sad_coeffs(batch)
            norm = coeff_norm_denominator(batch, self.n_orbitals, self.unique_atom_types)
            coeffs = delta + self._normalized_sad(batch, norm)
        else:
            coeffs = delta

        return coeffs, expo_scaling
    
    def set_gto_triton_coeff_eval(self, enabled: bool) -> None:
        """Enable Triton GTO pair density (inference/test only; training stays off)."""
        from scdp.model.gtos import apply_gto_triton_coeff_eval

        apply_gto_triton_coeff_eval(self, enabled)
    
    def orbital_inference(self, batch, coeffs, expo_scaling, n_probe, probe_coords):
        """
        Compute chg values at given probe points using <coeffs>.
        Inputs:
            - batch: batch (bsz B) object, N atoms
            - coeffs: orbital coefficients (N, max_orbital_outdim)
            - n_probes: number of probes for each batch (B,)
            - probe_coords: probe coordinates (M, 3)
        Outputs:
            - orbitals: chg values at probe points, (M,)
        """
        if self.density_eval_mode == "hybrid_fft":
            return self._orbital_inference_hybrid_fft(
                batch, coeffs, expo_scaling, n_probe, probe_coords
            )
        return self._orbital_inference_realspace(
            batch, coeffs, expo_scaling, n_probe, probe_coords
        )

    def _orbital_inference_realspace(self, batch, coeffs, expo_scaling, n_probe, probe_coords):
        unique_atom_types = torch.unique(batch.atom_types)
        pred = torch.zeros(probe_coords.shape[0], device=coeffs.device, dtype=coeffs.dtype)
        ng = num_graphs_in_batch(batch)
        for i in unique_atom_types:
            n_atom_i = scatter(
                (batch.atom_types == i).long(),
                batch.batch,
                ng,
            )
            orb_index = self.orb_index[i.item()]
            if expo_scaling is not None:
                L_index = self.L_index[i.item()]
            pred += self.gto_dict[str(i.item())](
                probe_coords=probe_coords,
                atom_coords=batch.coords[batch.atom_types == i],
                n_probes=n_probe,
                n_atoms=n_atom_i,
                coeffs=coeffs[batch.atom_types == i][:, orb_index],
                expo_scaling=expo_scaling[batch.atom_types == i][:, L_index]
                if expo_scaling is not None
                else None,
                pbc=self.pbc,
                cell=batch.cell,
                density_synthesis_mode=self.density_synthesis_mode,
                triton_coeff_eval=bool(getattr(self, "gto_triton_coeff_eval", False)),
            )
        pred = pred * self.scale
        return pred

    def _graph_grid_meta(self, batch, graph_idx: int):
        """Return (cell[3,3], grid_shape, grid_convention) for one graph in a batch."""
        cell = batch.cell
        if cell.dim() == 3:
            cell_g = cell[graph_idx]
        elif cell.dim() == 2:
            if num_graphs_in_batch(batch) != 1:
                raise ValueError("batched hybrid_fft requires batch.cell shaped (B,3,3)")
            cell_g = cell
        else:
            raise ValueError(f"unexpected batch.cell shape {tuple(cell.shape)}")

        if not hasattr(batch, "grid_size"):
            raise KeyError(
                "density_eval_mode=hybrid_fft requires batch.grid_size "
                "(nx,ny,nz) from the LMDB graph"
            )
        gs = batch.grid_size
        if torch.is_tensor(gs) and gs.dim() >= 2:
            gs_g = gs[graph_idx]
        else:
            if num_graphs_in_batch(batch) != 1 and torch.is_tensor(gs) and gs.dim() == 1:
                raise ValueError("batched hybrid_fft requires per-graph grid_size")
            gs_g = gs
        grid_shape = parse_grid_shape(gs_g)

        conv = getattr(batch, "grid_convention", "corner")
        if isinstance(conv, (list, tuple)):
            conv = conv[graph_idx]
        elif torch.is_tensor(conv):
            conv = str(conv[graph_idx].item()) if conv.dim() > 0 else str(conv.item())
        conv = str(conv)
        return cell_g, grid_shape, conv

    def _orbital_inference_hybrid_fft(self, batch, coeffs, expo_scaling, n_probe, probe_coords):
        if expo_scaling is not None:
            raise NotImplementedError(
                "density_eval_mode=hybrid_fft does not support expo_scaling; "
                "set expo_trainable=false"
            )
        if self.density_synthesis_mode != "linear":
            raise NotImplementedError(
                "density_eval_mode=hybrid_fft currently supports "
                "density_synthesis_mode='linear' only"
            )
        if not self.pbc:
            raise NotImplementedError(
                "density_eval_mode=hybrid_fft is intended for PBC crystals "
                "(model.pbc=true)"
            )

        ng = num_graphs_in_batch(batch)
        if not torch.is_tensor(n_probe):
            n_probe_t = torch.full((ng,), int(n_probe), device=probe_coords.device, dtype=torch.long)
        else:
            n_probe_t = n_probe.reshape(-1).to(device=probe_coords.device, dtype=torch.long)
            if n_probe_t.numel() == 1 and ng > 1:
                n_probe_t = n_probe_t.expand(ng)
        if int(n_probe_t.sum().item()) != int(probe_coords.shape[0]):
            raise ValueError(
                f"n_probe sum {int(n_probe_t.sum())} != probe_coords {probe_coords.shape[0]}"
            )

        outs = []
        probe_off = 0
        for g in range(ng):
            atom_mask = batch.batch == g
            n_p = int(n_probe_t[g].item())
            probes_g = probe_coords[probe_off : probe_off + n_p]
            probe_off += n_p
            cell_g, grid_shape, conv = self._graph_grid_meta(batch, g)
            rho_g = hybrid_density_at_probes(
                cell=cell_g,
                grid_shape=grid_shape,
                grid_convention=conv,
                atom_coords=batch.coords[atom_mask],
                atom_types=batch.atom_types[atom_mask],
                coeffs=coeffs[atom_mask],
                probe_coords=probes_g,
                n_probe=n_p,
                gto_dict=self.gto_dict,
                orb_index=self.orb_index,
                zeta_soft_max=self.hybrid_fft_zeta_max,
                phase_sign=self.hybrid_fft_phase_sign,
                chunk_G=self.hybrid_fft_chunk_G,
                pbc=self.pbc,
                density_synthesis_mode=self.density_synthesis_mode,
            )
            outs.append(rho_g)
        pred = torch.cat(outs, dim=0) * self.scale
        return pred
    
    def forward(self, batch):
        coeffs, expo_scaling = self.predict_coeffs(batch)
        pred = self.orbital_inference(batch, coeffs, expo_scaling, batch.n_probe, batch.probe_coords)        
        
        target = batch.chg_labels
        if self.hparams.criterion == 'mse':
            loss = (pred / self.scale - target / self.scale).pow(2).mean()
        else:
            loss = (pred / self.scale - target / self.scale).abs().mean()
            
        return loss, pred, batch.chg_labels, coeffs, expo_scaling
    
    def training_step(self, batch, batch_idx):        
        loss, _, _, coeffs, scaling = self(batch)
        self.log_dict({
            "loss/train": loss,
            }, 
            batch_size=batch["cell"].shape[0], 
            sync_dist=self.distributed
        )   
        
        if scaling is not None:
            self.log_dict({
                "trainer/scaling_mean": scaling.mean(),
                "trainer/scaling_std": scaling.std()
                }, 
                batch_size=batch["cell"].shape[0], 
                sync_dist=self.distributed
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        nmape = get_nmape(
            pred, target, 
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe)
        ).mean()
        self.log_dict({
            "loss/val": loss,
            "nmape/val": nmape
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        nmape = get_nmape(
            pred, target, 
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe)
        ).mean()
        self.log_dict({
            "loss/test": loss,
            "nmape/test": nmape
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
            )
        return loss

    def configure_optimizers(self):
        opt = instantiate(
            self.hparams.train.optim,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=opt)
        
        if 'lr_schedule_freq' in self.hparams.train:
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.hparams.train.lr_schedule_freq,
                'monitor': self.hparams.train.monitor.metric
            }
            
        return {"optimizer": opt, "lr_scheduler": scheduler, 'monitor': self.hparams.train.monitor.metric}
    
    def on_fit_start(self):
        self.ema.to(self.device)
        
    def on_save_checkpoint(self, checkpoint):
        with self.ema.average_parameters():
            checkpoint["ema_state_dict"] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        try:
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
        except Exception as e:
            print(e)
            print("Failed to load EMA state dict. Please make sure this was intended.")
            
            
    def on_validation_epoch_start(self):
        self.ema.store()
        self.ema.copy_to(self.parameters())
        
    def on_validation_epoch_end(self):
        self.ema.restore()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step(self.trainer.callback_metrics[self.hparams.train.monitor.metric])
    
    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.parameters())
        
    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), float('inf'), norm_type=2.0)
        self.log('trainer/grad_norm', total_norm)