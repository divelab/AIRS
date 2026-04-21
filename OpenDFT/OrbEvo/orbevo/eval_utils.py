# Utils for computing properties for one molecule
import torch
import scipy.constants as sc
import numpy as np


def S_prod(x, y, mat_S):
    """
    x: B, T, nbands, norbs
    y: B, T, nbands, norbs
    mat_S: B, norbs, norbs
    return: B, T, nbands
    """
    # prod = x[:, :, :, None, :] @ (mat_S[:, None, None, :, :] + 0j) @ y[:, :, :, :, None].conj()
    prod = x[:, :, :, None, :].conj() @ (mat_S[:, None, None, :, :] + 0j) @ y[:, :, :, :, None]

    # prod = x[:, None, :] @ (mat_S[None, :, :] + 0j) @ y[:, :, None].conj()

    return prod[..., 0, 0]


def normalize_density(coef_0_mol, preds_mol, mat_S):
    """
    coef_0, preds are in the original scales
    mat_S: norbs, nbands
    """

    # preds_normalized = preds_mol

    # norm 1
    coef_t = coef_0_mol.to(torch.complex128) + 0.001 * preds_mol.to(torch.complex128)
    norm_coef = S_prod(coef_t.unsqueeze(0), coef_t.unsqueeze(0), mat_S.unsqueeze(0).double()).squeeze(0)
    preds_normalized = 1000 * (coef_t / norm_coef.real.sqrt().unsqueeze(-1) - coef_0_mol.to(torch.complex128))

    preds_normalized = preds_normalized.to(torch.complex64)

    # preds_normalized = preds_mol

    # norm 2
    # sprod = S_prod(pred['delta_coef_t'], batch_data['coef_0'][:, :, :14, :], batch_data['mat_S'])
    # next_delta_pred = pred['delta_coef_t'] - sprod.real.unsqueeze(-1) * batch_data['coef_0'][:, :, :14, :]

    # no norm
    # next_delta_pred = pred['delta_coef_t']
    # preds_step.append(next_delta_pred)

    return preds_normalized


def get_dipole(coef_0, coef_t, mat_r, occ):
    """
    coef_0: 1, 1, 14, 85
    mat_r: 1, 3, 85, 85
    mat_S: 1, 85, 85
    occ: 14
    """
    mat_r_complex = mat_r.double() + 0j

    coef_t = coef_0.to(torch.complex128) + 0.001 * coef_t.to(torch.complex128)

    # norm_coef = S_prod(coef_t, coef_t, mat_S.double())
    # coef_t = coef_t / norm_coef.sqrt().unsqueeze(-1)
    # next_delta_pred = 1000 * (coef_t / norm_coef.real.sqrt().unsqueeze(-1) - batch_data['coef_0'].to(torch.complex128))

    # (B, T, nbands, 1, 1, norbitals) @ (B, T, 1, ndirections, norbitals, norbitals) @ (B, T,  nbands, 1, norbitals, 1)
    dp = coef_t[:, :, :, None, None, :].matmul(mat_r_complex[:, None, None, :, :, :]).matmul(coef_t.conj()[:, :, :, None, :, None])
    # print(dp.shape, occ.shape)
    dp = torch.sum(dp.squeeze(-1).squeeze(-1) * occ[None, None, :, None], dim=2)
    return dp


def tfn_to_abacus_mol(coef, abacus_to_tfn, atom_types, num_bands):
    """
    coef: T, N, 9, 2
    coef_mol: T, num_bands, num_orbs
    """
    num_atoms = atom_types.shape[0]
    T = coef.shape[0]
    coef_reshape = coef.transpose(-1, -2).reshape(T, num_bands * num_atoms, 18)
    coef_abacus = torch.zeros_like(coef_reshape)
    coef_abacus[:, :, abacus_to_tfn] = coef_reshape

    # num_atoms = data.atom_type.shape[0]
    num_orbitals = {1: 5, # ssp
                    6: 13, # ssppd
                    7: 13,
                    8: 13,
                    9: 13
                    }
    orbital_sizes = [num_orbitals[at.item()] for at in atom_types]

    coef_abacus = coef_abacus.reshape(T, num_bands, num_atoms, 18)
    coef_mol = []

    for i in range(num_atoms):
        coef_mol.append(coef_abacus[:, :, i, :orbital_sizes[i]])

    coef_mol = torch.cat(coef_mol, dim=-1)
    return coef_mol


def abscus_to_tfn_mol(coef_mol, abacus_to_tfn, atom_types):
    # 8, 24, 157
    T = coef_mol.shape[0]
    # coef_reshape = coef_mol.transpose(-1, -2).reshape(T, N_BANDS * N_ATOMS, 18)

    num_atoms = atom_types.shape[0]
    num_orbitals = {1: 5, # ssp
                    6: 13, # ssppd
                    7: 13,
                    8: 13,
                    9: 13
                    }
    orbital_sizes = [num_orbitals[at.item()] for at in atom_types]

    coef_tfn = torch.zeros(T, coef_mol.shape[1], num_atoms, 18, dtype=coef_mol.dtype).to(coef_mol.device)

    coef_split = coef_mol.split(orbital_sizes, dim=-1)
    for i in range(num_atoms):
        coef_tfn[:, :, i, :orbital_sizes[i]] = coef_split[i]

    coef_tfn = coef_tfn[:, :, :, abacus_to_tfn]
    coef_tfn = coef_tfn.reshape(T, coef_mol.shape[1] * num_atoms, 2, 9).transpose(-1, -2)

    return coef_tfn


Freq2eV = sc.h/sc.eV*1e15  # 1/fs to eV
# https://github.com/deepmodeling/abacus-develop/blob/develop/tools/rt-tddft-tools/dipole.py
class Absorption:
    """Calculate Absorption Spectrum under light field"""
    def __init__(self, length, dt) -> None:
        self._indices = np.arange(length)
        self.dt = dt
        # self.dipole_data = dipole_data
        # self.efield_data = efield_data

    def padding(self, data: np.ndarray):
        """Zero padding for FFT

        :params data: (np.ndarray) data to be padded
        """
        #mask part
        Ndim = len(self._indices) * 10
        index = np.linspace(0, Ndim - 1, Ndim)
        t = self._indices * self.dt
        # t=np.arange(len(data)) * self.dt
        #b = 5
        # b=1#
        # b=10
        b = 3.798 # band width 0.5 ev
        mask = np.exp(- b * t / t[-1])
        #padding part
        padding = np.zeros(Ndim - len(data))
        # data_pass = np.zeros(len(self._indices)*11)
        # data_pass[:len(data)] = data * mask
        data_pass = np.concatenate((data * mask, padding))
        return data_pass
    
    def alpha(self,dirc: int, dipole_data, efield_data):
        """Calculate alpha

        :params dirc: (int) 0->X, 1->Y, 2->Z
        """
        #FFT part
        dipole=self.padding(dipole_data[dirc])
        efield=self.padding(efield_data[dirc])
        dipole_fft = np.fft.fft(dipole)
        efield_fft = np.fft.fft(efield)
        alpha = np.abs((dipole_fft/efield_fft).imag)
        return alpha, dipole_fft, efield_fft
    
    def plot_abs(self, fig, ax, directions: list = [0, 1, 2], x_range: list = [], unit: str = 'eV'):
        """Plot Absportion Spectrum under Delta light field in x,y,z directions

        :params fig: (matplotlib.figure.Figure)
        :params ax: (matplotlib.axes.Axes)
        :params directions: (list) 0->X, 1->Y, 2->Z
        :params x_range: (list) range of energies (in unit eV) to plot
        :params unit: (str) 
        """

        assert unit in ['eV', 'nm']
        labels = {0: 'X', 1: 'Y', 2: 'Z'}
        Ndim=len(self._indices)*10
        index=np.linspace(0,Ndim-1,Ndim)
        energies = Freq2eV*index/self.dt/len(index)
        x_data = energies if unit == 'eV' else sc.nu2lambda(
            sc.eV/sc.h*energies)*1e9
        
        #plot the adsorption spectra and output the data
        adsorption_spectra_data = x_data[:, np.newaxis]
        for direc in directions:
            alpha = self.alpha(direc)
            ax.plot(x_data, alpha, label=labels[direc])
            adsorption_spectra_data = np.concatenate((adsorption_spectra_data, alpha[:, np.newaxis]),axis=1)
        np.savetxt('absorpation_spectra.dat', adsorption_spectra_data)

        xlabel = 'Energy (eV)' if unit == 'eV' else 'Wave Length (nm)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Absportion')
        ax.legend()
        if x_range:
            ax.set_xlim(x_range)
            lim_range=index[(x_data>x_range[0])&(x_data<x_range[1])]
            ax.set_ylim([0, 1.2*max(alpha[int(lim_range[0]):int(lim_range[-1])])])
        return fig, ax
