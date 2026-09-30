import os
import io
import pickle
import tempfile
import json
import torch
import numpy as np

import ase
from ase.calculators.vasp import VaspChargeDensity
from pyrho.charge_density import ChargeDensity

from scdp.common.utils import cell_heights

def read_pmg_pkl(fpath):
    with open(fpath, "rb") as f:
        chgcar = pickle.load(f)
    chgcar = ChargeDensity.from_pmg(chgcar)
    struct = chgcar.structure
    atomic_numbers = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
    cart_coords = torch.tensor(struct.cart_coords).float()
    cell = torch.tensor(struct.lattice.matrix).float()
    chg_density = torch.tensor(chgcar.normalized_data["total"]).float()
    return atomic_numbers, cart_coords, cell, chg_density, None, struct


def read_vasp(filecontent, read_spin=False):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="temp")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    try:
        density = vasp_charge.chg[-1]  # separate density
        if read_spin:
            if len(vasp_charge.chgdiff) != 0:
                spin_density = vasp_charge.chgdiff[-1]
            else:
                # assume non-spin-polarized if there's no spin density data
                spin_density = np.zeros_like(density)
            density = np.stack([density, spin_density], axis=-1)
    except IndexError as e:
        print(e, f"\nFileconents of {filecontent} do not contain chg field")
    atoms = vasp_charge.atoms[-1]  # separate atom positions
    atomic_numbers = torch.from_numpy(atoms.numbers).long()
    cart_coords = torch.from_numpy(atoms.positions).float()
    cell = torch.from_numpy(atoms.cell.array).float()
    chg_density = torch.from_numpy(density).float()
    return atomic_numbers, cart_coords, cell, chg_density, None, atoms


def read_cube(filecontent):
    textbuf = io.StringIO(filecontent.decode())
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr**3
    chg_density = torch.from_numpy(cube["data"])
    atoms = cube["atoms"]
    atomic_numbers = torch.from_numpy(atoms.numbers).long()
    cart_coords = torch.from_numpy(atoms.positions).float()
    cell = torch.from_numpy(cube["cell"]).float()
    origin = torch.from_numpy(origin).float()
    return atomic_numbers, cart_coords, cell, chg_density, origin, None

def read_2d_tensor(s):
    return torch.FloatTensor([[float(x) for x in line] for line in s])

def read_json(filecontent):
    data = json.loads(filecontent.decode())
    scale = float(data['vector'][0][0])
    cell = read_2d_tensor(data['lattice'][0]) * scale
    elements = data['elements'][0]
    n_atoms = [int(s) for s in data['elements_number'][0]]

    tot_atoms = sum(n_atoms)
    atom_coords = read_2d_tensor(data['coordinates'][0])
    atom_types = torch.empty(tot_atoms, dtype=torch.long)
    idx = 0
    for elem, n in zip(elements, n_atoms):
        atom_types[idx:idx + n] = ase.data.atomic_numbers[elem]
        idx += n
    atom_coords = atom_coords @ cell
    
    shape = [int(s) for s in data['FFTgrid'][0]]

    n_grid = shape[0] * shape[1] * shape[2]
    n_line = (n_grid + 9) // 10
    # pylint: disable=E1136
    density = torch.FloatTensor([
        float(s) if not s.startswith('*') else 0.
        for line in data['chargedensity'][0][:n_line]
        for s in line
    ]).view(-1)[:n_grid]
    volume = torch.linalg.det(cell).abs()
    density = density / volume
    density = density.view(shape[2], shape[1], shape[0]).transpose(0, 2).contiguous()
    
    return atom_types, atom_coords, cell, density, None, None

def calculate_grid_pos(
    density: torch.Tensor,
    cell: torch.Tensor,
    origin=None,
    grid_convention: str = "corner",
) -> torch.Tensor:
    """
    Cartesian probe coordinates for a regular density grid.

    grid_convention:
      - ``corner`` (default): fractional coords k/n for k=0..n-1 (QM9/VASP LMDB).
      - ``gpwno``: fractional coords k/n for k=1..n (GPWNO / ELECTRA MD:
        ``linspace(L/n, L, n)`` on cubic cells). Shifts voxel centers by one
        spacing h=L/n relative to ``corner``.
    """
    lx, ly, lz = density.shape
    if grid_convention == "corner":
        frac_x = torch.arange(lx, device=cell.device, dtype=cell.dtype) / lx
        frac_y = torch.arange(ly, device=cell.device, dtype=cell.dtype) / ly
        frac_z = torch.arange(lz, device=cell.device, dtype=cell.dtype) / lz
    elif grid_convention == "gpwno":
        frac_x = torch.arange(1, lx + 1, device=cell.device, dtype=cell.dtype) / lx
        frac_y = torch.arange(1, ly + 1, device=cell.device, dtype=cell.dtype) / ly
        frac_z = torch.arange(1, lz + 1, device=cell.device, dtype=cell.dtype) / lz
    else:
        raise ValueError(
            f"unknown grid_convention {grid_convention!r}; use 'corner' or 'gpwno'"
        )
    grid_pos = torch.meshgrid(frac_x, frac_y, frac_z, indexing="ij")
    grid_pos = torch.stack(grid_pos, 3)
    grid_pos = torch.matmul(grid_pos, cell)
    if origin is not None:
        grid_pos = grid_pos + origin
    return grid_pos


def make_graph(
    cell,
    coords_src,
    coords_dst,
    cutoff=4.0,
    disable_pbc=False,
    disable_sc=False,
    max_neigh=None,
):
    """
    compute edges between coords_src and coords_dst under PBC.
    coords_dst is meant to be used for the probes too, which is dense. Therefore, 
    This function may become memory intensive. 
    <max_n_node> is the maximum number of nodes to split <coords_dst>
    into multiple chunks so that we can run this on GPU.
    
    Args:
    """
    if disable_pbc:
        repeat_offsets = torch.tensor([[0, 0, 0]]).to(cell.device)
        shifts = torch.tensor([[0, 0, 0]], dtype=float).to(cell.device)
    else:
        n_rep = torch.ceil(cutoff / (cell_heights(cell) + 1e-12))
        _rep = lambda dim: torch.arange(-n_rep[dim], n_rep[dim] + 1)
        repeat_offsets = torch.tensor(
            [(x, y, z) for x in _rep(0) for y in _rep(1) for z in _rep(2)]
        ).to(cell.device)
        shifts = torch.matmul(repeat_offsets, cell)
        
    # m x supercell x 3. minus shifts because atoms are sources
    shifted = coords_src.unsqueeze(1) - shifts.unsqueeze(0)
    # m x 1 x 1 x 3 - 1 x n x supercell x 3
    diff_vec = coords_dst.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # m x n x supercell
    dist_sqr = diff_vec.pow(2).sum(dim=-1)
    mask = torch.le(dist_sqr, cutoff * cutoff)

    if dist_sqr.shape[0] == dist_sqr.shape[1] and disable_sc:
        mask = torch.logical_and(mask, torch.gt(dist_sqr, 1e-6))
        # remove self loops from topk
        dist_sqr[torch.le(dist_sqr, 1e-6)] = 100.
    
    if max_neigh is not None:
        flat_dist = dist_sqr.flatten(1, 2)
        k = min(max_neigh, flat_dist.shape[1])
        _, topk_indices = flat_dist.topk(k, dim=1, largest=False)
        topk_mask = torch.zeros_like(dist_sqr, dtype=torch.bool).flatten(1,2)
        topk_mask.scatter_(1, topk_indices, True)
        topk_mask = topk_mask.view(dist_sqr.shape)
        mask = torch.logical_and(mask, topk_mask)
            
    # dst - src in diff calc  
    idx_dst, idx_src, shift_idx = torch.nonzero(
        mask, as_tuple=True
    )
    unit_shifts = repeat_offsets[shift_idx]
    shifts = torch.matmul(unit_shifts.float(), cell)

    return idx_src, idx_dst, shifts, unit_shifts