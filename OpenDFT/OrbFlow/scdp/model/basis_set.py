"""
Util functions for using pre-defined DFT basis sets and even-tempered augmentation.
"""
import itertools
import logging
from typing import Dict, Optional, Sequence

import basis_set_exchange as bse
import numpy as np

from scdp.common.constants import orbital_config

logger = logging.getLogger(__name__)

# Per-element fallback when the primary basis (e.g. def2-qzvppd) omits Z.
SUPPLEMENTAL_BASIS_FALLBACK = ("def2-tzvp", "CRENBL")


def get_basis_set(name='def2-universal-JKFIT'):
    """
    get basis set from BSE.
    """
    basis = bse.get_basis(name)
    all_elems = basis['elements'].keys()
    for k in all_elems:
        for s in basis['elements'][k]['electron_shells']:
            s['angular_momentum'] = s['angular_momentum'][0]
            for i in range(len(s['coefficients'])):
                s['coefficients'][i] = [float(x) for x in s['coefficients'][i]]
            s['exponents'] = [float(x) for x in s['exponents']]
    return basis


def get_basis_set_merged(
    primary: str = "def2-qzvppd",
    supplemental: Sequence[str] = SUPPLEMENTAL_BASIS_FALLBACK,
    required_elements: Optional[Sequence[int]] = None,
) -> dict:
    """
    Load a BSE basis with per-element fallback for elements missing from primary.

    For each required Z, libraries are tried in order: primary, then each entry in
    supplemental. The first library that defines Z is used as the ETB seed source.
    """
    primary_basis = get_basis_set(primary)
    if not required_elements:
        return primary_basis

    merged_elements = dict(primary_basis["elements"])
    missing = sorted({int(z) for z in required_elements if int(z) > 0 and str(z) not in merged_elements})
    if not missing:
        return primary_basis

    fallback_bases = {name: get_basis_set(name) for name in supplemental}
    sources: Dict[int, str] = {}
    for z in missing:
        z_key = str(z)
        for name in supplemental:
            if z_key in fallback_bases[name]["elements"]:
                merged_elements[z_key] = fallback_bases[name]["elements"][z_key]
                sources[z] = name
                break
        else:
            tried = (primary,) + tuple(supplemental)
            raise KeyError(f"No basis found for element Z={z} in {tried}")

    for z, name in sources.items():
        logger.info("Basis for Z=%d from %s (missing in %s)", z, name, primary)

    merged = dict(primary_basis)
    merged["elements"] = merged_elements
    return merged


def build_transformed_basis_set(
    dft_basis_set: str = "def2-qzvppd",
    required_atom_types: Optional[Sequence[int]] = None,
    supplemental: Sequence[str] = SUPPLEMENTAL_BASIS_FALLBACK,
) -> dict:
    """Merged BSE basis (with supplements) in the torch GTO format."""
    return transform_basis_set(
        get_basis_set_merged(
            primary=dft_basis_set,
            supplemental=supplemental,
            required_elements=required_atom_types,
        )
    )


def transform_basis_set(basis):
    """
    transform BSE basis set for the torch GTO implementation in this repo.
    """
    all_elems = basis['elements'].keys()
    basis_set = {}
    for i in all_elems:
        basis_set[int(i)] = {'Ls':[], 'coeffs': [], 'expos': [], 'contraction': []}
        cur = 0
        for b in basis['elements'][i]['electron_shells']:
            nexpo = len(b['exponents'])
            for c in range(len(b['coefficients'])):
                basis_set[int(i)]['Ls'].extend([b['angular_momentum']] * nexpo)
                basis_set[int(i)]['coeffs'].extend(b['coefficients'][c])
                basis_set[int(i)]['expos'].extend(b['exponents'])
                basis_set[int(i)]['contraction'].extend([cur] * nexpo)
                cur += 1
    return basis_set

def concat_basis(basis_set_list):
    """
    concatenate a list of basis sets.
    """
    Ls, coeffs, expos, contraction = [], [], [], []
    cur_pt = 0
    for b in basis_set_list:
        Ls.extend(b['Ls'])
        coeffs.extend(b['coeffs'])
        expos.extend(b['expos'])
        cur_contraction = [c + cur_pt for c in b['contraction']]
        cur_pt = max(cur_contraction) + 1
        contraction.extend(cur_contraction)
    return {'Ls':Ls, 'coeffs': coeffs, 'expos': expos, 'contraction': contraction}

def basis_to_g2g(basis):
    """
    transform the basis set format in this codebase to the the gau2grid format.
    """
    out_basis = []
    c = 0
    head = True
    for i in range(len(basis['Ls'])):
        if basis['contraction'][i] == c:
            if head:
                cur_basis = {
                    'L': basis['Ls'][i], 
                    'expos': [basis['expos'][i]], 
                    'coeffs': [basis['coeffs'][i]]
                }
            else:
                cur_basis['expos'].append(basis['expos'][i])
                cur_basis['coeffs'].append(basis['coeffs'][i])
            head = False
        else:
            out_basis.append(cur_basis)
            cur_basis = {
                    'L': basis['Ls'][i], 
                    'expos': [basis['expos'][i]], 
                    'coeffs': [basis['coeffs'][i]]
                }
            c = basis['contraction'][i]
    out_basis.append(cur_basis)
    return out_basis
    
def basis_to_pyscf(basis):
    """
    transform the basis set format in this codebase to pyscf format.
    """
    out_basis = []
    c = 0
    head = True
    for i in range(len(basis['Ls'])):
        if basis['contraction'][i] == c:
            if head:
                cur_basis = [basis['Ls'][i]]
            cur_basis = cur_basis + [[basis['expos'][i], basis['coeffs'][i]]]
            head = False
        else:
            out_basis.append(cur_basis)
            cur_basis = [basis['Ls'][i], [basis['expos'][i], basis['coeffs'][i]]]
            c = basis['contraction'][i]
    out_basis.append(cur_basis)
    return out_basis

def basis_from_pyscf(basis):
    """
    transform the basis set format in pyscf to the format used in this codebase.
    """
    Ls, expos, coeffs, contraction = [], [], [], []
    for c, b in enumerate(basis):
        for i in range(len(b)-1):
            Ls.append(b[0])
            expos.append(b[i+1][0])
            coeffs.append(b[i+1][1])
            contraction.append(c)
    return {'Ls': Ls, 'coeffs': coeffs, 'expos': expos, 'contraction': contraction}

def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

def expand_etb(l, n, alpha, beta):
    return [[l, [alpha*beta**i, 1]] for i in reversed(range(n))]

def expand_etbs(etbs):
    return flatten([expand_etb(*etb) for etb in etbs])


def aug_etb_for_basis(basis, beta=2.0, lmax_restriction=False, lmax_relax=0):
    '''
    adapted from pyscf.
    augment weigend basis with even-tempered gaussian basis exps = alpha*beta^i for i = 1..N.
    Input basis should have the format used in this codebase.
    '''
    newbasis = {}
    for atomic_number in basis.keys():
        conf = orbital_config[atomic_number]
        max_shells = 4 - conf.count(0)
        emin_by_l = [1e99] * 8
        emax_by_l = [0] * 8
        l_max = 0
        for b in basis_to_pyscf(basis[atomic_number]):
            
            l = b[0]
            l_max = max(l_max, l)
            
            if lmax_restriction and l >= max_shells+1+lmax_relax:
                continue
                
            if isinstance(b[1], int):
                e_c = np.array(b[2:])
            else:
                e_c = np.array(b[1:])
            es = e_c[:,0]
            cs = e_c[:,1:]
            es = es[abs(cs).max(axis=1) > 1e-3]
            emax_by_l[l] = max(es.max(), emax_by_l[l])
            emin_by_l[l] = min(es.min(), emin_by_l[l])
        
        
        l_max1 = l_max + 1
        emin_by_l = np.array(emin_by_l[:l_max1])
        emax_by_l = np.array(emax_by_l[:l_max1])
        emax = np.sqrt(np.einsum('i,j->ij', emax_by_l, emax_by_l))
        emin = np.sqrt(np.einsum('i,j->ij', emin_by_l, emin_by_l))
        liljsum = np.arange(l_max1)[:,None] + np.arange(l_max1)
        emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max1*2-1)]
        emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max1*2-1)]
        # Tune emin and emax
        emin_by_l = np.array(emin_by_l) * 2  # *2 for alpha+alpha on same center
        emax_by_l = np.array(emax_by_l) * 2  #/ (np.arange(l_max1*2-1)*.5+1)

        ns = np.log((emax_by_l+emin_by_l)/emin_by_l) / np.log(beta)
        etb = []
        for l, n in enumerate(np.ceil(ns).astype(int)):
            if n > 0:
                etb.append((l, n, emin_by_l[l], beta))
        if etb:
            newbasis[atomic_number] = basis_from_pyscf(expand_etbs(etb))
        else:
            raise RuntimeError(f'Failed to generate even-tempered auxbasis for element {atomic_number}')

    return newbasis