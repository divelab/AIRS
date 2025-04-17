import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem

import io
import numpy as np
import threading

def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True)))



def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out





