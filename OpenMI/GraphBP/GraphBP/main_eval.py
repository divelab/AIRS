import pickle
import os
from rdkit.Chem import Draw

from utils import BondAdder
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import scipy as sp





### config
save_mol = True
uff = True
uff_w_rec = False # UFF in the context of binding site
save_sdf_before_uff = False
save_sdf = True
data_root='./data/crossdock2020'

path = './trained_model'
epoch = 33

all_mols_dict_path = os.path.join(path, '{}_mols.mol_dict'.format(epoch))


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Args:
        mol: Rdkit mol object
    
    :rtype:
        :class:`bool`, True if chemically valid, False otherwise
    """

    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False

def rd_mol_to_sdf(rd_mol, sdf_file, kekulize=False, name=''):
    writer = Chem.SDWriter(sdf_file)
    writer.SetKekulize(kekulize)
    if name:
        rd_mol.SetProp('_Name', name)
    writer.write(rd_mol)
    writer.close()
    
def get_rd_atom_res_id(rd_atom):
    '''
    Return an object that uniquely
    identifies the residue that the
    atom belongs to in a given PDB.
    '''
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )
    
def get_pocket(lig_mol, rec_mol, max_dist=8):
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_mol.GetConformer().GetPositions()
    dist = sp.spatial.distance.cdist(lig_coords, rec_coords)

    # indexes of atoms in rec_mol that are
    #   within max_dist of an atom in lig_mol
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])

    # determine pocket residues
    pocket_res_ids = set()
    for i in pocket_atom_idxs:
        atom = rec_mol.GetAtomWithIdx(int(i))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)

    # copy mol and delete atoms
    pkt_mol = rec_mol
    pkt_mol = Chem.RWMol(pkt_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())

    Chem.SanitizeMol(pkt_mol)
    return pkt_mol
    


with open(all_mols_dict_path, 'rb') as f:
    all_mols_dict = pickle.load(f)
    
bond_adder = BondAdder()


all_results_dict = {}
os.makedirs(os.path.join(path, 'gen_mols' + '_epoch_' + str(epoch) + '/') , exist_ok=True)

global_index = 0
global_index_to_rec_src = {}
global_index_to_ref_lig_src = {}
num_valid = 0
for index in all_mols_dict:
    # print(index)
    mol_dicts = all_mols_dict[index]
    for num_atom in mol_dicts:
        if type(num_atom) is int:
            mol_dicts_w_num_atom = mol_dicts[num_atom]
            num_mol_w_num_atom = len(mol_dicts_w_num_atom['_atomic_numbers'])
            for j in range(num_mol_w_num_atom):
                global_index += 1
                
                
                ### Add bonds
                atomic_numbers = mol_dicts_w_num_atom['_atomic_numbers'][j]
                positions = mol_dicts_w_num_atom['_positions'][j]
                rd_mol, ob_mol = bond_adder.make_mol(atomic_numbers, positions)
                
                ### check validity
                if check_chemical_validity(rd_mol):
                    num_valid += 1
                print('Valid molecules:', num_valid)

                rd_mol = Chem.AddHs(rd_mol, explicitOnly=True, addCoords=True)    
                if save_sdf_before_uff:
                    sdf_file = os.path.join(path, 'gen_mols' + '_epoch_' + str(epoch) + '/' + str(global_index) + '_beforeuff.sdf')
                    rd_mol_to_sdf(rd_mol, sdf_file)
                    print('Saving' + str(sdf_file))
                    
                    
                # ### UFF minimization
                if uff:
                    try:
                        # print(rd_mol.GetConformer().GetPositions())
                        UFFOptimizeMolecule(rd_mol)
                        print("Performing UFF...")
                        # print(rd_mol.GetConformer().GetPositions())
                    except:
                        print('Skip UFF...')
                        # pass
                    
                if uff_w_rec:
                    # try:
                    # print(rd_mol.GetConformer().GetPositions())
                    # print(rd_mol.GetConformer().GetPositions().shape)
                    rd_mol = Chem.RWMol(rd_mol)
                    rec_mol = Chem.MolFromPDBFile(os.path.join(data_root, mol_dicts['rec_src']), sanitize=True)
                    rec_mol = get_pocket(rd_mol, rec_mol)
                    
                    
                    uff_mol = Chem.CombineMols(rec_mol, rd_mol)
                    
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()])
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()].shape)
                    
                    try:
                        Chem.SanitizeMol(uff_mol)
                    except Chem.AtomValenceException:
                        print('Invalid valence')
                    except (Chem.AtomKekulizeException, Chem.KekulizeException):
                        print('Failed to kekulize')
                    try:
                        # UFFOptimizeMolecule(uff_mol)
                        uff = AllChem.UFFGetMoleculeForceField(
                            uff_mol, confId=0, ignoreInterfragInteractions=False
                        )
                        uff.Initialize()
                        # E_init = uff.CalcEnergy()
                        for i in range(rec_mol.GetNumAtoms()): # Fix the rec atoms
                            uff.AddFixedPoint(i)
                        converged = False
                        n_iters=200
                        n_tries=2
                        while n_tries > 0 and not converged:
                            print('.', end='', flush=True)
                            converged = not uff.Minimize(maxIts=n_iters)
                            n_tries -= 1
                        print(flush=True)
                        # E_final = uff.CalcEnergy()
                        print("Performed UFF with binding site...")
                    except:
                        print('Skip UFF...')
                    coords = uff_mol.GetConformer().GetPositions()
                    rd_conf = rd_mol.GetConformer()
                    for i, xyz in enumerate(coords[-rd_mol.GetNumAtoms():]):
                        rd_conf.SetAtomPosition(i, xyz)
                    # print(rd_mol.GetConformer().GetPositions())
                    # print(rd_mol.GetConformer().GetPositions().shape)
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()])
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()].shape)
                    # print(E_init, E_final)
                    
                if save_sdf:
                    
                    ###
                    try:
                        rd_mol = Chem.RemoveHs(rd_mol)
                        print("Remove H atoms before saving mol...")
                    except:
                        print("Cannot remove H atoms...")
                    
                    sdf_file = os.path.join(path, 'gen_mols' + '_epoch_' + str(epoch) + '/' + str(global_index) + '.sdf')
                    rd_mol_to_sdf(rd_mol, sdf_file)
                    print('Saving' + str(sdf_file))
                    global_index_to_rec_src[global_index] = mol_dicts['rec_src']
                    global_index_to_ref_lig_src[global_index] = mol_dicts['lig_src']
                
                if save_mol:
                    try:
                        img_path = os.path.join(path, 'gen_mols' + '_epoch_' + str(epoch) + '/' + str(global_index) + '.png')
                        img = Draw.MolsToGridImage([rd_mol])
                        img.save(img_path)
                        print('Saving' + str(img_path))
                    except:
                        pass
                print('------------------------------------------------')
        else:
            continue
            
if save_sdf:
    print('Saving dicts...')
    with open(os.path.join(path, 'gen_mols_epoch_{}/global_index_to_rec_src.dict').format(epoch),'wb') as f:
        pickle.dump(global_index_to_rec_src, f)
    with open(os.path.join(path, 'gen_mols_epoch_{}/global_index_to_ref_lig_src.dict').format(epoch),'wb') as f:
        pickle.dump(global_index_to_ref_lig_src, f)

print('Done!!!')

