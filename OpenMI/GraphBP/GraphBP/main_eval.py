import argparse
import pickle
import os
from rdkit.Chem import Draw
from utils import BondAdder
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import scipy as sp


def str2bool(v):
    """
    Converts a string representation of a boolean to a boolean.
    Args:
        v (str): The string to convert.
    Returns:
        bool: The converted boolean value.
    """
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    m = Chem.MolFromSmiles(s) # implicitly performs sanitization
    return m is not None

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
    # within max_dist of an atom in lig_mol
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

def main():
    parser = argparse.ArgumentParser(description="Wrapper for CADD pipeline targeting Aurora protein kinases.")
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()

    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    pdbid = args.pdbid.lower()
    epoch = args.epoch 

    save_mol = False
    uff = True
    uff_w_rec = False
    save_sdf_before_uff = False
    save_sdf = True
    data_root = './data/crossdock2020'
    path = './trained_model_reduced_dataset_100_epochs'
    all_mols_dict_path = os.path.join(path, f'epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}.mol_dict')

    with open(all_mols_dict_path, 'rb') as f:
        all_mols_dict = pickle.load(f)

    bond_adder = BondAdder()

    gen_mols_dir = os.path.join(path, f'gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}')
    os.makedirs(gen_mols_dir, exist_ok=True)
    os.makedirs(os.path.join(gen_mols_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(gen_mols_dir, 'sdf'), exist_ok=True)

    global_index = 0
    global_index_to_rec_src = {}
    global_index_to_ref_lig_src = {}
    num_valid = 0

    for index in all_mols_dict:
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
                        sdf_file = os.path.join(gen_mols_dir, f'{global_index}_beforeuff.sdf')
                        rd_mol_to_sdf(rd_mol, sdf_file)
                        print('Saving' + str(sdf_file))
                    
                    ### UFF minimization
                    if uff:
                        try:
                            UFFOptimizeMolecule(rd_mol)
                            print("Performing UFF...")
                        except Exception:
                            print('Skip UFF...')
                    
                    if uff_w_rec:
                        rd_mol = Chem.RWMol(rd_mol)
                        rec_mol = Chem.MolFromPDBFile(os.path.join(data_root, mol_dicts['rec_src']), sanitize=True)
                        rec_mol = get_pocket(rd_mol, rec_mol)
                        uff_mol = Chem.CombineMols(rec_mol, rd_mol)
                        try:
                            Chem.SanitizeMol(uff_mol)
                        except Chem.AtomValenceException:
                            print('Invalid valence')
                        except (Chem.AtomKekulizeException, Chem.KekulizeException):
                            print('Failed to kekulize')
                        try:
                            uff = AllChem.UFFGetMoleculeForceField(
                                uff_mol, confId=0, ignoreInterfragInteractions=False
                            )
                            uff.Initialize()
                            for i in range(rec_mol.GetNumAtoms()):
                                uff.AddFixedPoint(i)
                            converged = False
                            n_iters = 200
                            n_tries = 2
                            while n_tries > 0 and not converged:
                                print('.', end='', flush=True)
                                converged = not uff.Minimize(maxIts=n_iters)
                                n_tries -= 1
                            print(flush=True)
                            print("Performed UFF with binding site...")
                        except Exception:
                            print('Skip UFF...')
                        coords = uff_mol.GetConformer().GetPositions()
                        rd_conf = rd_mol.GetConformer()
                        for i, xyz in enumerate(coords[-rd_mol.GetNumAtoms():]):
                            rd_conf.SetAtomPosition(i, xyz)
                    if save_sdf:
                        try:
                            rd_mol = Chem.RemoveHs(rd_mol)
                            print("Remove H atoms before saving mol...")
                        except Exception:
                            print("Cannot remove H atoms...")
                        sdf_file = os.path.join(gen_mols_dir, 'sdf', f'{global_index}.sdf')
                        rd_mol_to_sdf(rd_mol, sdf_file)
                        print('Saving' + str(sdf_file))
                        global_index_to_rec_src[global_index] = mol_dicts['rec_src']
                        global_index_to_ref_lig_src[global_index] = mol_dicts['lig_src']
                    
                    if save_mol:
                        try:
                            img_path = os.path.join(gen_mols_dir, 'images', f'{global_index}.png')
                            img = Draw.MolsToGridImage([rd_mol])
                            img.save(img_path)
                            print('Saving' + str(img_path))
                        except Exception:
                            pass
                    print('------------------------------------------------')
            else:
                continue

    if save_sdf:
        print('Saving dicts...')
        with open(os.path.join(gen_mols_dir, 'global_index_to_rec_src.dict'), 'wb') as f:
            pickle.dump(global_index_to_rec_src, f)
        with open(os.path.join(gen_mols_dir, 'global_index_to_ref_lig_src.dict'), 'wb') as f:
            pickle.dump(global_index_to_ref_lig_src, f)

    print('Done!!!')

if __name__ == '__main__':
    main()
