import numpy as np
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

def nan_to_num(vec, num=0.0):
    idx = np.isnan(vec)
    vec[idx] = num
    return vec

def _normalize(vec, axis=-1):
    return nan_to_num(
        np.divide(vec, np.linalg.norm(vec, axis=axis, keepdims=True)))

def write_xyz_file(atom_types, atom_coordinates, file_path):
    with open(file_path, 'w') as file:
        num_atoms = len(atom_types)
        file.write(f"{num_atoms}\n")
        file.write('\n')

        for atom_type, coords in zip(atom_types, atom_coordinates):
            x, y, z = coords
            file.write(f"{atom_type} {np.format_float_positional(x)} {np.format_float_positional(y)} {np.format_float_positional(z)}\n")



dict_qm9 = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F'}


def write_qm9_to_seq(raw_path='data/QM9Gen/processed_edm/',
                     write_path='QM9_seq/',
                     split='train',
                     order_type='order',
                     remove_h=False,
                     sample=False):
        
    write_name_ori_coord = f"{order_type}{'_ori_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
    write_name_invariant = f"{order_type}{'_invariant_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
    write_name_spherical = f"{order_type}{'_spherical_cord'}{'_noH' if remove_h else '_adH'}{'_sample' if sample else ''}{'_seq'}"
    
    write_path_ori_coord = write_path + write_name_ori_coord + '.txt'
    write_path_invariant = write_path + write_name_invariant + '.txt'
    write_path_spherical = write_path + write_name_spherical + '.txt'
    

    if split == 'train':
        raw_file = raw_path + 'train.npz'
    elif split == 'valid':
        raw_file = raw_path + 'valid.npz'
    else:
        print('!!! split not supported !!!')
        exit()

    smiles_seq = []
    atom_seq = []
    coords_seq = []
    invariant_coords_seq = []
    spherical_coords_seq = []

    all_data = np.load(raw_file)
    keys = all_data.files 

    ############### all files ################
    # 'num_atoms', 'charges', 'positions', 
    # 'index', 
    # properties: 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 
    # properties: 'U0', 'U', 'H', 'G', 'Cv', 'omega1', 'zpve_thermo', 'U0_thermo', 'U_thermo', 'H_thermo', 'G_thermo', 'Cv_thermo'
    # need to change units: 
    # qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 
    # 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    if not sample:
        num_mol = len(all_data[keys[0]])
    else:
        num_mol = 5

    all_num_atoms = all_data['num_atoms']
    all_charges = all_data['charges']
    all_positions = all_data['positions']

    for i in tqdm(range(num_mol)):
        num_atom = all_num_atoms[i]
        file_path = 'Molecule3D/example.xyz'
        ori_z = all_charges[i][:num_atom]
        ori_coords = all_positions[i][:num_atom]
        
        write_xyz_file([dict_qm9[key] for key in ori_z], ori_coords, file_path)
        raw_mol = Chem.MolFromXYZFile(file_path)
        mol = Chem.Mol(raw_mol)
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except:
            pass
        smiles = Chem.MolToSmiles(mol)

        order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
        reorder_mol = Chem.RenumberAtoms(mol,order)
        atom_type = np.array([atom.GetSymbol() for atom in reorder_mol.GetAtoms()])
        coords = reorder_mol.GetConformer().GetPositions()

        if remove_h:
            mask = atom_type != 'H'
            atom_type = atom_type[mask]
            coords = coords[mask]
            num_atom = len(atom_type)

        centered_coords = coords - coords[0]
        invariant_coords = np.zeros_like(coords)
        spherical_coords = np.zeros_like(coords)

        # we have to select three nodes to build a global frame
        flag = False 

        if num_atom == 1:
            pass
        elif num_atom == 2:
            d = np.linalg.norm(coords[1] - coords[0], axis=-1)
            invariant_coords[1,0] = d
            spherical_coords[1,0] = d
        else:
            v1 = centered_coords[1] - centered_coords[0]
            for i in range(2, num_atom):
                v2 = centered_coords[i] - centered_coords[0]
                if np.linalg.norm(np.cross(v1, v2)) != 0:
                    flag = True # # can find the third node that is not on the same line as the first two nodes
                    break
            if flag == False and i == num_atom - 1: # cannot find the third node that is not on the same line as the first two nodes
                invariant_coords = centered_coords
            else:
                # build a global frame (xyz axis)
                x = _normalize(v1)
                y = _normalize(np.cross(v1, v2))
                z = np.cross(x, y)
                # invariant coords
                invariant_coords = np.dot(centered_coords, np.stack((x, y, z)).T)
            d = np.linalg.norm(invariant_coords, axis=-1)
            theta = np.zeros_like(d)
            theta[1:] = np.arccos(invariant_coords[1:,2]/d[1:])
            phi = np.arctan2(invariant_coords[:,1], invariant_coords[:,0])
            # invariant_spherical_coords
            spherical_coords = np.stack((d, theta, phi)).T

        
        coords = np.array([["{:.2f}".format(value) for value in row] for row in coords])
        invariant_coords = np.array([["{:.2f}".format(value) for value in row] for row in invariant_coords])
        spherical_coords = np.array([["{:.2f}".format(value) for value in row] for row in spherical_coords])
        coords_seq.append(coords)
        invariant_coords_seq.append(invariant_coords)
        spherical_coords_seq.append(spherical_coords)

        smiles_seq.append(smiles)
        atom_seq.append(atom_type)

    
    with open(write_path_ori_coord, 'w') as file:
        for i in range(len(atom_seq)):
            for j in range(len(atom_seq[i])):
                file.write(atom_seq[i][j])
                file.write(' ')
                file.write(str(coords_seq[i][j][0]) + ' ' + str(coords_seq[i][j][1]) + ' ' + str(coords_seq[i][j][2]) + ' ')
            file.write('\n')

    with open(write_path_invariant, 'w') as file:
        for i in range(len(atom_seq)):
            for j in range(len(atom_seq[i])):
                file.write(atom_seq[i][j])
                file.write(' ')
                file.write(str(invariant_coords_seq[i][j][0]) + ' ' + str(invariant_coords_seq[i][j][1]) + ' ' + str(invariant_coords_seq[i][j][2]) + ' ')
            file.write('\n')
    
    with open(write_path_spherical, 'w') as file:
        for i in range(len(atom_seq)):
            for j in range(len(atom_seq[i])):
                file.write(atom_seq[i][j])
                file.write(' ')
                file.write(str(spherical_coords_seq[i][j][0]) + ' ' + str(spherical_coords_seq[i][j][1]) + '° ' + str(spherical_coords_seq[i][j][2]) + '° ')
            file.write('\n')

