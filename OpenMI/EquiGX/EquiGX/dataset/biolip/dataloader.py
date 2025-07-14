import os.path as osp
import h5py
import numpy as np
import warnings
from tqdm import tqdm

import torch 
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from Bio.PDB import PDBParser

import re 

class BioLiPdataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                ):

        self.root = root

        super(BioLiPdataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed_oneGraph'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self):
        name = 'BioLip.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'

    def ligand_data(self, ligandPath):  
        element_to_atomic_number = {
            'H': 0, 'HE': 1, 'LI': 2, 'BE': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'NE': 9,
            'NA': 10, 'MG': 11, 'AL': 12, 'SI': 13, 'P': 14, 'S': 15, 'CL': 16, 'AR': 17, 'K': 18, 'CA': 19,
            'SC': 20, 'TI': 21, 'V': 22, 'CR': 23, 'MN': 24, 'FE': 25, 'CO': 26, 'NI': 27, 'CU': 28, 'ZN': 29,
            'GA': 30, 'GE': 31, 'AS': 32, 'SE': 33, 'BR': 34, 'KR': 35, 'RB': 36, 'SR': 37, 'Y': 38, 'ZR': 39,
            'NB': 40, 'MO': 41, 'TC': 42, 'RU': 43, 'RH': 44, 'PD': 45, 'AG': 46, 'CD': 47, 'IN': 48, 'SN': 49,
            'SB': 50, 'TE': 51, 'I': 52, 'XE': 53, 'CS': 54, 'BA': 55, 'LA': 56, 'CE': 57, 'PR': 58, 'ND': 59,
            'PM': 60, 'SM': 61, 'EU': 62, 'GD': 63, 'TB': 64, 'DY': 65, 'HO': 66, 'ER': 67, 'TM': 68, 'YB': 69,
            'LU': 70, 'HF': 71, 'TA': 72, 'W': 73, 'RE': 74, 'OS': 75, 'IR': 76, 'PT': 77, 'AU': 78, 'HG': 79,
            'TL': 80, 'PB': 81, 'BI': 82, 'PO': 83, 'AT': 84, 'RN': 85, 'FR': 86, 'RA': 87, 'AC': 88, 'TH': 89,
            'PA': 90, 'U': 91, 'NP': 92, 'PU': 93, 'AM': 94, 'CM': 95, 'BK': 96, 'CF': 97, 'ES': 98, 'FM': 99,
            'MD': 100, 'NO': 101, 'LR': 102, 'RF': 103, 'DB': 104, 'SG': 105, 'BH': 106, 'HS': 107, 'MT': 108,
            'DS': 109, 'RG': 110, 'CN': 111, 'NH': 112, 'FL': 113, 'MC': 114, 'LV': 115, 'TS': 116, 'OG': 117,
        }
        
        # Initialize an array to hold the extracted information
        atom_types = []
        atom_coords = []
        
        with open(ligandPath, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Extract atom coordinates
                    atom_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    
                    # Assuming element symbol is right-aligned within columns 77-78 as per standard PDB files
                    element_symbol = line[76:78].strip().upper()
                    
                    # Map element symbol to adjusted atomic number, default to -1 if not found
                    atomic_number = element_to_atomic_number.get(element_symbol, -1)
                    
                    if atomic_number == -1:
                        continue
                    
                    atom_types.append(atomic_number)
                    atom_coords.append(atom_coord)
                    
        return torch.tensor(atom_types), torch.tensor(atom_coords)

        
    
    def protein_data(self, proteinPath):
        # backbone_atoms = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
        backbone_atoms = {'CA': 0}
    
        # Define a mapping for residue types to integers (for the 20 standard amino acids)
        residue_mapping = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }

        
        # Initialize an array to hold the extracted information
        atom_types = []
        atom_coords = []
        residue_idices = []
        residue_types = []
        
        with open(proteinPath, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[13:16].strip()
                    if atom_name in backbone_atoms:
                        # Convert atom name to its corresponding integer
                        atom_type = backbone_atoms[atom_name]
                        residue_type_str = line[17:20].strip()
                        
                        # Convert residue type to its corresponding integer, default to -1 if not found
                        residue_type = residue_mapping.get(residue_type_str, -1)
                        
                        # Extract needed information
                        atom_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        residue_idx = int(line[22:26].strip())
                        if (line[26]!= ' '):                                    # Deal with insertion code
                            residue_idx += (ord(line[26]) - ord('A') + 1) / 24 
                        
                        atom_types.append(atom_type)
                        atom_coords.append(atom_coord)
                        residue_idices.append(residue_idx)
                        residue_types.append(residue_type)
                        
        return torch.tensor(atom_types), torch.tensor(atom_coords), torch.tensor(residue_idices), torch.tensor(residue_types)
        
    def convert_to_molar(self, value, unit):
        if unit == "mM":
            return value * 1e3
        elif unit == "uM":
            return value * 1
        elif unit == "nM":
            return value * 1e-3
        else:
            return None
    
    def process(self):
        print('Beginning Processing ...')
        label_path = osp.join(self.root, "BioLiP.txt")
        with open(label_path, 'r') as file:
            # Read all lines in the file and store them in a list
            lines = file.readlines()
            
        data_list = [] 
        
        for line in tqdm(lines):
            cols = line.split("\t")

            molar_value = None
            
            if cols[15]!='':
                kd_value = re.search(r'Kd=([0-9.]+)(uM|nM|mM)', cols[15])
                ki_value = re.search(r'Ki=([0-9.]+)(uM|nM|mM)', cols[15])
                if (kd_value):
                    value, unit = float(kd_value.group(1)), kd_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
                elif (ki_value):
                    value, unit = float(ki_value.group(1)), ki_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
                
                    
            if molar_value is None and cols[16]!='': 
                kd_value = re.search(r'Kd=([0-9.]+)(uM|nM|mM)', cols[16])
                ki_value = re.search(r'Ki=([0-9.]+)(uM|nM|mM)', cols[16])
                if (kd_value):
                    value, unit = float(kd_value.group(1)), kd_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
                elif (ki_value):
                    value, unit = float(ki_value.group(1)), ki_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
            
            if molar_value is None and cols[14]!='': 
                kd_value = re.search(r'Kd=([0-9.]+)(uM|nM|mM)', cols[14])
                ki_value = re.search(r'Ki=([0-9.]+)(uM|nM|mM)', cols[14])
                if (kd_value):
                    value, unit = float(kd_value.group(1)), kd_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
                elif (ki_value):
                    value, unit = float(ki_value.group(1)), ki_value.group(2)
                    molar_value = self.convert_to_molar(value, unit)
                    
            if molar_value is None:
                continue
            
            if (cols[4] == "dna" or cols[4] == 'rna'):
                continue
            
            receptor_path = osp.join(self.root, "BioLiP_updated_set/receptor/"+cols[0]+cols[1]+".pdb")
            ligand_path = osp.join(self.root, "BioLiP_updated_set/ligand/"+cols[0]+"_"+cols[4]+"_"+cols[5]+"_"+cols[6]+".pdb")
            
            receptor_atom_types, receptor_atom_coords, receptor_residue_idices, receptor_residue_types = self.protein_data(receptor_path)
            ligand_atom_types, ligand_atom_coords = self.ligand_data(ligand_path)
            
            
            residue_mapping = {
                'A': 0,  # Alanine
                'R': 1,  # Arginine
                'N': 2,  # Asparagine
                'D': 3,  # Aspartic acid
                'C': 4,  # Cysteine
                'E': 5,  # Glutamic acid
                'Q': 6,  # Glutamine
                'G': 7,  # Glycine
                'H': 8,  # Histidine
                'I': 9,  # Isoleucine
                'L': 10, # Leucine
                'K': 11, # Lysine
                'M': 12, # Methionine
                'F': 13, # Phenylalanine
                'P': 14, # Proline
                'S': 15, # Serine
                'T': 16, # Threonine
                'W': 17, # Tryptophan
                'Y': 18, # Tyrosine
                'V': 19, # Valine
                'X': -1, # Unknown
            }
            
            binding_site_residues = cols[7].split(' ')
            binding_site_residues_idx = []
            binding_site_residues_types = []
            for s in binding_site_residues:
                if (s[-1].isnumeric()):
                    binding_site_residues_idx.append(int(s[1:]))
                else:
                    tmp = int(s[1:-1])
                    tmp += (ord(s[-1]) - ord('A') + 1) / 24 
                    binding_site_residues_idx.append(tmp)
                
                try:
                    binding_site_residues_types.append(residue_mapping[s[0]])
                except:
                    import pdb
                    pdb.set_trace()
                    
                
            xai_labels = torch.zeros_like(receptor_atom_types)
            
            for i in range(len(receptor_atom_types)):
                if (receptor_residue_idices[i] in binding_site_residues_idx): 
                    try:
                        if binding_site_residues_types[binding_site_residues_idx.index(receptor_residue_idices[i])] != -1:              
                            assert binding_site_residues_types[binding_site_residues_idx.index(receptor_residue_idices[i])] == receptor_residue_types[i]
                        xai_labels[i] = 1.0 
                    except AssertionError:
                        continue
                        import pdb
                        pdb.set_trace()
            
            data = Data()
            
            # data.x = F.one_hot(receptor_residue_types, num_classes = 20)
            # data.x = torch.cat( [F.one_hot(receptor_atom_types, num_classes = 4), F.one_hot(receptor_residue_types, num_classes = 20), receptor_residue_idices.reshape(-1,1)], dim = 1)
            # data.receptor_atom_coords = receptor_atom_coords
            
            # data.ligand_x = F.one_hot(ligand_atom_types, num_classes = 118)
            
            residue_types = torch.cat([receptor_residue_types, torch.full(ligand_atom_types.shape, 20)], dim = 0)
            
            atom_types = torch.cat([torch.full(receptor_residue_types.shape, 5), ligand_atom_types], dim = 0)
            
            data.x = torch.cat( [F.one_hot(residue_types, num_classes = 21), F.one_hot(atom_types, num_classes = 118) ], dim = 1)
             
            # data.ligand_atom_coords = ligand_atom_coords
            
            data.pos = torch.cat([receptor_atom_coords, ligand_atom_coords], dim = 0)
            
            data.xai_labels = xai_labels.reshape(-1,1) # (n_atom, 1)
            data.affinity_value = torch.tensor(molar_value) 
            
            data_list.append(data) 
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')
 

if __name__ == "__main__":
    dataset = PDBbinddataset(root = "./")
    print(dataset)
