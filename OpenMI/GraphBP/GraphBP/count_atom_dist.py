from dataset import CrossDocked2020_SBDD, collate_mols
from torch.utils.data import DataLoader

from rdkit import Chem
dataset = CrossDocked2020_SBDD()
atomic_num_to_type = {}

atomic_element_to_type = {}

for i in range(len(dataset)-1, -1, -1):
    if i%1000 == 0:
        print(i)
        print(atomic_num_to_type)
        print(atomic_element_to_type)
        print('=======')
    rec_structure, lig_supplier, rec_src, lig_src = dataset[i]
    for atom in rec_structure.get_atoms():
        if atom.element!='H':
            if atom.element not in atomic_element_to_type:
                atomic_element_to_type[atom.element] = 0
            else:
                atomic_element_to_type[atom.element] += 1
        else:
            pass
    lig_mol = Chem.rdmolops.RemoveAllHs(lig_supplier[0], sanitize=False)
    for atom in lig_mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if atom_num not in atomic_num_to_type:
            atomic_num_to_type[atom_num] = 0
        else:
            atomic_num_to_type[atom_num] += 1
    del lig_supplier
    del rec_structure
print(atomic_num_to_type)
print(atomic_element_to_type)


### results

# atomic_num_to_type = {6: 7896717, 7: 1467608, 8: 1827282, 16: 132998, 17: 80952, 15: 122108, 9: 136520, 23: 145, 12: 63, 42: 1002, 33: 45, 35: 21643, 53: 4453, 5: 3295, 14: 55, 34: 312, 30: 18, 21: 4, 26: 223, 44: 214, 45: 19, 74: 47, 39: 1, 29: 32, 79: 6, 51: 3, 13: 29}

# atomic_element_to_type = {'N': 280071932, 'C': 1049822037, 'O': 306441297, 'S': 8411128, 'MG': 101404, 'CA': 99148, 'CL': 62654, 'SE': 21025, 'NA': 46982, 'CD': 7655, 'ZN': 100731, 'P': 47588, 'K': 14544, 'CU': 513, 'CO': 8088, 'CS': 41, 'I': 28335, 'HG': 1941, 'MN': 25}