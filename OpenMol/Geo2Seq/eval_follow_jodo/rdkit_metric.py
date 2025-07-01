from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
import copy
import numpy as np


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info)

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        return [validity, uniqueness, novelty], unique


def compute_validity(rdmols):
    valid = []

    for mol in rdmols:
        smiles = mol2smiles(mol)
        if smiles is not None:
            valid.append(smiles)

    return valid, len(valid) / len(rdmols)


def eval_rdmol(rd_mols, train_smiles=None):
    # validity and complete rate
    valid_smiles = []

    complete_n = 0
    for mol in rd_mols:
        mol = copy.deepcopy(mol)
        smiles = mol2smiles(mol)
        if smiles is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            except:
                continue
            if len(mol_frags) == 1:
                complete_n += 1
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
            valid_smiles.append(smiles)

    Validity = len(valid_smiles) / len(rd_mols)
    Complete = complete_n / len(rd_mols)
    if Validity > 0:
        # unique & valid rate
        Unique = len(set(valid_smiles)) / len(rd_mols)
    else:
        Unique = 0

    Novelty = -1
    if train_smiles is not None:
        # num_novel = 0
        # for smiles in set(valid_smiles):
        #     if smiles not in train_smiles:
        #         num_novel += 1
        # Novelty = num_novel / len(rd_mols)
        gen_smiles_set = set(valid_smiles) - {None}
        train_set = set(train_smiles) - {None}
        Novelty = len(gen_smiles_set - train_set) / len(rd_mols)

    return dict(
        Validity=Validity,
        Complete=Complete,
        Unique=Unique,
        Novelty=Novelty
    )


def get_rdkit_rmsd(mols, n_conf=32, random_seed=42, num_workers=16):
    # check the best alignment between generated mols and rdkit conformers

    lowest_rmsd = []
    for mol in mols:
        mol_3d = copy.deepcopy(mol)
        try:
            Chem.SanitizeMol(mol_3d)
        except:
            continue
        confIds = AllChem.EmbedMultipleConfs(mol_3d, n_conf, randomSeed=random_seed,
                                             clearConfs=True, numThreads=num_workers)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol_3d, numThreads=num_workers)
        except:
            continue
        tmp_rmsds = []
        for confId in confIds:
            # AllChem.UFFOptimizeMolecule(mol, confId=confId)
            # try:
            #     AllChem.MMFFOptimizeMolecule(mol_3d, confId=confId)
            # except:
            #     continue
            try:
                rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol_3d, refId=confId)
                tmp_rmsds.append(rmsd)
            except:
                continue

        if len(tmp_rmsds) != 0:
            lowest_rmsd.append(np.min(np.array(tmp_rmsds)))

    return np.array(lowest_rmsd)

