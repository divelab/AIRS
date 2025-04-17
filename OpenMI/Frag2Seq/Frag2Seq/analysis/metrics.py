import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from analysis.SA_Score.sascorer import calculateScore

from analysis.molecule_builder import build_molecule
from copy import deepcopy


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / self.p + self.EPS))


def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None,
                 connectivity_thresh=1.0):
        self.atom_decoder = dataset_info['atom_decoder']
        if dataset_smiles_list is not None:
            dataset_smiles_list = set(dataset_smiles_list)
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        if len(generated) < 1:
            return [], 0.0

        valid = []
        for mol in generated:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            valid.append(mol)

        return valid, len(valid) / len(generated)

    def compute_connectivity(self, valid):
        """ Consider molecule connected if its largest fragment contains at
        least x% of all atoms, where x is determined by
        self.connectivity_thresh (defaults to 100%). """
        if len(valid) < 1:
            return [], 0.0

        connected = []
        connected_smiles = []
        for mol in valid:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = \
                max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
                    connected.append(largest_mol)

        return connected, len(connected_smiles) / len(valid), connected_smiles

    def compute_uniqueness(self, connected):
        """ valid: list of SMILES strings."""
        if len(connected) < 1 or self.dataset_smiles_list is None:
            return [], 0.0

        return list(set(connected)), len(set(connected)) / len(connected)

    def compute_novelty(self, unique):
        if len(unique) < 1:
            return [], 0.0

        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate_rdmols(self, rdmols):
        valid, validity = self.compute_validity(rdmols)
        print(f"Validity over {len(rdmols)} molecules: {validity * 100 :.2f}%")

        connected, connectivity, connected_smiles = \
            self.compute_connectivity(valid)
        print(f"Connectivity over {len(valid)} valid molecules: "
              f"{connectivity * 100 :.2f}%")

        unique, uniqueness = self.compute_uniqueness(connected_smiles)
        print(f"Uniqueness over {len(connected)} connected molecules: "
              f"{uniqueness * 100 :.2f}%")

        _, novelty = self.compute_novelty(unique)
        print(f"Novelty over {len(unique)} unique connected molecules: "
              f"{novelty * 100 :.2f}%")

        return [validity, connectivity, uniqueness, novelty], [valid, connected]

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """

        rdmols = [build_molecule(*graph, self.dataset_info)
                  for graph in generated]
        return self.evaluate_rdmols(rdmols)


class MoleculeProperties:

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def evaluate(self, pocket_rdmols, write_path=None):
        """
        Run full evaluation
        Args:
            pocket_rdmols: list of lists, the inner list contains all RDKit
                molecules generated for a pocket
        Returns:
            QED, SA, LogP, Lipinski (per molecule), and Diversity (per pocket)
        """

        for pocket in pocket_rdmols:
            for mol in pocket:
                Chem.SanitizeMol(mol)
                assert mol is not None, "only evaluate valid molecules"

        all_qed = []
        all_sa = []
        all_logp = []
        all_lipinski = []
        per_pocket_diversity = []
        for pocket in tqdm(pocket_rdmols):
            all_qed.append([self.calculate_qed(mol) for mol in pocket])
            all_sa.append([self.calculate_sa(mol) for mol in pocket])
            all_logp.append([self.calculate_logp(mol) for mol in pocket])
            all_lipinski.append([self.calculate_lipinski(mol) for mol in pocket])
            per_pocket_diversity.append(self.calculate_diversity(pocket))

        if write_path is not None:
            with open(write_path, 'a') as f:
                f.write("\n ======================================= \n")
                print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
                    f"{len(pocket_rdmols)} pockets evaluated.")
                f.write(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
                    f"{len(pocket_rdmols)} pockets evaluated. \n")

                qed_flattened = [x for px in all_qed for x in px]
                print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")
                f.write(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f} \n")

                sa_flattened = [x for px in all_sa for x in px]
                print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")
                f.write(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f} \n")

                logp_flattened = [x for px in all_logp for x in px]
                print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")
                f.write(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f} \n")

                lipinski_flattened = [x for px in all_lipinski for x in px]
                print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")
                f.write(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f} \n")

                print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")
                f.write(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f} \n")

        else:
            print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
                f"{len(pocket_rdmols)} pockets evaluated.")

            qed_flattened = [x for px in all_qed for x in px]
            print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")

            sa_flattened = [x for px in all_sa for x in px]
            print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")

            logp_flattened = [x for px in all_logp for x in px]
            print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")

            lipinski_flattened = [x for px in all_lipinski for x in px]
            print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")

            print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")

        return all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity

    def evaluate_mean(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            QED, SA, LogP, Lipinski, and Diversity
        """

        if len(rdmols) < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        for mol in rdmols:
            Chem.SanitizeMol(mol)
            assert mol is not None, "only evaluate valid molecules"

        qed = np.mean([self.calculate_qed(mol) for mol in rdmols])
        sa = np.mean([self.calculate_sa(mol) for mol in rdmols])
        logp = np.mean([self.calculate_logp(mol) for mol in rdmols])
        lipinski = np.mean([self.calculate_lipinski(mol) for mol in rdmols])
        diversity = self.calculate_diversity(rdmols)

        return qed, sa, logp, lipinski, diversity
