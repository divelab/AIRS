# Code adapted from https://github.com/luost26/3D-Generative-SBDD/tree/6f9c7d92784e58474b9c22a74c8113f0344ca795 and https://github.com/mattragoza/liGAN

import numpy as np

from openbabel import openbabel as ob
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

class MolReconsError(Exception):
    pass

class BondAdder():
    '''
    An algorithm for constructing a valid molecule
    from a structure of atomic coordinates and types.
    
    First, it converts the struture to OBAtoms and
    tries to maintain as many of the atomic proper-
    ties defined by the atom types as possible.
    
    Next, it add bonds to the atoms, using the atom
    properties and coordinates as constraints.
    '''
    def __init__(
        self,
        min_bond_len=0.01,
        max_bond_len=4.0,
        max_bond_stretch=0.45,
        min_bond_angle=45
    ):
        self.min_bond_len = min_bond_len
        self.max_bond_len = max_bond_len

        self.max_bond_stretch = max_bond_stretch
        self.min_bond_angle = min_bond_angle
        self.UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE:Chem.BondType.DOUBLE, Chem.BondType.DOUBLE:Chem.BondType.TRIPLE}
        
    def to_ob_mol(self, xyz, atomic_nums):
        '''
        Convert numpy arrays to ob_mol
        '''
        mol = ob.OBMol()
        mol.BeginModify()
        atoms = []
        for xyz,t in zip(xyz, atomic_nums):
            x,y,z = xyz
            atom = mol.NewAtom()
            atom.SetAtomicNum(t)
            atom.SetVector(x,y,z)
            atoms.append(atom)
        return mol, atoms
    
    
    def fixup(self, mol):
        mol.SetAromaticPerceived(True)  #avoid perception
        for atom in ob.OBMolAtomIter(mol):


            if atom.IsAromatic():
                atom.SetHyb(2)

            if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():     # Nitrogen, Oxygen
                #this is a little iffy, ommitting until there is more evidence it is a net positive
                #we don't have aromatic types for nitrogen, but if it
                #is in a ring with aromatic carbon mark it aromatic as well
                acnt = 0
                for nbr in ob.OBAtomAtomIter(atom):
                    if nbr.IsAromatic():
                        acnt += 1
                if acnt > 1:
                    atom.SetAromatic(True)

    def connect_the_dots(self, mol, atoms):
        '''
        Add bonds based on distance
        '''
        pt = Chem.GetPeriodicTable()
        
        mol.BeginModify()

        #just going to to do n^2 comparisons, can worry about efficiency later
        coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))
        
        for (i,a) in enumerate(atoms):
            for (j,b) in enumerate(atoms):
                if i == j: # Note that this differs from https://github.com/luost26/3D-Generative-SBDD/blob/6f9c7d92784e58474b9c22a74c8113f0344ca795/utils/reconstruct.py#L93
                    break
                if self.min_bond_len < dists[i,j] < self.max_bond_len:
                    flag = 0
                    ### Aromatic
                    if a.IsAromatic() and b.IsAromatic():
                        flag = ob.OB_AROMATIC_BOND
                    mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)
                
        atom_maxb = {}
        for (i,a) in enumerate(atoms):
        #set max valance to the smallest max allowed by openbabel or rdkit
        #since we want the molecule to be valid for both (rdkit is usually lower)
            maxb = ob.GetMaxBonds(a.GetAtomicNum())
            maxb = min(maxb,pt.GetDefaultValence(a.GetAtomicNum()))
            if a.GetAtomicNum() == 16: # sulfone check
                if self.count_nbrs_of_elem(a, 8) >= 2:
                    maxb = 6
            atom_maxb[a.GetIdx()] = maxb
            
        #remove any impossible bonds between halogens
        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
                mol.DeleteBond(bond)
        
        def get_bond_info(biter):
            '''Return bonds sorted by their distortion'''
            bonds = [b for b in biter]
            binfo = []
            for bond in bonds:
                bdist = bond.GetLength()
                #compute how far away from optimal we are
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum()) 
                stretch = bdist-ideal
                binfo.append((stretch,bdist,bond))
            binfo.sort(reverse=True, key=lambda t: t[:2]) #most stretched bonds first
            return binfo
        
        #prioritize removing hypervalency causing bonds, do more valent constrained atoms first since their bonds introduce the most problems with reachability (e.g. oxygen)
        hypers = sorted([(atom_maxb[a.GetIdx()],a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms],key=lambda aa: (aa[0],-aa[1]))
        for mb,diff,a in hypers:
            if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                continue
            binfo = get_bond_info(ob.OBAtomBondIter(a))
            for stretch,bdist,bond in binfo:
                #can we remove this bond without disconnecting the molecule?
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                #get right valence
                if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or \
                    a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                    #don't fragment the molecule
                    # if not self.reachable(a1,a2):
                    #     continue
                    mol.DeleteBond(bond)
                    if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                        break #let nbr atoms choose what bonds to throw out
        
        
        binfo = get_bond_info(ob.OBMolBondIter(mol))
        #now eliminate geometrically poor bonds
        for stretch,bdist,bond in binfo:
            #can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            #as long as we aren't disconnecting, let's remove things
            #that are excessively far away (0.45 from ConnectTheDots)
            #get bonds to be less than max allowed
            #also remove tight angles, because that is what ConnectTheDots does
            if stretch > self.max_bond_stretch or self.forms_small_angle(a1,a2) or self.forms_small_angle(a2,a1):
                #don't fragment the molecule
                if not self.reachable(a1,a2):
                    continue
                mol.DeleteBond(bond)

        mol.EndModify()
        ### Use the largest fragment if the mol is seperated
        if len(mol.Separate()) > 1:
            sep_mols = sorted([sep_mol for sep_mol in mol.Separate()], key=lambda x: x.NumAtoms())
            print("Using LCC with num_atoms: ", sep_mols[-1].NumAtoms())
            return sep_mols[-1]
            
        return mol
        
    def forms_small_angle(self, a, b):
        '''Return true if bond between a and b is part of a small angle
        with a neighbor of a only.'''

        cutoff=self.min_bond_angle
        for nbr in ob.OBAtomAtomIter(a):
            if nbr != b:
                degrees = b.GetAngle(a,nbr)
                if degrees < cutoff:
                    return True
        return False
    
    
    def reachable(self, a, b):
        '''Return true if atom b is reachable from a without using the bond between them.'''
        if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
            return False #this is the _only_ bond for one atom
        #otherwise do recursive traversal
        seenbonds = set([a.GetBond(b).GetIdx()])
        return self.reachable_r(a,b,seenbonds)
    
    def reachable_r(self, a, b, seenbonds):
        '''Recursive helper.'''

        for nbr in ob.OBAtomAtomIter(a):
            bond = a.GetBond(nbr).GetIdx()
            if bond not in seenbonds:
                seenbonds.add(bond)
                if nbr == b:
                    return True
                elif self.reachable_r(nbr,b,seenbonds):
                    return True
        return False

    
    def count_nbrs_of_elem(self, atom, atomic_num):
        '''
        Count the number of neighbors atoms
        of atom with the given atomic_num.
        '''
        count = 0
        for nbr in ob.OBAtomAtomIter(atom):
            if nbr.GetAtomicNum() == atomic_num:
                count += 1
        return count
    
    
    def calc_valence(self, rdatom):
        '''Can call GetExplicitValence before sanitize, but need to
        know this to fix up the molecule to prevent sanitization failures'''
        cnt = 0.0
        for bond in rdatom.GetBonds():
            cnt += bond.GetBondTypeAsDouble()
        return cnt
    
    
    def convert_ob_mol_to_rd_mol(self, ob_mol):
        '''
        Convert ob_mol to rd_mol
        '''
        ob_mol.DeleteHydrogens()
        n_atoms = ob_mol.NumAtoms()
        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)
        
        for ob_atom in ob.OBMolAtomIter(ob_mol):
            rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
            #TODO copy format charge
            if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
                #don't commit to being aromatic unless rdkit will be okay with the ring status
                #(this can happen if the atoms aren't fit well enough)
                rd_atom.SetIsAromatic(True)
            i = rd_mol.AddAtom(rd_atom)
            ob_coords = ob_atom.GetVector()
            x = ob_coords.GetX()
            y = ob_coords.GetY()
            z = ob_coords.GetZ()
            rd_coords = Geometry.Point3D(x, y, z)
            rd_conf.SetAtomPosition(i, rd_coords)
            
        rd_mol.AddConformer(rd_conf)
        
        for ob_bond in ob.OBMolBondIter(ob_mol):
            i = ob_bond.GetBeginAtomIdx()-1
            j = ob_bond.GetEndAtomIdx()-1
            bond_order = ob_bond.GetBondOrder()
            if bond_order == 1:
                rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
            elif bond_order == 2:
                rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond_order == 3:
                rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
            else:
                raise Exception('unknown bond order {}'.format(bond_order))

            if ob_bond.IsAromatic():
                bond = rd_mol.GetBondBetweenAtoms (i,j)
                bond.SetIsAromatic(True)
                
                
        rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

        pt = Chem.GetPeriodicTable()
        
        positions = rd_mol.GetConformer().GetPositions()
        nonsingles = []
        for bond in rd_mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                dist = np.linalg.norm(positions[i]-positions[j])
                nonsingles.append((dist,bond))
        nonsingles.sort(reverse=True, key=lambda t: t[0])
        
        for (d,bond) in nonsingles:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            if self.calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
               self.calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
                btype = Chem.BondType.SINGLE
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    btype = Chem.BondType.DOUBLE
                bond.SetBondType(btype)
                
                
        for atom in rd_mol.GetAtoms():
            #set nitrogens with 4 neighbors to have a charge
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
                atom.SetFormalCharge(1)
                
                
        rd_mol = Chem.AddHs(rd_mol,addCoords=True)

        positions = rd_mol.GetConformer().GetPositions()
        center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
        for atom in rd_mol.GetAtoms():
            i = atom.GetIdx()
            pos = positions[i]
            if not np.all(np.isfinite(pos)):
                #hydrogens on C fragment get set to nan (shouldn't, but they do)
                rd_mol.GetConformer().SetAtomPosition(i,center)
                
                
        try:
            Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
        except:
            raise MolReconsError()
            
        #but at some point stop trying to enforce our aromaticity -
        #openbabel and rdkit have different aromaticity models so they
        #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
        for bond in rd_mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if bond.GetIsAromatic():
                if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                    bond.SetIsAromatic(False)
            elif a1.GetIsAromatic() and a2.GetIsAromatic():
                bond.SetIsAromatic(True)
                
                
        return rd_mol
    

    
    def postprocess_rd_mol_1(self, rdmol):

        rdmol = Chem.RemoveHs(rdmol)

        # Construct bond nbh list
        nbh_list = {}
        for bond in rdmol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
            if begin not in nbh_list: nbh_list[begin] = [end]
            else: nbh_list[begin].append(end)

            if end not in nbh_list: nbh_list[end] = [begin]
            else: nbh_list[end].append(begin)

        # Fix missing bond-order
        for atom in rdmol.GetAtoms():
            idx = atom.GetIdx()
            num_radical = atom.GetNumRadicalElectrons()
            if num_radical > 0:
                for j in nbh_list[idx]:
                    if j <= idx: continue
                    nb_atom = rdmol.GetAtomWithIdx(j)
                    nb_radical = nb_atom.GetNumRadicalElectrons()
                    if nb_radical > 0:
                        bond = rdmol.GetBondBetweenAtoms(idx, j)
                        bond.SetBondType(self.UPGRADE_BOND_ORDER[bond.GetBondType()])
                        nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                        num_radical -= 1
                if num_radical > 0:
                    atom.SetNumRadicalElectrons(num_radical)

            num_radical = atom.GetNumRadicalElectrons()
            if num_radical > 0:
                atom.SetNumRadicalElectrons(0)
                num_hs = atom.GetNumExplicitHs()
                atom.SetNumExplicitHs(num_hs + num_radical)

        return rdmol
    
    
    def postprocess_rd_mol_2(self, rdmol):
        rdmol_edit = Chem.RWMol(rdmol)

        ring_info = rdmol.GetRingInfo()
        ring_info.AtomRings()
        rings = [set(r) for r in ring_info.AtomRings()]
        for i, ring_a in enumerate(rings):
            if len(ring_a) == 3:
                non_carbon = []
                atom_by_symb = {}
                for atom_idx in ring_a:
                    symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                    if symb != 'C':
                        non_carbon.append(atom_idx)
                    if symb not in atom_by_symb:
                        atom_by_symb[symb] = [atom_idx]
                    else:
                        atom_by_symb[symb].append(atom_idx)
                if len(non_carbon) == 2:
                    rdmol_edit.RemoveBond(*non_carbon)
                if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                    rdmol_edit.RemoveBond(*atom_by_symb['O'])
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                        rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                    )
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                        rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                    )
        rdmol = rdmol_edit.GetMol()

        for atom in rdmol.GetAtoms():
            if atom.GetFormalCharge() > 0:
                atom.SetFormalCharge(0)

        return rdmol

        
    def make_mol(self, atomic_numbers, positions):
        '''
        Creat molecules with added bonds
        atomic_numbers: [N]
        positions: [N, 3]
        '''
        xyz = positions.tolist()
        atomic_nums = atomic_numbers.tolist()
        
        mol, atoms = self.to_ob_mol(xyz, atomic_nums)
        self.fixup(mol)
        
        ob_mol = self.connect_the_dots(mol, atoms)
        self.fixup(ob_mol)
        mol.EndModify()
        
        
        self.fixup(ob_mol)
        
        ob_mol.AddPolarHydrogens()
        ob_mol.PerceiveBondOrders()
        self.fixup(ob_mol)
        
        
        for (i,a) in enumerate(atoms):
            ob.OBAtomAssignTypicalImplicitHydrogens(a)
        self.fixup(ob_mol)
        
            
        ob_mol.AddHydrogens()
        self.fixup(ob_mol)
        
        #make rings all aromatic if majority of carbons are aromatic
        for ring in ob.OBMolRingIter(ob_mol):
            if 5 <= ring.Size() <= 6:
                carbon_cnt = 0
                aromatic_ccnt = 0
                for ai in ring._path:
                    a = ob_mol.GetAtom(ai)
                    if a.GetAtomicNum() == 6:
                        carbon_cnt += 1
                        if a.IsAromatic():
                            aromatic_ccnt += 1
                if aromatic_ccnt >= carbon_cnt/2 and aromatic_ccnt != ring.Size():
                    #set all ring atoms to be aromatic
                    for ai in ring._path:
                        a = ob_mol.GetAtom(ai)
                        a.SetAromatic(True)
                        
                        
        #bonds must be marked aromatic for smiles to match        
        for bond in ob.OBMolBondIter(ob_mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsAromatic() and a2.IsAromatic():
                bond.SetAromatic(True)
                
        ob_mol.PerceiveBondOrders()
        
        
        rd_mol = self.convert_ob_mol_to_rd_mol(ob_mol)

        # Post-processing
        rd_mol = self.postprocess_rd_mol_1(rd_mol)
        rd_mol = self.postprocess_rd_mol_2(rd_mol)
        
        # rd_mol = Chem.RemoveHs(rd_mol)
        

        return rd_mol, ob_mol
