import os.path as osp
import h5py
import numpy as np
import warnings
from tqdm import tqdm

import torch 
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class FOLDBackBonedataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='train'
                ):

        self.split = split
        self.root = root

        super(FOLDBackBonedataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed_backbone'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'

    def atom_data(self, pFilePath):
        h5File = h5py.File(pFilePath, "r")
        data = Data()
        
        coordinates = h5File['atom_pos'][0,:]           #size: (n_atom,3)
        atom_names = h5File['atom_names'][:]            #size: (n_atom,)
        residue_names = h5File['atom_residue_names'][:] #size: (n_atom,)
        residue_nums = h5File['atom_residue_id'][:]     #size: (n_atom,)
        atom_types = h5File['atom_types'][:]            #size: (n_atom,)
        
        data.x = torch.unsqueeze(torch.tensor(atom_types),1)
        data.pos = torch.tensor(coordinates)
        
        h5File.close()
        return data
    
    def backbone_data(self, pFilePath):
        h5File = h5py.File(pFilePath, "r")
        data = Data()
        
        coordinates = h5File['atom_pos'][0,:]           #size: (n_atom,3)
        atom_names = h5File['atom_names'][:]            #size: (n_atom,)
        residue_names = h5File['atom_residue_names'][:] #size: (n_atom,)
        residue_nums = h5File['atom_residue_id'][:]     #size: (n_atom,)
        atom_types = h5File['atom_types'][:]            #size: (n_atom,)
        amino_types = h5File['amino_types'][:]          #size: (n_residue,)
        
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        
        mask = np.ma.mask_or(np.ma.mask_or(mask_n, mask_ca), mask_c)
        
        atom_type_tensor = F.one_hot(torch.tensor(atom_types[mask]), num_classes = 16) # atom type 
        residue_idx_tensor = torch.unsqueeze(torch.tensor(residue_nums[mask]),1) # residue idx
        data.x = torch.cat([atom_type_tensor, residue_idx_tensor], 1)
        data.pos = torch.FloatTensor(coordinates[mask])
        
        xai_labels_raw = np.load(pFilePath[:-4] + "npy") # size: (n_residue,)
        residue_idx = residue_nums[mask]
        # print("residue length")
        # print("h5 file: ", len(amino_types), max(residue_idx))
        # print("DSSP results,", len(xai_labels_raw))
        data.xai_labels_raw = (xai_labels_raw[residue_idx]).reshape(-1,1)
        xai_labels = np.zeros(data.xai_labels_raw.shape)
        xai_labels[data.xai_labels_raw == 'alpha'] = 1.0
        xai_labels[data.xai_labels_raw == 'beta'] = 1.0       
        data.xai_labels = torch.tensor(xai_labels)
        
        h5File.close()
        return data
        

    
    def process(self):
        print('Beginning Processing ...')

        # Load the file with the list of functions.        
        classes_ = {}
        with open(self.root+"/class_map.txt", 'r') as mFile:    
            for line in mFile:
                lineList = line.rstrip().split('\t')
                classes_[lineList[0]] = ord(lineList[0][0]) - ord('a')

        # Get the file list.
        fileList_ = []
        cathegories_ = []
        from pathlib import Path
        
        root_path = Path(self.root+"/"+self.split)
        all_files = list(root_path.rglob('*.npy'))
        all_files = [str(file)[:-4] for file in all_files if file.is_file()]

        with open(self.root+"/"+self.split+".txt", 'r') as mFile:
            for curLine in mFile:
                splitLine = curLine.rstrip().split('\t')
                curClass = classes_[splitLine[-1]]
                # fileList_.append(self.root+"/"+self.split+"/"+splitLine[0])
                if (self.root+"/"+self.split+"/"+splitLine[0]) in all_files:
                    fileList_.append(self.root+"/"+self.split+"/"+splitLine[0])
                    cathegories_.append(curClass)
                    

        # Load the dataset
        print("Reading the data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_list = []
            for fileIter, curFile in tqdm(enumerate(fileList_)):
                fileName = curFile.split('/')[-1]
                curProtein = self.backbone_data(curFile+".hdf5")   
                curProtein.id = fileName           
                curProtein.y = torch.tensor(cathegories_[fileIter])
                if not curProtein.x is None:
                    data_list.append(curProtein)   
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')
 
if __name__ == "__main__":
    for split in ['training', 'validation', 'test_fold', 'test_superfamily', 'test_family']:
        print('#### Now processing {} data ####'.format(split))
        dataset = FOLDBackBonedataset(root='/data/hongyiling/XAI4e3nn/dataset/scope/', split=split)
        print(dataset)