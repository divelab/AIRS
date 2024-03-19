import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch_geometric.data import InMemoryDataset, download_url

import pyscf
from pyscf import dft
import apsw
import os
BOHR2ANG = 1.8897259886


class GenerateQH9Stable(InMemoryDataset):
    def __init__(self, root='dataset/', db_idx=0, db_split=5000, verbose=0, transform=None, pre_transform=None, pre_filter=None):
        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.db_idx = db_idx
        self.db_split = db_split  # how many molecules are included in one sub database
        self.verbose = verbose    # 4 -> log; 0 -> no log
        self.folder = osp.join(root, 'QH9Stable')
        self.db_dir = os.path.join(self.folder, 'processed', f"{self.db_idx}")
        if not os.path.isdir(self.db_dir):
            os.makedirs(self.db_dir)

        self.db_path = os.path.join(self.db_dir, f"db_{self.db_idx}.db")
        super(GenerateQH9Stable, self).__init__(self.folder, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return os.path.join(self.db_dir, f"db_{self.db_idx}.db")

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))
        R = data['R']
        Z = data['Z']
        N = data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z, split)

        target = {}
        names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        for name in names:
            target[name] = np.expand_dims(data[name], axis=-1)
        self.R_qm9 = R_qm9
        self.Z_qm9 = Z_qm9
        self.target = target
        self.names = names

        self._open()
        cursor = apsw.Connection(self.db_path, flags=apsw.SQLITE_OPEN_READWRITE).cursor()
        cursor.execute('''BEGIN''')
        self.cursor = cursor
        for i in tqdm(range(self.db_split * self.db_idx, self.db_split * (self.db_idx + 1))):
            self.get_mol(i)
            if (i + 1) % self.db_split == 0 or i == len(N) - 1:
                self.cursor.execute("COMMIT")
                print(f'Saving db_{self.db_idx}...')

    def _open(self):
        newdb = not os.path.isfile(self.db_path)
        cursor = apsw.Connection(self.db_path, flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE).cursor()
        if newdb:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS data (id INTEGER NOT NULL PRIMARY KEY,  N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")

    def _blob(self, array):
        """Convert numpy array to blob/buffer object."""
        if array is None:
            return None
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    def _deblob(self, buf, dtype=np.float32, shape=None):
        """Convert blob/buffer object to numpy array."""
        if buf is None:
            return np.zeros(shape)
        array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return array

    def get_mol(self, idx):
        mol = pyscf.gto.Mole()
        t = [[self.Z_qm9[idx][atom_idx], self.R_qm9[idx][atom_idx]] for atom_idx in range(self.Z_qm9[idx].shape[0])]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')
        mf_hf = dft.RKS(mol)
        mf_hf.xc = 'b3lyp'
        mf_hf.kernel()
        Ham = mf_hf.get_fock()
        try:
            self.cursor.execute("INSERT INTO data (id, N, Z, pos, Ham) VALUES (?,?,?,?,?)",
                                (idx, len(self.Z_qm9[idx]), self._blob(self.Z_qm9[idx]), self._blob(self.R_qm9[idx]),
                                 self._blob(Ham)))
        except Exception as exc:
            self.cursor.execute("ROLLBACK")
            raise exc


if __name__ == '__main__':
    # an example of running two molecules for QM9 dataset, verbose is set to 4 to print the log information
    dataset = GenerateQH9Stable(db_idx=0, db_split=2, verbose=4)
