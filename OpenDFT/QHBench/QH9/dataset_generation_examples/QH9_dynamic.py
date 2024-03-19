import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch_geometric.data import InMemoryDataset, download_url

import pyscf
from pyscf import dft
import pyscf.md as md
import apsw
import os
BOHR2ANG = 1.8897259886


class Frame:
    def __init__(self,
                 ekin=None,
                 epot=None,
                 coord=None,
                 veloc=None,
                 time=None,
                 converged=None,
                 Hamiltonian=None):
        self.ekin = ekin
        self.epot = epot
        self.etot = self.ekin + self.epot
        self.coord = coord
        self.veloc = veloc
        self.time = time
        self.Hamiltonian = Hamiltonian
        self.converged = converged


def _toframe(integrator):
    '''Convert an _Integrator to a Frame given current saved data.

    Args:
        integrator : md.integrator._Integrator object

    Returns:
        Frame with all data taken from the integrator.
    '''
    return Frame(
        ekin=integrator.ekin,
        epot=integrator.epot,
        coord=integrator.mol.atom_coords(),
        veloc=integrator.veloc,
        time=integrator.time,
        Hamiltonian=integrator.scanner.base.get_fock(),
        converged=integrator.scanner.converged
    )


class MyNVE(pyscf.md.NVE):
    def __init__(self, *args, **kwargs):
        super(MyNVE, self).__init__(*args, **kwargs)

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new geometry,
         then the next acceleration, and finally the velocity, all according
         to the Velocity Verlet algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()

        else:
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel
        return _toframe(self)


class GenerateQH9Dynamic(InMemoryDataset):
    def __init__(self, root='dataset/', db_idx=0, db_split=1,
                 dt=50, steps=100, temperate=300, verbose=0,
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'QH9Dynamic')
        self.db_idx = db_idx
        self.db_split = db_split  # how many MD is included in one sub database

        self.db_dir = os.path.join(self.folder, 'processed', f'{db_idx}')
        if not os.path.isdir(self.db_dir):
            os.makedirs(self.db_dir)

        self.db_path = os.path.join(self.db_dir, f"QH9Dynamic_{self.db_idx}.db")

        # hyperparameters for MD simulation with default NVE
        self.dt = dt                # The unit of time in pyscf is atomic_units (https://en.wikipedia.org/wiki/Atomic_units)
        self.steps = steps          # number of steps for MD
        self.temperate = temperate  # Temperate for NVE constrain

        self.verbose = verbose      # whether to prin the log information
        super(GenerateQH9Dynamic, self).__init__(self.folder, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return [os.path.join(f"{self.db_idx}", f"QH9Dynamic_{self.db_idx}.pkl"),
                os.path.join(f"{self.db_idx}", f"QH9Dynamic_{self.db_idx}.db")]

    def download(self):
        download_url(self.url, self.raw_dir)

    def is_isomer(self, Z_molecule):
        if (Z_molecule == 1).sum() == 10 and (Z_molecule == 6).sum() == 7 and (Z_molecule == 8).sum() == 2:
            return True
        else:
            return False

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))
        R = data['R']
        Z = data['Z']
        N = data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z, split)
        self.R_qm9 = R_qm9
        self.Z_qm9 = Z_qm9

        rng = np.random.default_rng(43)
        molecule_indices_sampled = rng.choice(np.arange(len(split)), size=10000, replace=False)
        target = {}
        names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        for name in names:
            target[name] = np.expand_dims(data[name], axis=-1)

        self.target = target
        self.names = names

        self.traj_file = os.path.join(self.db_dir, f"QH9Dynamic_{self.db_idx}.pkl")
        self.db = os.path.join(self.db_dir, f"QH9Dynamic_{self.db_idx}.db")
        self._open()
        split_md_info = []
        for i, example_id in tqdm(enumerate(
                molecule_indices_sampled[self.db_split * self.db_idx:self.db_split * (self.db_idx + 1)])):
            single_md_info = self.get_mol(example_id)
            split_md_info.append(single_md_info)
            if (i + 1) % self.db_split == 0 or i == len(molecule_indices_sampled) - 1:
                print(f"Saving QH9_md_{self.db_idx}...")
                with open(self.traj_file, 'wb') as f:
                    pickle.dump(split_md_info, f)

    def _open(self):
        newdb = not os.path.isfile(self.db_path)
        cursor = apsw.Connection(self.db_path, flags=apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE).cursor()
        if newdb:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS data (id INTEGER NOT NULL, geo_id INTEGER NOT NULL, N INTEGER, Z BLOB, pos BLOB, ekin FLOAT, epot FLOAT, etot FLOAT, time FLOAT, Ham BLOB, converged INT)")

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
        t = [[self.Z_qm9[idx][atom_idx], self.R_qm9[idx][atom_idx]]
             for atom_idx in range(self.Z_qm9[idx].shape[0])]
        mol.build(verbose=self.verbose, atom=t, basis='def2svp', unit='ang')
        self.cursor = apsw.Connection(self.db_path, flags=apsw.SQLITE_OPEN_READWRITE).cursor()

        mf_hf = dft.RKS(mol)
        mf_hf.xc = 'b3lyp'
        init_veloc = md.distributions.MaxwellBoltzmannVelocity(mol, T=self.temperate)
        myintegrator = MyNVE(mf_hf, dt=self.dt, steps=self.steps, veloc=init_veloc)
        myintegrator.run(verbose=self.verbose)

        self.cursor.execute('''BEGIN''')
        self.cursor.execute('pragma busy_timeout=2147483647')
        md_info_all = []
        for iteration, frame in enumerate(myintegrator):
            md_info = [
                idx, iteration, frame.coord, frame.ekin, frame.epot, frame.etot,
                frame.veloc, frame.time, int(frame.converged)]
            md_info_all.append(md_info)
            try:
                self.cursor.execute(
                    "INSERT INTO data (id, geo_id, N, Z, pos, ekin, epot, etot, time, Ham, converged) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (idx, iteration, len(self.Z_qm9[idx]), self._blob(self.Z_qm9[idx]),
                     frame.coord, frame.ekin, frame.epot, frame.etot, frame.time,
                     self._blob(frame.Hamiltonian), int(frame.converged))
                )
            except Exception as exc:
                self.cursor.execute("ROLLBACK")
                raise exc
        self.cursor.execute("COMMIT")
        return md_info_all


if __name__ == '__main__':
    # simple example with dt=50, and steps=2, where NVE is set to 300K
    dataset = GenerateQH9Dynamic(db_idx=0, db_split=1, dt=50, steps=2, temperate=300)
