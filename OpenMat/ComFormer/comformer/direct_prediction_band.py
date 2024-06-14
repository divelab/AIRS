# direct prediction using iComFormer
import sys
sys.path.append("/yourpath/Comformer")
import pickle as pk
import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from comformer.graphs import PygGraph, PygStructureDataset
from torch.utils.data import DataLoader
from comformer.models.comformer import iComFormer
from ignite.handlers import Checkpoint
from pymatgen.core.structure import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
adaptor = JarvisAtomsAdaptor()

def atoms_to_graph(atoms):
    """Convert structure to Atom."""
    structure = adaptor.get_atoms(atoms)
    return PygGraph.atom_dgl_multigraph(
        structure,
        neighbor_strategy="k-nearest",
        cutoff=4.0,
        atom_features="atomic_number",
        max_neighbors=25,
        compute_line_graph=False,
        use_canonize=True,
        use_lattice=True,
        use_angle=False,
    )
# load data
with open("/data/yourpath/data.pkl", "rb") as f:
    dat_list = pk.load(f)

structures = []
for itm in dat_list:
    try:
        structures.append(Structure.from_str(itm, fmt="cif"))
    except:
        continue
graphs = [atoms_to_graph(itm) for itm in structures]

features = PygStructureDataset._get_attribute_lookup("cgcnn")

for g in graphs:
    z = g.x
    g.atomic_number = z
    z = z.type(torch.IntTensor).squeeze()
    f = torch.tensor(features[z]).type(torch.FloatTensor)
    if g.x.size(0) == 1:
        f = f.unsqueeze(0)
    g.x = f
    g.batch = torch.zeros(1, dtype=torch.int64)

net = iComFormer()
device = torch.device("cuda")
net.to(device)

checkpoint_tmp = torch.load('your_path/model_pth.pt')
to_load = {
    "model": net,
}
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_tmp)
net.eval()
predictions = []
import time
t1 = time.time()
with torch.no_grad():
    for dat in graphs:
        g = dat
        out_data = net([g.to(device), g.to(device), g.to(device)])
        out_data = out_data.cpu().numpy().tolist()
        predictions.append(out_data)
t2 = time.time()
from sklearn.metrics import mean_absolute_error
targets = torch.ones(len(predictions)) * 0.25
predictions = np.array(predictions) * 1.404828 + 0.68637 # bandgap
print(predictions)
print("Total test time:", t2-t1)