import sys
sys.path.append("/yourpath/comformer")
from comformer.train_props import train_prop_model 
props = [
    "e_form",
    "gap pbe",
    "bulk modulus",
    "shear modulus",
]
train_prop_model(learning_rate=0.001,name="iComformer", dataset="megnet", prop=props[0], pyg_input=True, n_epochs=500, max_neighbors=25, cutoff=4.0, batch_size=64, use_lattice=True, output_dir="yourpath", use_angle=False, save_dataloader=False)