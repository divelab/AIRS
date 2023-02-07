from matformer.train_props import train_prop_model 
props = [
    "e_form",
    "gap pbe",
    # "mu_b",
    "bulk modulus",
    "shear modulus",
    # "elastic anisotropy",
]
train_prop_model(learning_rate=0.001,name="pygatt", dataset="megnet", prop=props[0], pyg_input=True, n_epochs=500, batch_size=64, use_lattice=True, output_dir="./matformer_mp_formation", use_angle=False, save_dataloader=False)
