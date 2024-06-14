"""Helper function for high-throughput GNN trainings."""
"""Implementation based on the template of Matformer."""
import matplotlib.pyplot as plt

# import numpy as np
import time
from comformer.train import train_main

# from sklearn.metrics import mean_absolute_error
plt.switch_backend("agg")


def train_prop_model(
    prop="",
    dataset="dft_3d",
    write_predictions=True,
    name="iComformer",
    save_dataloader=False,
    train_ratio=None,
    classification_threshold=None,
    val_ratio=None,
    test_ratio=None,
    learning_rate=0.001,
    batch_size=None,
    scheduler=None,
    n_epochs=None,
    id_tag=None,
    num_workers=None,
    weight_decay=None,
    edge_input_features=None,
    triplet_input_features=None,
    embedding_features=None,
    hidden_features=None,
    output_features=None,
    random_seed=None,
    n_early_stopping=None,
    cutoff=None,
    max_neighbors=None,
    matrix_input=False,
    pyg_input=False,
    use_lattice=False,
    use_angle=False,
    output_dir=None,
    neighbor_strategy="k-nearest",
    test_only=False,
    use_save=True,
    mp_id_list=None,
    file_name=None,
    atom_features="cgcnn",
):
    """Train models for a dataset and a property."""
    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 64
    if n_epochs is None:
        n_epochs = 500
    if num_workers is None:
        num_workers = 4
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "criterion": "mse",
        # "criterion": "l1",
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "atom_features": atom_features,
        "model": {
            "name": name,
        },
    }
    if n_early_stopping is not None:
        config["n_early_stopping"] = n_early_stopping
    if cutoff is not None:
        config["cutoff"] = cutoff
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if edge_input_features is not None:
        config["model"]["edge_input_features"] = edge_input_features
    if hidden_features is not None:
        config["model"]["hidden_features"] = hidden_features
    if embedding_features is not None:
        config["model"]["embedding_features"] = embedding_features
    if output_features is not None:
        config["model"]["output_features"] = output_features
    if random_seed is not None:
        config["random_seed"] = random_seed
    if file_name is not None:
        config["filename"] = file_name
    config["matrix_input"] = matrix_input
    config["pyg_input"] = pyg_input
    config["use_lattice"] = use_lattice
    config["use_angle"] = use_angle
    config["model"]["use_angle"] = use_angle
    config["neighbor_strategy"] = neighbor_strategy
    # config["output_dir"] = '.'
    if output_dir is not None:
        config["output_dir"] = output_dir
    
    if id_tag is not None:
        config["id_tag"] = id_tag
    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")

        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio
    if dataset == "jv_3d":
        config["num_workers"] = 4
        config["pin_memory"] = False

    if dataset == "mp_3d_2020":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet2":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239
            # config["learning_rate"] = 0.01
            # config["epochs"] = 300
            config["num_workers"] = 8
        else:
            config["n_train"] = 4664
            config["n_val"] = 393
            config["n_test"] = 393
    
    if dataset == 'mpf':
        config["id_tag"] = "id"
        config["n_train"] = 169516
        config["n_val"] = 9417
        config["n_test"] = 9417
        
    if test_only:
        t1 = time.time()
        result = train_main(config, test_only=test_only, use_save=use_save, mp_id_list=mp_id_list)
        t2 = time.time()
        print("test mae=", result)
        print("Toal time:", t2 - t1)
        print()
        print()
        print()
    else:
        t1 = time.time()
        result = train_main(config, use_save=use_save, mp_id_list=mp_id_list)
        t2 = time.time()
        print("train=", result["train"])
        print("validation=", result["validation"])
        print("Toal time:", t2 - t1)
        print()
        print()
        print()
