from pdearena.modules.twod_resnet import DilatedBasicBlock


MODEL_REGISTRY = {
    ## sinenet: zeros
    "sinenet1-dual-128": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 128,
            "num_waves": 1,
            "mult": 2,
            "padding_mode": "zeros",
            "par1": 0
        },
    },
    "sinenet8-dual": { # 35490219
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "deeper_unet8-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "num_blocks": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "par1": 0
        },
    },
    "sinenet8-dual-tangle": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "disentangle": False,
            "par1": 0
        },
    },
    "sinenet6-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 6,
            "mult": 1.5,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "sinenet4-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 4,
            "mult": 1.611,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "sinenet2-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 2,
            "mult": 1.8075,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    ## sinenet: circular
    "sinenet1-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "mult": 2,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "sinenet8-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "deeper_unet8-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "num_blocks": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "sinenet8-dual-tangle-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "disentangle": False,
            "par1": 0
        },
    },
    "sinenet6-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 6,
            "mult": 1.5,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    "sinenet4-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 4,
            "mult": 1.611,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    "sinenet2-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 2,
            "mult": 1.8075,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    # SineNet-neural-ODE
    "sinenet-neural-ODE":{
        "class_path": "pdearena.modules.sinenet_neural_ode.sinenet_node",
        "init_args": {
            "tol": 0.01,
            "hidden_channels": 64,
            "mult": 1.7,
            "padding_mode": "zeros",
        },
    },
    # DilResNet
    "DilResNet-128-norm": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": True,
            "block": DilatedBasicBlock,
            "num_blocks": [1, 1, 1, 1],
            "padding_mode": "zeros"
        },
    },
    "DilResNet-128-circ-norm": {
        "class_path": "pdearena.modules.twod_resnet.ResNet",
        "init_args": {
            "hidden_channels": 128,
            "norm": True,
            "block": DilatedBasicBlock,
            "num_blocks": [1, 1, 1, 1],
            "padding_mode": "circular"
        },
    },
    # FFNO
    "FFNO-24-32-96-noShare": {
        "class_path": "pdearena.modules.ffno.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,
            "n_layers": 24,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "max_accumulations": 10000,
            "noise_std": 0.01,
            "should_normalize": True
        }
    },
    "FFNO-24-32-96-share": {
        "class_path": "pdearena.modules.ffno.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,
            "n_layers": 24,
            "share_weight": True,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "max_accumulations": 10000,
            "noise_std": 0.01,
            "should_normalize": True
        }
    },
    # UNet-Mod
    "Unetmod-64": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "padding_mode": "zeros"
        },
    },
    "Unetmod-64-circ": {
        "class_path": "pdearena.modules.twod_unet.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "padding_mode": "circular"
        },
    },
    # MP-PDE
    "mp_pde": {
        "class_path": "pdearena.modules.mp_pde.MP_PDE_Solver",
        "init_args": {
            "hidden_features": 128,
            "hidden_layer": 6,
        },
    },
    # DeepONet
    "deepONet-INS": {
        "class_path": "pdearena.modules.deepONet.deeponet",
        "init_args": {
            "hidden_channels": 64,
            "mult": 2,
            "padding_mode": "zeros",
            "nbasis": 512,
            "nx": 128,
            "ny":128
        },
    },
}
