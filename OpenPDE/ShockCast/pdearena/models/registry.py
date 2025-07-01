UNET = {
    "Unetmod-cond1-v2-64_4_1": {
        "class_path": "pdearena.models.unet.time_cond_LN.twod_unet_cond_v2.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 1
        },
    },
    "Unetmod-cond2-v2-64_4_1": {
        "class_path": "pdearena.models.unet.time_cond_LN.twod_unet_cond_v2.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 2
        },
    },
    "Unetmod-cond2-v5-MOEx4-euler-32_4_1": {
        "class_path": "pdearena.models.unet.moe.twod_unet_cond_v5_moeuler.Unet",
        "init_args": {
            "num_experts": 4,
            "hidden_channels": 32,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 2
        },
    },
    "Unetmod-cond1-v5-MOEx4-euler-32_4_1": {
        "class_path": "pdearena.models.unet.moe.twod_unet_cond_v5_moeuler.Unet",
        "init_args": {
            "num_experts": 4,
            "hidden_channels": 32,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 1
        },
    },
    "Unetmod-cond2-v3-euler-64_4_1": {
        "class_path": "pdearena.models.unet.euler.twod_unet_cond_v3_euler.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 2
        },
    },
    "Unetmod-cond1-v3-euler-64_4_1": {
        "class_path": "pdearena.models.unet.euler.twod_unet_cond_v3_euler.Unet",
        "init_args": {
            "hidden_channels": 64,
            "norm": True,
            "ch_mults": (2, 2, 2, 2),
            "is_attn": (False, False, False, False),
            "sine_spacing": 1e-2,
            "n_blocks": 1,
            "use_scale_shift_norm": True,
            "num_param_conditioning": 1
        },
    },

}

FFNO = {
    "FFNO_cond2_spatial_spectral": {
        "class_path": "pdearena.models.ffno.time_cond_LN.ffno_cond_spatial_spectral.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,            
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 2,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        },
    },
    "FFNO_cond1_spatial_spectral": {
        "class_path": "pdearena.models.ffno.time_cond_LN.ffno_cond_spatial_spectral.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,            
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 1,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        },
    },
    "FFNO_cond2_euler_residual": {
        "class_path": "pdearena.models.ffno.euler.ffno_cond_euler_residual.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,            
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 2,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        },
    },
    "FFNO_cond1_euler_residual": {
        "class_path": "pdearena.models.ffno.euler.ffno_cond_euler_residual.FFNO",
        "init_args": {
            "modes": 32,
            "width": 96,            
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 1,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        },
    },
    "FFNO_cond2_moeulerx4": {
        "class_path": "pdearena.models.ffno.moe.ffno_cond_moeuler.FFNO",
        "init_args": {
            "modes": 32,
            "width": 48,         
            "num_experts": 4,
            "gate_channels": 96,  
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 2,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        }
    },
    "FFNO_cond1_moeulerx4": {
        "class_path": "pdearena.models.ffno.moe.ffno_cond_moeuler.FFNO",
        "init_args": {
            "modes": 32,
            "width": 48,         
            "num_experts": 4,
            "gate_channels": 96,  
            "n_layers": 12,
            "share_weight": False,
            "factor": 4,
            "ff_weight_norm": True,
            "gain": 0.1,
            "dropout": 0.0,
            "in_dropout": 0.0,
            "sinusoidal_embedding": False,
            "sine_spacing": 1e-2,
            "num_param_conditioning": 1,
            "diffusion_time_conditioning": False,
            "use_dt": True,
            "activation": "gelu",
        }
    }
}

CNO = {
    "CNO-v2-L-cond2": {
        "class_path": "pdearena.models.cno.time_cond_LN.CNO_timeModule_CINv2.CNO_time",
        "init_args": {
            "num_param_conditioning": 2,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },
    "CNO-v2-L-cond1": {
        "class_path": "pdearena.models.cno.time_cond_LN.CNO_timeModule_CINv2.CNO_time",
        "init_args": {
            "num_param_conditioning": 1,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },
    "CNO-v2-L-cond2-euler": {
        "class_path": "pdearena.models.cno.euler.CNO_timeModule_CINv2_euler.CNO_time",
        "init_args": {
            "num_param_conditioning": 2,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },
    "CNO-v2-L-cond1-euler": {
        "class_path": "pdearena.models.cno.euler.CNO_timeModule_CINv2_euler.CNO_time",
        "init_args": {
            "num_param_conditioning": 1,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },
    "CNO-v2-L-cond2-moeulerx4": {
        "class_path": "pdearena.models.cno.moe.CNO_timeModule_CINv2_moeuler.CNO_time",
        "init_args": {
            "num_param_conditioning": 2,
            "num_experts": 4,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },
    "CNO-v2-L-cond1-moeulerx4": {
        "class_path": "pdearena.models.cno.moe.CNO_timeModule_CINv2_moeuler.CNO_time",
        "init_args": {
            "num_param_conditioning": 1,
            "num_experts": 4,
            "channel_multiplier": 54,
            "N_res": 6,
            "N_res_neck": 6,
            "activation": "cno_lrelu_torch"
        }
    },

}

CONVNEXT = {
    "convNeXT-tiny":{
        "class_path": "pdearena.models.cfl.convNeXT.ConvNeXt",
        "init_args": {
            "depths": (3, 3, 9, 3), 
            "dims": [96, 192, 384, 768],
            "num_param_conditioning": 0,
        }
    },
    "convNeXT-tiny-max":{
        "class_path": "pdearena.models.cfl.convNeXT.ConvNeXt",
        "init_args": {
            "max_pool": True,
            "depths": (3, 3, 9, 3), 
            "dims": [96, 192, 384, 768],
            "num_param_conditioning": 0,
        }
    }
}


TRANSOLVER = {
    "Transolver_cond2": {
        "class_path": "pdearena.models.transolver.time_cond_LN.Transolver_Structured_Mesh_2D_cond.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 2,
            "n_hidden": 224,
        }
    },
    "Transolver_cond1": {
        "class_path": "pdearena.models.transolver.time_cond_LN.Transolver_Structured_Mesh_2D_cond.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 1,
            "n_hidden": 224,
        }
    },
    "Transolver_cond2_euler": {
        "class_path": "pdearena.models.transolver.euler.Transolver_Structured_Mesh_2D_euler.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 2,
            "n_hidden": 224,
        }
    },
    "Transolver_cond1_euler": {
        "class_path": "pdearena.models.transolver.euler.Transolver_Structured_Mesh_2D_euler.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 1,
            "n_hidden": 224,
        }
    },
    "Transolver_cond2_moeulerx4": {
        "class_path": "pdearena.models.transolver.moe.Transolver_Structured_Mesh_2D_moeuler.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 2,
            "n_hidden": 158,
            "num_experts": 2,
        }
    },
    "Transolver_cond1_moeulerx4": {
        "class_path": "pdearena.models.transolver.moe.Transolver_Structured_Mesh_2D_moeuler.Model",
        "init_args": {
            "n_layers": 8,
            "num_param_conditioning": 1,
            "n_hidden": 158,
            "num_experts": 2,
        }
    },
}


MODEL_REGISTRY = {
    **CNO,
    **FFNO,
    **UNET,
    **TRANSOLVER,
    **CONVNEXT
}

if __name__ == "__main__":
    # %%
    from pdearena.models.registry import MODEL_REGISTRY
    from importlib import import_module
    # %%
    missing = []
    for k, v in MODEL_REGISTRY.items():
        try:
            package, name = v["class_path"].rsplit(".", 1)
            import_module(package, name)
        except ModuleNotFoundError:
            missing.append(k)
    # %%
    print(missing)
    # %%
    len(missing)
    # %%
