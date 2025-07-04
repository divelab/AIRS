from CEL.utils.register import register
def load_model(dataset, ds_config, model_config):
    model_name = model_config.name
    model = register.models[model_name](ds_config.name, dataset=dataset, **model_config[model_name])
    return model