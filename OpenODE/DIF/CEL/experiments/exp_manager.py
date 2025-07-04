
from .aphynity_exp import APHYNITYExp
from .soft_intervention_exp import *
from CEL.utils.register import register

def load_experiment(dataloader, model, exp_config):
    exp = register.experiments[exp_config.name](dataloader, model, exp_config)
    return exp