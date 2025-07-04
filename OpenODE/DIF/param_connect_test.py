import torch
import torch.nn as nn

r'''
Minimal implementation to test: Fast parameter insert for PyTorch modules (Forecaster).
'''

def replace_params_with_custom_tensor(module, custom_tensor_func):
    for name, param in list(module.named_parameters(recurse=False)):
        # Delete the original parameter
        delattr(module, name)

        # Create a custom tensor using the provided function
        custom_tensor = custom_tensor_func(param)

        # Assign the custom tensor to the original parameter address
        setattr(module, name, custom_tensor)

    for child_name, child_module in module.named_children():
        replace_params_with_custom_tensor(child_module, custom_tensor_func)


# Example custom tensor function
def custom_tensor_func(param):
    return torch.zeros_like(param)  # Example: replace with zeros


# Example model with nested modules
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Sequential(
        nn.ReLU(),
        nn.Linear(20, 10)
    )
)

# Replace all parameters in the model
replace_params_with_custom_tensor(model, custom_tensor_func)

# Verify the changes
for name, param in model.named_parameters():
    print(f"{name}: {param}")