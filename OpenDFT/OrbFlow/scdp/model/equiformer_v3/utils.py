import torch


def reduce_edge(inputs, edge_index, output_shape):
    outputs = torch.zeros(
        *output_shape,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    outputs.index_add_(0, edge_index, inputs)
    return outputs