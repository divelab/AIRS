import torch
import torch.nn
import torch.utils.data


def compute_forces(energy, positions, training=True):
    grad_outputs = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  
        inputs=[positions],  
        grad_outputs=grad_outputs,
        retain_graph=training, 
        create_graph=training,  
        allow_unused=True,  
    )[0]  
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def get_edge_vectors_and_lengths(
    positions, 
    edge_index,
    normalize=False,
    eps=1e-9,
):
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths
