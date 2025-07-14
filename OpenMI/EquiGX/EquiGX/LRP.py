import torch
from e3nn import o3

def FirstOrder4NormActivation(act, features, R, irreps = None):
    
    norms = act.norm(features)
    if act._eps_squared > 0:
        # See TFN for the original version of this approach:
        # https://github.com/tensorfieldnetworks/tensorfieldnetworks/blob/master/tensorfieldnetworks/utils.py#L22
        norms[norms < act._eps_squared] = act._eps_squared
        norms = norms.sqrt()

    nonlin_arg = norms
    if act.bias:
        nonlin_arg = nonlin_arg + act.biases

    scalings = act.scalar_nonlinearity(nonlin_arg)
    if act.normalize:
        scalings = scalings / norms
        
    scalings = 1 / scalings 
    
    R_out = act.scalar_multiplier(scalings, R)
    return  R_out / (R_out.sum() / R.sum())

        
def FirstOrder4FullyConnectedTensorProduct(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point = False): # R (#nodes, #features(including different rotation orders))
    # mulu = max([ w.shape[0] for w in tp.weight_views(weight)] )
    # mulv = max([ w.shape[1] for w in tp.weight_views(weight)] )
    # mulw = max([ w.shape[2] for w in tp.weight_views(weight)] )
    # U,V,W = len(tp.irreps_in1), len(tp.irreps_in2), len(tp.irreps_out)

    epsilon = 1e-7
    node_features = node_features.clone()
    node_features.requires_grad_()
    out = tp(node_features[edge_src], edge_attr, weight) + epsilon

    R_tmp = (R / out).clone().detach()

    if zero_point:
        zero_input = torch.zeros_like(node_features)
        zero_input.requires_grad_()
        out = tp(zero_input[edge_src], edge_attr, weight)

        (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = zero_input, create_graph = True)
        
    else:
        (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = node_features, create_graph = True)

    return node_features * grads

def FirstOrder4FullyConnectedTensorProductWithEdgeOutput(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point = False): # R (#nodes, #features(including different rotation orders))
    
    def nodeFeature(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point = False):
        epsilon = 1e-7
        node_features = node_features.clone()
        node_features.requires_grad_()
        out = tp(node_features[edge_src], edge_attr, weight) + epsilon

        R_tmp = (R / out).clone().detach()

        if zero_point:
            zero_input = torch.zeros_like(node_features)
            zero_input.requires_grad_()
            out = tp(zero_input[edge_src], edge_attr, weight)

            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = zero_input, create_graph = True)

        else:
            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = node_features, create_graph = True)

        return node_features * grads
    
    def edgeAttr(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point = False):
        epsilon = 1e-7
        edge_attr = edge_attr.clone()
        edge_attr.requires_grad_()
        out = tp(node_features[edge_src], edge_attr, weight) + epsilon
        R_tmp = (R / out).clone().detach()
        
        if zero_point:
            zero_input = torch.zeros_like(edge_attr)
            zero_input.requires_grad_()
            out = tp(node_features[edge_src], zero_input, weight)

            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = zero_input, create_graph = True)

        else:
            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = edge_attr, create_graph = True)

        return edge_attr * grads
        
        
    def edgeDistance(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point = False):
        epsilon = 1e-7
        weight = weight.clone()
        weight.requires_grad_()
        out = tp(node_features[edge_src], edge_attr, weight) + epsilon
        R_tmp = (R / out).clone().detach()
        
        if zero_point:
            zero_input = torch.zeros_like(weight)
            zero_input.requires_grad_()
            out = tp(node_features[edge_src], edge_attr, zero_input)

            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = zero_input, create_graph = True)

        else:
            (grads,) = torch.autograd.grad(outputs= (out * R_tmp).sum(), inputs = weight, create_graph = True)

        return weight * grads
    
    return nodeFeature(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point) / 3, \
            edgeAttr(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point) / 3, \
            edgeDistance(tp, R, node_features, edge_src, edge_dst, edge_attr, weight, zero_point) / 3 
        
        

def FirstOrder4Scatter(R, edge_dst, edge_features): # Z rule
    epsilon = 1e-7
    total_edge_features_per_node = torch.zeros_like(R).to(R.device) + epsilon 
    total_edge_features_per_node.index_add_(0, edge_dst, edge_features)
    
    # Compute the proportion of each edge feature relative to its node's total
    proportions = edge_features / (torch.index_select(total_edge_features_per_node, 0, edge_dst) + epsilon)
    
    # Distribute the node scores to edges based on the proportions
    distributed_edge_features = proportions * torch.index_select(R, 0, edge_dst)
    return distributed_edge_features
    
def FirstOrder4Linear(R, linear, input):
    epsilon = 1e-7
    R_ret = torch.zeros_like(R).to(R.device)
    B = input.shape[0]
    output = linear(input)
    for idx, instruction in enumerate(linear.instructions):

        irreps_in, irreps_out, path_shape, path_weight = instruction
        W = linear.weight_view_for_instruction(idx)
        n = W.shape[0]
        dim = linear.irreps_in[irreps_in].ir.dim
        slice = linear.irreps_in.slices()[idx]
        
        norm_term = output[:, slice].reshape(B,n,dim).clone() # B,n,dim
        tmp_input = input[:, slice].reshape(B,n,dim)
        tmp_R = R[:,slice].reshape(B,n,dim)
        tmp_R_ret = []
        for i in range(n): 
            contribution = torch.matmul( W.T[:,[i]], tmp_input[:,[i],:]) * path_weight # B,n,dim
            tmp = contribution / (norm_term + epsilon) * tmp_R #B,n,dim
            tmp = tmp.sum(dim = 1) # B,1,dim
            tmp_R_ret.append(tmp.clone())
        tmp_R_ret = torch.cat(tmp_R_ret, dim = 1) # B,n,dim
        
        R_ret[:,slice] = tmp_R_ret.reshape(B,-1)

    return R_ret