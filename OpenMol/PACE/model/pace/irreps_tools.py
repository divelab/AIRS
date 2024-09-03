###########################################################################################
# Adapted from MACE
###########################################################################################
import torch
from e3nn import o3


def get_feasible_irrep(irreps1, irreps2, target_irreps, mode="uvu", trainable=True, path_weights=None):
    irrep_mid = []
    instructions = []

    for i, (mul_in_i, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    if len(target_irreps) == 1:
                        mul_out = target_irreps[0].mul
                    else:
                        mul_out = target_irreps.count(ir_out)

                    if (mul_out, ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((mul_in_i, ir_out))
                    else:
                        k = irrep_mid.index((mul_out, ir_out))
                    if path_weights is None:
                        instructions.append((i, j, k, mode, trainable))
                    else:
                        instructions.append((i, j, k, mode, trainable, 1))

    irrep_mid = o3.Irreps(irrep_mid)
    irrep_mid, p, _ = irrep_mid.sort()
    return irrep_mid, instructions


class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor):
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)


def recollect_irreps(irreps_in):
    ir_collection_dir = {}
    for mul_ir, ir in irreps_in:
        if ir not in ir_collection_dir.keys():
            ir_collection_dir[ir] = mul_ir
        else:
            ir_collection_dir[ir] = mul_ir + ir_collection_dir[ir]

    ret_total_irreps_list = []
    for ir, ir_mul in ir_collection_dir.items():
        ret_total_irreps_list.append((ir_mul, ir))

    irreps_out = o3.Irreps(ret_total_irreps_list)
    irreps_out, _, _ = irreps_out.sort()
    return irreps_out


def recollect_features(irreps_features, irreps_in):
    irreps_out = recollect_irreps(irreps_in)
    irreps_feature_collection_dir = {}
    for slice_idx, slice in enumerate(irreps_in.slices()):
        slice_ir = irreps_in[slice_idx].ir
        if slice_ir not in irreps_feature_collection_dir.keys():
            irreps_feature_collection_dir[slice_ir] = [irreps_features[:, slice]]
        else:
            irreps_feature_collection_dir[slice_ir] += [irreps_features[:, slice]]

    for ir in irreps_feature_collection_dir.keys():
        irreps_feature_collection_dir[ir] = torch.cat(irreps_feature_collection_dir[ir], dim=-1)

    ret_irreps_features = []
    for mul, ir in irreps_out:
        ret_irreps_features.append(irreps_feature_collection_dir[ir])
    ret_irreps_features = torch.cat(ret_irreps_features, dim=-1)
    return ret_irreps_features, irreps_out
