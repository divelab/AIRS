import torch
import torch.nn as nn
from .utils import prod
import e3nn.o3 as o3


class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    result = torch.einsum(
                        f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
                            reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
                )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                           device=x_in.device).type(x_in.type())]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'
