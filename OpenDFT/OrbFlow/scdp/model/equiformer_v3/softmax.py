import torch


def _maybe_num_nodes(index: torch.Tensor, num_nodes=None) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def _scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int, reduce: str) -> torch.Tensor:
    index = index.reshape(-1)
    out_shape = (dim_size, *src.shape[1:])
    if reduce == "sum":
        out = src.new_zeros(out_shape)
        idx = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_add_(0, idx, src)
    if reduce == "max":
        out = src.new_full(out_shape, float("-inf"))
        idx = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
        out[out == float("-inf")] = 0.0
        return out
    raise ValueError(f"unsupported reduce={reduce!r}")


def _segment(src: torch.Tensor, ptr: torch.Tensor, reduce: str) -> torch.Tensor:
    out = []
    for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        chunk = src[start:end]
        if chunk.numel() == 0:
            out.append(src.new_zeros(1, *src.shape[1:]))
        elif reduce == "max":
            out.append(chunk.max(dim=0, keepdim=True).values)
        elif reduce == "sum":
            out.append(chunk.sum(dim=0, keepdim=True))
        else:
            raise ValueError(f"unsupported reduce={reduce!r}")
    return torch.cat(out, dim=0)


class SoftCap(torch.nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap


    def forward(self, inputs):
        outputs = inputs / self.cap
        outputs = torch.nn.functional.tanh(outputs)
        outputs = outputs * self.cap
        return outputs


    def __repr__(self):
        return f"{self.__class__.__name__}(cap={self.cap})"
        

class GraphSoftmax(torch.nn.Module):
    """
        1.  Reference: https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/utils/softmax.html
        2.  Add `exp_dropout` so that we can remove the contributions of some neighbors while keeping
            the sum equal to 1.
        3.  Add `exp_rescale` to rescale the outputs of the exponentation function so that we can downscale the 
            contributions of some neighbors with an envelope function.
        4.  Add `softcap` to limit input logits to the range [- `softcap`, + `softcap`].
        5.  Add `eps` for numerical stability.
    """
    def __init__(self, eps=1e-16, exp_dropout=0.0, softcap=None):
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = torch.nn.Dropout(exp_dropout) if self.exp_dropout > 0.0 else torch.nn.Identity()
        self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()


    def forward(
        self, 
        src, 
        index=None, 
        ptr=None, 
        num_nodes=None, 
        dim=0,
        exp_rescale=None
    ):
        src = self.softcap(src)
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            count = ptr[1:] - ptr[:-1]
            ptr = ptr.view(size)
            src_max = _segment(src.detach(), ptr, reduce="max")
            src_max = src_max.repeat_interleave(count, dim=dim)
            out = (src - src_max).exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out = self.dropout(out)
            out_sum = _segment(out, ptr, reduce="sum") + self.eps
            out_sum = out_sum.repeat_interleave(count, dim=dim)
        elif index is not None:
            N = _maybe_num_nodes(index, num_nodes)
            src_max = _scatter(src.detach(), index, dim_size=N, reduce="max")
            out = src - src_max.index_select(dim, index)
            out = out.exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out = self.dropout(out)
            out_sum = _scatter(out, index, dim_size=N, reduce="sum") + self.eps
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError
        
        out = out / out_sum
        
        return out
    

    def extra_repr(self):
        return 'eps={}'.format(self.eps)