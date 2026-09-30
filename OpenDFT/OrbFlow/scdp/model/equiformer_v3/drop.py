import torch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class GraphDropPath(torch.nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob


    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size, ) + (1, ) * (x.ndim - 1)  # work with different dim tensors
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out


    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)


class EquivariantDropout(torch.nn.Module):
    """
        1.  When dropping one type-L vector, we set all the m components to zeros.
    """
    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False):
        super(EquivariantDropout, self).__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.drop_prob = drop_prob
        self.use_m_primary = use_m_primary

        self.drop = torch.nn.Dropout(drop_prob, True)
        
        expand_index = []
        if not self.use_m_primary:
            for l in range(self.lmax + 1):
                mmax = min(l, self.mmax)
                l_index_tensor = torch.ones(((2 * mmax + 1), ), dtype=torch.long) * l
                expand_index.append(l_index_tensor)
        elif self.use_m_primary:
            for m in range(self.mmax + 1):
                l_index = torch.arange((self.lmax + 1 - m))
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)    # +- m
        expand_index = torch.cat(expand_index, dim=0)
        expand_index = expand_index.long()
        self.register_buffer('expand_index', expand_index)


    def extra_repr(self):
        return 'lmax={}, mmax={}, drop_prob={}, use_m_primary={}'.format(
            self.lmax, self.mmax, self.drop_prob, self.use_m_primary
        )
    
        
    def forward(self, x):
        # x shape: (num_tokens, num_m_coefficients, num_channels)
        if not self.training or self.drop_prob == 0.0:
            return x
        
        assert len(x.shape) == 3
        shape = (x.shape[0], (self.lmax + 1), x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        mask = torch.index_select(mask, dim=1, index=self.expand_index)
        out = x * mask
        return out