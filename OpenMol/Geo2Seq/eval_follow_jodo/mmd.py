# Geometric substructure mmd evaluation; More advanced MMD metric could be used here.

import torch


def compute_mmd(source, target, batch_size=1000, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    From DIG.
    Calculate the `maximum mean discrepancy distance <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>`_
    between two sample set.
    This implementation is based on `this open source code <https://github.com/ZongxianLee/MMD_Loss.Pytorch>`_.
    Args:
        source (pytorch tensor): the pytorch tensor containing data samples of the source distribution.
        target (pytorch tensor): the pytorch tensor containing data samples of the target distribution.
    :rtype:
        :class:`float`
    """

    n_source = int(source.size()[0])
    n_target = int(target.size()[0])
    n_samples = n_source + n_target

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth, id = 0.0, 0
        while id < n_samples:
            bandwidth += torch.sum((total0 - total1[id:id + batch_size]) ** 2)
            id += batch_size
        bandwidth /= n_samples ** 2 - n_samples

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    XX_kernel_val = [0 for _ in range(kernel_num)]
    for i in range(kernel_num):
        XX_kernel_val[i] += torch.sum(
            torch.exp(-((total0[:, :n_source] - total1[:n_source, :]) ** 2) / bandwidth_list[i]))
    XX = sum(XX_kernel_val) / (n_source * n_source)

    YY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            YY_kernel_val[i] += torch.sum(
                torch.exp(-((total0[:, n_source:] - total1[id:id + batch_size, :]) ** 2) / bandwidth_list[i]))
        id += batch_size
    YY = sum(YY_kernel_val) / (n_target * n_target)

    XY_kernel_val = [0 for _ in range(kernel_num)]
    id = n_source
    while id < n_samples:
        for i in range(kernel_num):
            XY_kernel_val[i] += torch.sum(
                torch.exp(-((total0[:, id:id + batch_size] - total1[:n_source, :]) ** 2) / bandwidth_list[i]))
        id += batch_size
    XY = sum(XY_kernel_val) / (n_source * n_target)

    return XX.item() + YY.item() - 2 * XY.item()

