# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import timeit
from functools import partialmethod
from typing import Tuple

import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def bootstrap(x: torch.Tensor, Nboot: int, binsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bootstrapping the mean of tensor.

    Args:
        x (torch.Tensor):
        Nboot (int): _description_
        binsize (int): _description_

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): bootstrapped mean and bootstrapped variance
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()


# From https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
def partialclass(name, cls, *args, **kwds):
    new_cls = type(name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwds)})

    # The following is copied nearly ad verbatim from `namedtuple's` source.

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).

    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass

    return new_cls


class PDECLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("data.time_gap", "model.time_gap")
        parser.link_arguments("data.time_history", "model.time_history")
        parser.link_arguments("data.time_future", "model.time_future")
        parser.link_arguments("data.pde.n_scalar_components", "model.pdeconfig.n_scalar_components")
        parser.link_arguments("data.pde.n_vector_components", "model.pdeconfig.n_vector_components")
        parser.link_arguments("data.pde.trajlen", "model.pdeconfig.trajlen")
        parser.link_arguments("data.pde.n_spatial_dim", "model.pdeconfig.n_spatial_dim")
        # parser.link_arguments("data.usegrid", "model.usegrid")
