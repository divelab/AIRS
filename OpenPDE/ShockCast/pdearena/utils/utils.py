# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import timeit
from functools import partialmethod

from pytorch_lightning.utilities.rank_zero import rank_zero_only

import sys


class Timer:
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(level=logging.INFO)
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

# https://stackoverflow.com/a/38637774/10965084
GETTRACE = getattr(sys, 'gettrace', None)
IS_DEBUGGING = False
if GETTRACE is not None and GETTRACE():
    IS_DEBUGGING = True
get_logger().info(f"IS_DEBUGGING: {IS_DEBUGGING}")

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
