"""Shared pydantic settings configuration."""
"""Implementation based on the template of ALIGNN."""
import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

from pydantic import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


def plot_learning_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val
