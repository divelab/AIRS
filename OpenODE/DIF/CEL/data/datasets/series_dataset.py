"""
Datasets.

Usage:
  datasets.py [--root=root] --data=data 

Options:
  --root=root      Root directory [default: ./data]
  --data=data      Dataset name
"""
import time

import jax
import jax.numpy as jnp
from jax import lax
import diffrax
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint as scipy_odeint
# from scipy.integrate import solve_ivp, solve_bvp
from jax.experimental.ode import odeint as jax_odeint
from torch.utils.data import Dataset
from abc import ABC
import os
from hashlib import sha1
import json
from docopt import docopt
import pysindy as ps
from .utils import DiffTVR
from jsonargparse import class_from_function
from functools import partial
import wandb
from typing import List
import pandas as pd
from einops import rearrange

# __all__ = ["SeriesDataset", "SubsetStar", "Concat", "DampedPendulumDataset", "LotkaVolterraDataset", "SIREpidemicDataset"]

def estimate_derivatives_single_(i, y, differentiation_method, t):
    print(f"{i},")
    y_i = y[i].numpy()
    return differentiation_method._differentiate(y_i, t)


@jax.jit
def delta_func(t, mu=0, intensity=1., sigma=2e-1):
    """
    Delta function implemented as a Gaussian function.

    Parameters:
    t : float
        The point at which the delta function is evaluated.
    mu : float, optional
        The mean of the Gaussian function, default is 0.
    sigma : float, optional
        The standard deviation of the Gaussian function, default is 1e-5.

    Returns:
    float
        The value of the delta function at point t.
    """
    return intensity * (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-0.5 * ((t - mu) / sigma) ** 2)


class SeriesDataset(ABC, Dataset):  #
    """
    Abstract class for Time Series Datasets
    y, t
    """

    def __init__(self, max_for_scaling=None):
        # y shape: (n_samples, time_steps, dimension)
        # t shape: (time_steps)

        self.state_dim = None
        self.state_names = None
        self.y = None
        self.dy = None  # Estimated derivatives
        self.t = None
        self.input_length = None
        self.max_for_scaling = max_for_scaling
        self.phy_params = None  # Fitted parameters

    def plot(self, y, dim=0, **kwargs):
        unscaled_y = self.return_unscaled_y(y)
        for i in range(len(self)):
            plt.plot(
                self.t.numpy(),
                unscaled_y[i, :, dim].numpy(),
                label=f"y(t): dimension {dim}",
            )
        if self.input_length > 0:
            plt.axvline(
                x=self.t.numpy()[self.input_length - 1], linestyle="--", color="black"
            )

        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            plt.ylabel(kwargs["ylabel"])
        if "title" in kwargs:
            plt.title(kwargs["title"])
        plt.show()

    def multi_env_plot(self, y, env_names, feature_names, **kwargs):
        unscaled_y = self.return_unscaled_y(y)  # Assuming this function rescales y
        num_env = len(env_names)  # Infer the number of environments from the env_names list
        num_time_steps = y.shape[1]
        num_features = y.shape[2]

        # Reshape y using einops
        y_reshaped = rearrange(unscaled_y, '(env sample) time feature -> env sample time feature',
                                      env=num_env)
        num_sample_per_env = y_reshaped.shape[1]

        # Create subplots for each feature and environment
        fig, axes = plt.subplots(num_features, num_env, figsize=(num_env * 5, num_features * 5), sharex=True)

        for env in range(num_env):
            for feature in range(num_features):
                ax = axes[feature, env] if num_env > 1 else axes[feature]  # Handle single row/column case
                for sample in range(num_sample_per_env):
                    ax.plot(self.t.numpy(), y_reshaped[env, sample, :, feature].numpy(), label=f"Sample {sample}")

                if self.input_length > 0:
                    ax.axvline(x=self.t.numpy()[self.input_length - 1], linestyle="--", color="black")

                # Set limits, labels, and titles
                if "ylim" in kwargs:
                    ax.set_ylim(kwargs["ylim"])
                if "xlim" in kwargs:
                    ax.set_xlim(kwargs["xlim"])

                ax.set_xlabel("Time" if feature == num_features - 1 else "")
                ax.set_ylabel(f"{feature_names[feature]}" if env == 0 else "")
                ax.set_title(f"{env_names[env]} - {feature_names[feature]}")
                # ax.legend()
        if "title" in kwargs:
            fig.suptitle(kwargs["title"], fontsize=16)

        plt.tight_layout()
        plt.show()

    def scale(self, y, is_scale=False):
        if self.max_for_scaling is None:
            if is_scale:
                self.max_for_scaling = y.amax(dim=[0, 1]) / 10.
            else:
                self.max_for_scaling = torch.ones(self.state_dim)

        y = y / self.max_for_scaling
        return y

    def return_unscaled_y(self, y):
        return y * self.max_for_scaling

    def estimate_derivatives(self, y, method="smooth"):
        if y is None:
            return

        t = self.t.numpy()
        if method == "tvr":
            differentiation_method = DiffTVR(t, 0.2)
        elif method == "smooth":
            differentiation_method = ps.SmoothedFiniteDifference(order=2, smoother_kws={'window_length': 5})
        else:
            differentiation_method = ps.FiniteDifference(order=2)

        # Sequential
        dy = []
        for i in range(y.shape[0]):
            print(i)
            y_i = y[i].numpy()
            dy.append(differentiation_method._differentiate(y_i, t))
        dy = torch.tensor(np.stack(dy))

        return dy

    def estimate_all_derivatives(self, y):
        dy = {}
        dy["smooth"] = self.estimate_derivatives(y, method="smooth")
        # self.dy["tvr"] = self.estimate_derivatives(method="tvr")
        return dy

    def get_initial_value_array(self, y0, n_samples):
        initial_value_array = []
        for i in range(self.state_dim):
            if isinstance(y0[i], tuple):
                array = np.random.uniform(*y0[i], n_samples)
            else:
                array = np.tile(y0[i], n_samples)
            initial_value_array.append(array)

        initial_value_array = np.stack(initial_value_array, axis=1)
        return initial_value_array

    def get_param_arrays(self, params, n_samples):
        param_arrays = []
        for param in params:
            if isinstance(param, tuple):
                param_array = np.random.uniform(*param, n_samples)
            else:
                param_array = np.tile(param, n_samples)
            param_arrays.append(param_array)

        return param_arrays if len(param_arrays) > 1 else param_arrays[0]

    def save(self, other_var: dict = None):
        if other_var is None:
            other_var = dict()
        with open(self.save_filename, "wb") as f:
            all_var = {
                'state_names': self.state_names,
                'state_dim': self.state_dim,
                'input_length': self.input_length,
                't': self.t,
                'y': self.y,
                'dy': self.dy,
                'X': self.X,
                'env': self.env,
                'diff_friction': self.diff_friction,
                'W': self.W,
                'max_for_scaling': self.max_for_scaling,
                'phy_params': self.phy_params,
            } | other_var
            torch.save(all_var, f)

    def load(self):
        print(f"Using saved file: {self.save_filename}")
        all_vars = torch.load(self.save_filename)
        for k, v in all_vars.items():
            setattr(self, k, v)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        item = {'idx': idx,
                'states': self.y[idx],
                'dy': self.dy['smooth'][idx],
                't': self.t}
        if self.X is not None:
            item['X'] = self.X[idx]
        if self.env is not None:
            item['env'] = self.env[idx]
        if self.W is not None:
            item['W'] = self.W[idx]
        if hasattr(self, 'y_inv'):
            item['invariant_states'] = self.y_inv[idx]
        return item
        # return idx, self.y[idx]


class SubsetStar(SeriesDataset):
    """
    Subset of a dataset at specified indices.
    Extended version of what is implemented in Pytorch.

    Arguments:
        dataset (SeriesDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        super().__init__()
        self.y = dataset.y[indices]
        self.t = dataset.t
        self.state_names = dataset.state_names
        self.state_dim = dataset.state_dim
        self.input_length = dataset.input_length
        self.max_for_scaling = dataset.max_for_scaling

        self.phy_params = dataset.phy_params[indices] if dataset.phy_params is not None else None
        if dataset.dy is None:
            self.dy = None
        else:
            self.dy = {}
            for k in dataset.dy.keys():
                self.dy[k] = dataset.dy[k][indices]


class Concat(SeriesDataset):
    """
    Concat two/more Series Datasets
    """

    def __init__(self, dataset1, dataset2):
        super().__init__()
        ## Assumes there are common attributes like t, etc.
        self.y = torch.cat([dataset1.y, dataset2.y])
        self.t = dataset1.t
        self.state_names = dataset1.state_names
        self.state_dim = dataset1.state_dim
        self.input_length = dataset1.input_length
        self.max_for_scaling = torch.maximum(dataset1.max_for_scaling, dataset2.max_for_scaling)

        if dataset1.phy_params is None:
            self.phy_params = None
        else:
            self.phy_params = torch.cat([dataset1.phy_params, dataset2.phy_params])

        if dataset1.dy is None:
            self.dy = None
        else:
            self.dy = {}
            for k in dataset1.dy.keys():
                self.dy[k] = torch.cat([dataset1.dy[k], dataset2.dy[k]])


class DampedPendulumDataset(SeriesDataset):
    """
    Generate damped pendulum data
    $\frac{d^2\theta}{dt^2} + \omega_0^2 \sin(\theta) + \alpha \frac{d\theta}{dt} = 0$
    where $\omega_0 = 2\pi/T_0$, and $T_0$ is the time period.
    """

    def __init__(
            self,
            n_samples,
            t,
            input_length=1,
            y0=[(0, 1), (0, 1)],
            omega0=1.0,
            alpha=0.2,
            is_scale=False,
            max_for_scaling=None,
            seed=0,
            root="./data",
            reload=False,
            num_env=0,
            diff_friction=None,
            split_name='train',
            diverse_env_param=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        # Create files to save the dataset (load if already present)
        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            omega0,
            alpha,
            is_scale,
            seed,
            split_name,
        ]
        if num_env != 0:
            dataset_config.append(num_env)
        if diff_friction is not None:
            dataset_config.append(diff_friction)
        if diverse_env_param:
            dataset_config.append(diverse_env_param)
        # dataset config validation
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )

        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        # Dataset building
        num_env = num_env if num_env != 0 else n_samples
        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$\theta$', r'$\omega$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        # ODE to generate the data from
        EPS = 1e-6
        def pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta))
        def damped_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * dtheta)

        def fix_friction_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * dtheta / (np.abs(dtheta) + EPS))

        def kinetic_friction_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * np.cos(theta) * dtheta / (np.abs(dtheta) + EPS))

        def air_friction_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * dtheta * np.abs(dtheta))

        def thick_air_friction_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - 10 * alpha * dtheta * np.abs(dtheta))

        def energy_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) + 0.5 * alpha * dtheta / (np.abs(dtheta) + EPS))

        def spring_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) - alpha * theta)

        def mul_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) * alpha * theta)

        def freq_pendulum_ode_func(y, t, omega0, alpha):
            theta, dtheta = y
            return (dtheta, -(omega0 ** 2) * np.sin(theta) * np.cos(theta) / (alpha + EPS))

        diff_func_dict = {
            'fix': fix_friction_pendulum_ode_func,
            'kinetic': kinetic_friction_pendulum_ode_func,
            'air': air_friction_pendulum_ode_func,
            'thick_air': thick_air_friction_pendulum_ode_func,
            'energy': energy_pendulum_ode_func,
            'spring': spring_pendulum_ode_func,
            'damped': damped_pendulum_ode_func,
            'mul': mul_pendulum_ode_func,
            'freq': freq_pendulum_ode_func,
            'none': pendulum_ode_func,
        }

        # ---------- Setup initial values and constants -----------------------
        # Sample initial values from the given range
        y0_array = self.get_initial_value_array(y0, n_samples)

        # Sample parameter values from the given range
        # param_arrays = self.get_param_arrays([omega0, alpha], n_samples)
        # omega0_array = param_arrays[0]
        # alpha_array = param_arrays[1]
        omega0_array = self.get_param_arrays([omega0], n_samples)
        alpha_array = self.get_param_arrays([alpha], n_samples if diverse_env_param else num_env) # FIXME: Why one environment only has one alpha? Tried to extract environment invariance.
        env_array = np.arange(num_env).repeat(n_samples // num_env + 1 if n_samples % num_env != 0 else n_samples / num_env)[:n_samples] # note that np.repeat and torch.repeat are different.

        if not diverse_env_param:
            alpha_array = alpha_array[env_array]
            assert (torch.unique(torch.FloatTensor(alpha_array), return_counts=True)[1].max() <= n_samples // num_env + 1).item()
        # ---------------------------------------------------------------------

        # --------------- Generate data ---------------------------
        self.y = [None] * n_samples
        if split_name.startswith('id_test'):
            self.y_inv = [None] * n_samples
        if diff_friction is not None:
            assert len(diff_friction) == num_env, "Number of friction types should be equal to number of environments"
            for i in range(n_samples):
                self.y[i] = scipy_odeint(
                    diff_func_dict[diff_friction[env_array[i]]],
                    y0_array[i],
                    t,
                    args=(omega0_array[i], alpha_array[i]),
                )
                if split_name.startswith('id_test'):
                    self.y_inv[i] = scipy_odeint(
                        diff_func_dict['none'],
                        y0_array[i],
                        t,
                        args=(omega0_array[i], alpha_array[i]),
                    )

        else:
            for i in range(n_samples):
                self.y[i] = scipy_odeint(
                    damped_pendulum_ode_func,
                    y0_array[i],
                    t,
                    args=(omega0_array[i], alpha_array[i]),
                )
        self.env = torch.LongTensor(env_array)
        self.diff_friction = diff_friction
        self.y = torch.FloatTensor(np.stack(self.y))
        self.W = torch.FloatTensor(np.stack([omega0_array, alpha_array], axis=-1))

        # Max scale the data if required
        self.y = self.scale(self.y, is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.X = None
        self.dy = self.estimate_all_derivatives(self.y)
        if split_name.startswith('id_test'):
            self.y_inv = torch.FloatTensor(np.stack(self.y_inv))
            self.y_inv = self.scale(self.y_inv, is_scale)
            self.dy_inv = self.estimate_all_derivatives(self.y_inv)
            self.save({'y_inv': self.y_inv, 'dy_inv': self.dy_inv})
        else:
            self.save()  # Save the data into the file.
        self.plot(self.y, 0, title=f"{split_name} data:")
        if split_name.startswith('id_test'):
            self.plot(self.y_inv, 0, title=f"{split_name} inv data:")

    @classmethod
    def get_standard_dataset(cls, root: str, datatype: str, num_env: int, diff_friction: List[str]=None, n_samples: int=100, input_length_factor: int=3, reload: bool=False, diverse_env_param: bool=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        for seed in range(k):
            np.random.seed(seed)  # Set seed
            if datatype == '1':
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(0, np.pi / 2), (0.0, 0.0)]
                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_id = omega0_ood = 1.0
                alpha_id = alpha_ood = 0.2

            elif datatype == '2':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(0, np.pi / 2), (0.0, -1.0)]
                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_id = omega0_ood = (1.0, 2.0)
                alpha_id = alpha_ood = (0.2, 0.4)

            elif datatype == '3':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(0, np.pi / 2), (0.0, -1.0)]
                omega0_id = (1.0, 2.0)
                alpha_id = (0.2, 0.4)

                y0_ood = [(np.pi - 0.1, np.pi - 0.05), (0.0, -1.0)]
                omega0_ood = (2.0, 3.0)
                alpha_ood = (0.4, 0.6)
            elif datatype == 'inv_check':
                # Arithmetic alignment: ill-post question → Break positive assumption → Break invariance (red) if out-of-support at y.
                y0_id = y0_ood = [(0, np.pi / 2), (0.0, -1.0)]
                omega0_id = omega0_ood = (1.0, 2.0)
                alpha_id = (0.2, 0.4)

                alpha_ood = (0.0, 0.0)
            elif datatype == 'inv_ood':
                y0_id = y0_ood = [(0, np.pi / 2), (0.0, -1.0)]
                omega0_id = (1.0, 2.0)
                alpha_id = (0.2, 0.4)

                omega0_ood = (2.0, 3.0)
                alpha_ood = (0.0, 0.0)
            elif datatype == 'env_adapt':
                y0_id = y0_ood = [(0, np.pi / 2), (0.0, -1.0)]
                omega0_id = omega0_ood = (1.0, 2.0)
                alpha_id = (0.2, 0.4)

                alpha_ood = (0.4, 0.6)


            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                omega0=omega0_id,
                alpha=alpha_id,
                seed=seed,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'train {datatype}',
                diverse_env_param=diverse_env_param,
            )
            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                omega0=omega0_id,
                alpha=alpha_id,
                seed=seed,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'id_test {datatype}',
                diverse_env_param=diverse_env_param,
            )
            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                omega0=omega0_ood,
                alpha=alpha_ood,
                seed=seed,
                root=root,
                reload=reload,
                num_env=len(diff_friction[-1:]),
                diff_friction=diff_friction[-1:] if diff_friction is not None else None,
                split_name=f'ood_test {datatype}',
                diverse_env_param=diverse_env_param,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold

class_from_function(DampedPendulumDataset.get_standard_dataset, Dataset, name='getDampedPendulumDataset')

class LotkaVolterraDataset(SeriesDataset):
    """
    Generate LV data
    Prey:
          dx/dt = \alpha * x  - \beta * x * y
    Predator
          dy/dt = \delta * x * y - \gamma * y
    """

    def __init__(
            self,
            n_samples,
            t,
            input_length=1,
            y0=[(1000, 2000), (10, 20)],
            alpha=0.1 * 12,
            beta=0.005 * 12,
            gamma=0.04 * 12,
            delta=0.00004 * 12,
            is_scale=False,
            max_for_scaling=None,
            seed=0,
            root=".",
            reload=False,
            num_env=0,
            diff_friction=None,
            split_name='train',
            diverse_env_param=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            alpha,
            beta,
            gamma,
            delta,
            is_scale,
            seed,
            split_name,
        ]
        if num_env != 0:
            dataset_config.append(num_env)
        if diff_friction is not None:
            dataset_config.append(diff_friction)
        if diverse_env_param:
            dataset_config.append(diverse_env_param)
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        # Dataset building
        num_env = num_env if num_env != 0 else n_samples
        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$x$', r'$y$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        # --- ODE to generate the data from ---
        EPS = 1e-6

        def lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )
        def kill_sin_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1 + np.sin(prey / 100)),
                delta * prey * predator * (1 + np.sin(prey / 100)) - gamma * predator,
            )

        def competition_sin_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator * (1 + np.sin(predator / 100)),
            )
        def limited_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey * (1.0 - prey / 2000) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        def value_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1. + np.exp(-predator / 10)),
                delta * prey * predator - gamma * predator,
            )

        def value2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1. + 10 * np.exp(-predator / 10)),
                delta * prey * predator - gamma * predator,
            )

        def fight_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator * (1. + np.exp(-predator / 10)) - gamma * predator,
            )

        def fight2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator * (1. + 10 * np.exp(-predator / 10)) - gamma * predator,
            )

        def omnivore_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator + 10 * gamma * (1. - predator / 30),
            )

        def omnivore2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator + 20 * gamma * (1. - predator / 100),
            )
        def season_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey * (1 + np.sin(prey / 100)) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        def predator_season_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey * (1 + np.sin(predator / 100)) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )


        diff_func_dict = {
            'season': season_lotka_volterra_ode_func,
            'pseason': predator_season_lotka_volterra_ode_func,
            'kill': kill_sin_lotka_volterra_ode_func,
            'sin': competition_sin_lotka_volterra_ode_func,
            'comp': competition_sin_lotka_volterra_ode_func,
            'limited': limited_lotka_volterra_ode_func,
            'omnivore': omnivore_lotka_volterra_ode_func,
            'omnivore2': omnivore2_lotka_volterra_ode_func,
            'value': value_lotka_volterra_ode_func,
            'value2': value2_lotka_volterra_ode_func,
            'fight': fight_lotka_volterra_ode_func,
            'fight2': fight2_lotka_volterra_ode_func,
            'none': lotka_volterra_ode_func,
            'orig': lotka_volterra_ode_func,
        }

        # ---------- Setup initial values and constants -----------------------

        y0_array = self.get_initial_value_array(y0, n_samples)

        beta_array, delta_array = self.get_param_arrays([beta, delta], n_samples)
        alpha_array, gamma_array = self.get_param_arrays([alpha, gamma], n_samples if diverse_env_param else num_env)
        env_array = np.arange(num_env).repeat(n_samples // num_env + 1 if n_samples % num_env != 0 else n_samples / num_env)[:n_samples]

        if not diverse_env_param:
            alpha_array = alpha_array[env_array]
            gamma_array = gamma_array[env_array]
            assert (torch.unique(torch.FloatTensor(alpha_array), return_counts=True)[1].max() <= n_samples // num_env + 1).item()
        # ---------------------------------------------------------------------

        # --------------- Generate data ---------------------------
        self.y = [None] * n_samples
        if split_name.startswith('id_test'):
            self.y_inv = [None] * n_samples
        if diff_friction is not None:
            assert len(diff_friction) == num_env, "Number of friction types should be equal to number of environments"
            for i in range(n_samples):
                args = (
                    alpha_array[i],
                    beta_array[i],
                    gamma_array[i],
                    delta_array[i],
                )
                self.y[i] = scipy_odeint(
                    diff_func_dict[diff_friction[env_array[i]]],
                    y0_array[i],
                    t,
                    args=args,
                )
                if split_name.startswith('id_test'):
                    self.y_inv[i] = scipy_odeint(
                        diff_func_dict['none'],
                        y0_array[i],
                        t,
                        args=args,
                    )
        else:
            for i in range(n_samples):
                args = (
                    alpha_array[i],
                    beta_array[i],
                    gamma_array[i],
                    delta_array[i],
                )
                self.y[i] = scipy_odeint(lotka_volterra_ode_func, y0_array[i], t, args=args)

        self.env = torch.LongTensor(env_array)
        self.diff_friction = diff_friction
        self.y = torch.FloatTensor(np.stack(self.y))
        self.W = torch.FloatTensor(np.stack([alpha_array, beta_array, gamma_array, delta_array], axis=-1))

        # Max scale the data if required
        self.y = self.scale(self.y, is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.X = None
        self.dy = self.estimate_all_derivatives(self.y)
        if split_name.startswith('id_test'):
            self.y_inv = torch.FloatTensor(np.stack(self.y_inv))
            self.y_inv = self.scale(self.y_inv, is_scale)
            self.dy_inv = self.estimate_all_derivatives(self.y_inv)
            self.save({'y_inv': self.y_inv, 'dy_inv': self.dy_inv})
        else:
            self.save()  # Save the data into the file.
        # self.plot(self.y, 0, title=f"{split_name} data:")
        self.multi_env_plot(self.y, diff_friction, ['prey', 'predator'], title=f"{split_name} data:")
        # if split_name.startswith('id_test'):
        #     self.plot(self.y_inv, 0, title=f"{split_name} inv data:")

    @classmethod
    def get_standard_dataset(cls, root: str, datatype: str, num_env: int, diff_friction: List[str]=None, n_samples: int=100, input_length_factor: int=3, reload: bool=False, diverse_env_param: bool=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)  # Set seed
            if datatype == '1':
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = 0.1 * 12
                beta_id = beta_ood = 0.005 * 12
                gamma_id = gamma_ood = 0.04 * 12
                delta_id = delta_ood = 0.00004 * 12

            elif datatype == '2':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

            elif datatype == '3':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]

                alpha_id = (0.1 * 12, 0.2 * 12)
                beta_id = (0.005 * 12, 0.01 * 12)
                gamma_id = (0.04 * 12, 0.08 * 12)
                delta_id = (0.00004 * 12, 0.00008 * 12)
                alpha_ood = (0.2 * 12, 0.3 * 12)
                beta_ood = (0.01 * 12, 0.015 * 12)
                gamma_ood = (0.08 * 12, 0.12 * 12)
                delta_ood = (0.00008 * 12, 0.00012 * 12)
            elif datatype == 'inv_check':
                y0_id = y0_ood = [(1000, 2000), (10, 20)]

                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
                num_env=int(num_env * 0.8),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'train {datatype}',
                diverse_env_param=diverse_env_param,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
                num_env=int(num_env * 0.8),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'id_test {datatype}',
                diverse_env_param=diverse_env_param,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                alpha=alpha_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                delta=delta_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
                num_env=int(num_env * 0.2),
                diff_friction=diff_friction[-1:] if diff_friction is not None else None,
                split_name=f'ood_test {datatype}',
                diverse_env_param=diverse_env_param,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold


class_from_function(LotkaVolterraDataset.get_standard_dataset, Dataset, name='getLotkaVolterraDataset')

class LotkaVolterra2Dataset(SeriesDataset):
    """
    Generate LV data
    Prey:
          dx/dt = \alpha * x  - \beta * x * y
    Predator
          dy/dt = \delta * x * y - \gamma * y
    """

    def __init__(
            self,
            n_samples,
            t,
            input_length=1,
            y0=[(1000, 2000), (10, 20)],
            alpha=0.1 * 12,
            beta=0.005 * 12,
            gamma=0.04 * 12,
            delta=0.00004 * 12,
            alpha2=0.1 * 12,
            beta2=0.005 * 12,
            gamma2=0.04 * 12,
            delta2=0.00004 * 12,
            is_scale=False,
            max_for_scaling=None,
            seed=0,
            root=".",
            reload=False,
            num_env=0,
            diff_friction=None,
            split_name='train',
            diverse_env_param=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            alpha,
            beta,
            gamma,
            delta,
            alpha2,
            beta2,
            gamma2,
            delta2,
            is_scale,
            seed,
            split_name,
        ]
        if num_env != 0:
            dataset_config.append(num_env)
        if diff_friction is not None:
            dataset_config.append(diff_friction)
        if diverse_env_param:
            dataset_config.append(diverse_env_param)
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        # Dataset building
        num_env = num_env if num_env != 0 else n_samples
        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$x$', r'$y$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        # --- ODE to generate the data from ---
        EPS = 1e-6

        def lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )
        def kill_sin_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1 + np.sin(prey / 100)),
                delta * prey * predator * (1 + np.sin(prey / 100)) - gamma * predator,
            )

        def competition_sin_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator * (1 + np.sin(predator / 100)),
            )
        def limited_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey * (1.0) + alpha2 * prey * (- prey / 2000) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        def value_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1. + np.exp(-predator / 10)),
                delta * prey * predator - gamma * predator,
            )

        def value2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator * (1.) - beta2 * prey * predator * (10 * np.exp(-predator / 10)),
                delta * prey * predator - gamma * predator,
            )

        def fight_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator * (1. + np.exp(-predator / 10)) - gamma * predator,
            )

        def fight2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator * (1.) + delta2 * prey * predator * (10 * np.exp(-predator / 10)) - gamma * predator,
            )

        def omnivore_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator + 10 * gamma * (1. - predator / 30),
            )

        def omnivore2_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta, alpha2, beta2, gamma2, delta2):
            prey, predator = y
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator + 20 * gamma2 * (1. - predator / 100),
            )
        def season_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey * (1 + np.sin(prey / 100)) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        def predator_season_lotka_volterra_ode_func(y, t, alpha, beta, gamma, delta):
            prey, predator = y
            return (
                alpha * prey * (1 + np.sin(predator / 100)) - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )


        diff_func_dict = {
            'season': season_lotka_volterra_ode_func,
            'pseason': predator_season_lotka_volterra_ode_func,
            'kill': kill_sin_lotka_volterra_ode_func,
            'sin': competition_sin_lotka_volterra_ode_func,
            'comp': competition_sin_lotka_volterra_ode_func,
            'limited': limited_lotka_volterra_ode_func,
            'omnivore': omnivore_lotka_volterra_ode_func,
            'omnivore2': omnivore2_lotka_volterra_ode_func,
            'value': value_lotka_volterra_ode_func,
            'value2': value2_lotka_volterra_ode_func,
            'fight': fight_lotka_volterra_ode_func,
            'fight2': fight2_lotka_volterra_ode_func,
            'none': lotka_volterra_ode_func,
            'orig': lotka_volterra_ode_func,
        }

        # ---------- Setup initial values and constants -----------------------

        y0_array = self.get_initial_value_array(y0, n_samples)

        beta_array, delta_array, beta2_array, delta2_array = self.get_param_arrays([beta, delta, beta2, delta2], n_samples)
        alpha_array, gamma_array, alpha2_array, gamma2_array = self.get_param_arrays([alpha, gamma, alpha2, gamma2], n_samples if diverse_env_param else num_env)
        env_array = np.arange(num_env).repeat(n_samples // num_env + 1 if n_samples % num_env != 0 else n_samples / num_env)[:n_samples]

        if not diverse_env_param:
            alpha_array = alpha_array[env_array]
            gamma_array = gamma_array[env_array]
            assert (torch.unique(torch.FloatTensor(alpha_array), return_counts=True)[1].max() <= n_samples // num_env + 1).item()
        # ---------------------------------------------------------------------

        # --------------- Generate data ---------------------------
        self.y = [None] * n_samples
        if split_name.startswith('id_test'):
            self.y_inv = [None] * n_samples
        if diff_friction is not None:
            assert len(diff_friction) == num_env, "Number of friction types should be equal to number of environments"
            for i in range(n_samples):
                args = (
                    alpha_array[i],
                    beta_array[i],
                    gamma_array[i],
                    delta_array[i],
                    alpha2_array[i],
                    beta2_array[i],
                    gamma2_array[i],
                    delta2_array[i],
                )
                self.y[i] = scipy_odeint(
                    diff_func_dict[diff_friction[env_array[i]]],
                    y0_array[i],
                    t,
                    args=args,
                )
                if split_name.startswith('id_test'):
                    self.y_inv[i] = scipy_odeint(
                        diff_func_dict['none'],
                        y0_array[i],
                        t,
                        args=args,
                    )
        else:
            for i in range(n_samples):
                args = (
                    alpha_array[i],
                    beta_array[i],
                    gamma_array[i],
                    delta_array[i],
                    alpha2_array[i],
                    beta2_array[i],
                    gamma2_array[i],
                    delta2_array[i],
                )
                self.y[i] = scipy_odeint(lotka_volterra_ode_func, y0_array[i], t, args=args)

        self.env = torch.LongTensor(env_array)
        self.diff_friction = diff_friction
        self.y = torch.FloatTensor(np.stack(self.y))
        self.W = torch.FloatTensor(np.stack([alpha_array, beta_array, gamma_array, delta_array, alpha2_array, beta2_array, gamma2_array, delta2_array], axis=-1))

        # Max scale the data if required
        self.y = self.scale(self.y, is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.X = None
        self.dy = self.estimate_all_derivatives(self.y)
        if split_name.startswith('id_test'):
            self.y_inv = torch.FloatTensor(np.stack(self.y_inv))
            self.y_inv = self.scale(self.y_inv, is_scale)
            self.dy_inv = self.estimate_all_derivatives(self.y_inv)
            self.save({'y_inv': self.y_inv, 'dy_inv': self.dy_inv})
        else:
            self.save()  # Save the data into the file.
        # self.plot(self.y, 0, title=f"{split_name} data:")

        # -- Nice multi-env plot --
        # self.multi_env_plot(self.y, diff_friction, ['prey', 'predator'], title=f"{split_name} data:")
        # if split_name.startswith('id_test'):
        #     self.plot(self.y_inv, 0, title=f"{split_name} inv data:")

    @classmethod
    def get_standard_dataset(cls, root: str, datatype: str, num_env: int, diff_friction: List[str]=None, n_samples: int=100, input_length_factor: int=3, reload: bool=False, diverse_env_param: bool=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)  # Set seed
            if datatype == '1':
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = 0.1 * 12
                beta_id = beta_ood = 0.005 * 12
                gamma_id = gamma_ood = 0.04 * 12
                delta_id = delta_ood = 0.00004 * 12

            elif datatype == '2':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

            elif datatype == '3':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]

                alpha_id = (0.1 * 12, 0.2 * 12)
                beta_id = (0.005 * 12, 0.01 * 12)
                gamma_id = (0.04 * 12, 0.08 * 12)
                delta_id = (0.00004 * 12, 0.00008 * 12)
                alpha_ood = (0.2 * 12, 0.3 * 12)
                beta_ood = (0.01 * 12, 0.015 * 12)
                gamma_ood = (0.08 * 12, 0.12 * 12)
                delta_ood = (0.00008 * 12, 0.00012 * 12)
            elif datatype == 'inv_check':
                y0_id = y0_ood = [(1000, 2000), (10, 20)]

                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                alpha2=alpha_id,
                beta2=beta_id,
                gamma2=gamma_id,
                delta2=delta_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'train {datatype}',
                diverse_env_param=diverse_env_param,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                alpha2=alpha_id,
                beta2=beta_id,
                gamma2=gamma_id,
                delta2=delta_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'id_test {datatype}',
                diverse_env_param=diverse_env_param,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                alpha=alpha_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                delta=delta_ood,
                alpha2=alpha_ood,
                beta2=beta_ood,
                gamma2=gamma_ood,
                delta2=delta_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
                num_env=len(diff_friction[-1:]),
                diff_friction=diff_friction[-1:] if diff_friction is not None else None,
                split_name=f'ood_test {datatype}',
                diverse_env_param=diverse_env_param,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold


class_from_function(LotkaVolterra2Dataset.get_standard_dataset, Dataset, name='getLotkaVolterra2Dataset')

class SIREpidemicDataset(SeriesDataset):
    """
    Generate SIR epidemic data
        ds/dt = -\beta is/(s + i + r)
        di/dt = \beta is/(s + i + r) - \gamma i
        dr/dt = \gamma i
    """

    def __init__(
            self,
            n_samples,
            t,
            input_length=1,
            y0=[(90, 100), (0, 5), (0, 0)],
            beta=4,
            gamma=0.4,
            is_scale=False,
            max_for_scaling=None,
            seed=0,
            root=".",
            reload=False,
            num_env=0,
            diff_friction=None,
            split_name='train',
            diverse_env_param=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            beta,
            gamma,
            is_scale,
            seed,
            split_name,
        ]
        if num_env != 0:
            dataset_config.append(num_env)
        if diff_friction is not None:
            dataset_config.append(diff_friction)
        if diverse_env_param:
            dataset_config.append(diverse_env_param)
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        # Dataset building
        num_env = num_env if num_env != 0 else n_samples
        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 3
        self.state_names = [r'$S$', r'$I$', r'$R$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        # --- ODE to generate the data from ---
        EPS = 1e-6

        def sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r) - gamma * i,
                gamma * i
            )
        def fake_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r) + gamma * np.log(i),
                - gamma * np.log(i)
            )
        def speed_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r) - gamma * i ** 2,
                gamma * i ** 2
            )
        def new_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r) + gamma * i,
                beta * i * s / (s + i + r) - gamma * i,
                gamma * i
            )
        def dead_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r) + gamma * r,
                beta * i * s / (s + i + r) - 2 * gamma * i,
                gamma * i - gamma * r
            )
        def hope_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r) - gamma * (1. / (1. + np.exp(-r))) * i,
                gamma * (1. / (1. + np.exp(-r))) * i
            )
        def infect_sir_ode_func(y, t, beta, gamma):
            s, i, r = y
            return (
                -beta * i * s / (s + i + r),
                beta * i * s / (s + i + r),
                0
            )

        diff_func_dict = {
            'orig': sir_ode_func,
            'fake': fake_sir_ode_func,
            'speed': speed_sir_ode_func,
            'hope': hope_sir_ode_func,
            'new': new_sir_ode_func,
            'dead': dead_sir_ode_func,
            'none': infect_sir_ode_func,
        }

        y0_array = self.get_initial_value_array(y0, n_samples)

        param_arrays = self.get_param_arrays([beta, gamma], n_samples)
        beta_array = param_arrays[0]
        gamma_array = param_arrays[1]
        env_array = np.arange(num_env).repeat(
            n_samples // num_env + 1 if n_samples % num_env != 0 else n_samples / num_env)[:n_samples]

        self.y = [None] * n_samples
        if split_name.startswith('id_test'):
            self.y_inv = [None] * n_samples
        if diff_friction is not None:
            assert len(diff_friction) == num_env, "Number of friction types should be equal to number of environments"
            for i in range(n_samples):
                args = (
                    beta_array[i],
                    gamma_array[i],
                )
                print(diff_friction[env_array[i]])
                self.y[i] = scipy_odeint(
                    diff_func_dict[diff_friction[env_array[i]]],
                    y0_array[i],
                    t,
                    args=args,
                )
                if split_name.startswith('id_test'):
                    self.y_inv[i] = scipy_odeint(
                        diff_func_dict['none'],
                        y0_array[i],
                        t,
                        args=args,
                    )
        else:
            for i in range(n_samples):
                args = (
                    beta_array[i],
                    gamma_array[i],
                )
                self.y[i] = scipy_odeint(sir_ode_func, y0_array[i], t, args=args)

        self.env = torch.LongTensor(env_array)
        self.diff_friction = diff_friction
        self.y = torch.FloatTensor(np.stack(self.y))
        self.W = torch.FloatTensor(np.stack([beta_array, gamma_array], axis=-1))

        # Max scale the data if required
        self.y = self.scale(self.y, is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.X = None
        self.dy = self.estimate_all_derivatives(self.y)
        if split_name.startswith('id_test'):
            self.y_inv = torch.FloatTensor(np.stack(self.y_inv))
            self.y_inv = self.scale(self.y_inv, is_scale)
            self.dy_inv = self.estimate_all_derivatives(self.y_inv)
            self.save({'y_inv': self.y_inv, 'dy_inv': self.dy_inv})
        else:
            self.save()  # Save the data into the file.

        # -- Nice multi-env plot --
        # self.multi_env_plot(self.y, diff_friction, ['S', 'I', 'R'], title=f"{split_name} data:")
        # self.plot(self.y, 1, title=f"{split_name} data:")
        # if split_name.startswith('id_test'):
        #     self.plot(self.y_inv, 1, title=f"{split_name} inv data:")

    @classmethod
    def get_standard_dataset(cls, root: str, datatype: str, num_env: int, diff_friction: List[str]=None, n_samples: int=100, input_length_factor: int=5, reload: bool=False, diverse_env_param: bool=False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)  # Set seed
            if datatype == '1':
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]
                beta_id = beta_ood = 4
                gamma_id = gamma_ood = 0.4

            elif datatype == '2':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]
                beta_id = beta_ood = (4, 8)
                gamma_id = gamma_ood = (0.4, 0.8)

            elif datatype == '3':
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(9, 10), (1, 5), (0, 0)]
                y0_ood = [(90, 100), (1, 5), (0, 0)]

                beta_id = (4, 8)
                gamma_id = (0.4, 0.8)
                beta_ood = (8, 12)
                gamma_ood = (0.8, 1.2)
            elif datatype == 'inv_check':
                y0_id = y0_ood = [(9, 10), (1, 5), (0, 0)]
                beta_id = beta_ood = (4, 8)
                gamma_id = gamma_ood = (0.4, 0.8)

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                beta=beta_id,
                gamma=gamma_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'train {datatype}',
                diverse_env_param=diverse_env_param,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                beta=beta_id,
                gamma=gamma_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
                num_env=len(diff_friction[:-1]),
                diff_friction=diff_friction[:-1] if diff_friction is not None else None,
                split_name=f'id_test {datatype}',
                diverse_env_param=diverse_env_param,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
                num_env=len(diff_friction[-1:]),
                diff_friction=diff_friction[-1:] if diff_friction is not None else None,
                split_name=f'ood_test {datatype}',
                diverse_env_param=diverse_env_param,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold

class_from_function(SIREpidemicDataset.get_standard_dataset, Dataset, name='getSIREpidemicDataset')


def generate_all_datasets(cls, root, reload=False):
    for datatype in range(1, 4):
        print(f"Data type: {datatype}")
        data_kfold = cls.get_standard_dataset(root=root, datatype=datatype, n_samples=1000, reload=reload)
        train_data, id_test_data, ood_test_data = data_kfold[0]
        print("=" * 20 + "Train data" + "=" * 20)
        for d in range(train_data.state_dim):
            train_data.plot(dim=d, title=f"Dim: {d}")
        print("=" * 20 + "ID Test data" + "=" * 20)
        for d in range(train_data.state_dim):
            id_test_data.plot(dim=d, title=f"Dim: {d}")
        print("=" * 20 + "OOD Test data" + "=" * 20)
        for d in range(train_data.state_dim):
            ood_test_data.plot(dim=d, title=f"Dim: {d}")


class LotkaVolterraCDEDataset(SeriesDataset):
    """
    Generate LV data
    Prey:
          dx/dt = \alpha * x  - \beta * x * y
    Predator
          dy/dt = \delta * x * y - \gamma * y
    """

    def __init__(
            self,
            n_samples,
            t,
            input_length=1,
            y0=[(1000, 2000), (10, 20)],
            alpha=0.1 * 12,
            beta=0.005 * 12,
            gamma=0.04 * 12,
            delta=0.00004 * 12,
            is_scale=False,
            max_for_scaling=None,
            seed=0,
            root=".",
            reload=False,
            intervention_type=None,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        os.makedirs(root, exist_ok=True)
        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0,
            alpha,
            beta,
            gamma,
            delta,
            is_scale,
            seed,
            intervention_type,
        ]
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        from jax import config
        if not reload and os.path.exists(self.save_filename):
            if config.read('jax_enable_x64'):
                raise ValueError("jax_enable_x64 is enabled for general trainings. Please disable it by adding environment variable JAX_ENABLE_X64=False")
            self.load()
            return

        if not config.read('jax_enable_x64'):
            raise ValueError("jax_enable_x64 is not enabled for data generation. Please enable it by adding environment variable JAX_ENABLE_X64=True")

        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$x$', r'$y$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        y0_array = self.get_initial_value_array(y0, n_samples)
        alpha_array, beta_array, gamma_array, delta_array = self.get_param_arrays([alpha, beta, gamma, delta],
                                                                                  n_samples)

        if intervention_type:
            intervention_time_idx, intervention_time, intervention_intensity = self.get_interventions(t, n_samples,
                                                                                                      intervention_type)
            self.X = torch.zeros((n_samples, len(t)))
            self.X[
                torch.arange(n_samples)[:, None].repeat(1, 3).reshape(-1),
                intervention_time_idx.reshape(-1)
            ] = torch.tensor(intervention_intensity.reshape(-1), dtype=torch.float32)

            solve_iv_ode = jax.jit(self.solve_ode, static_argnames=('intervention_type',))
            y, dydt = jax.vmap(solve_iv_ode, in_axes=(None, 0, 0, 0, 0, 0, None, 0, 0))(t, y0_array,
                                                                                    alpha_array, beta_array,
                                                                                    gamma_array, delta_array,
                                                                                    intervention_type,
                                                                                    intervention_time,
                                                                                    intervention_intensity)
        else:
            self.X = None
            solve_orig_ode = jax.jit(self.solve_ode, static_argnames=('intervention_type', 'iv_time', 'iv_intensity'))
            y, dydt = jax.vmap(solve_orig_ode, in_axes=(None, 0, 0, 0, 0, 0, None, None, None))(t, y0_array, alpha_array, beta_array,
                                                                               gamma_array, delta_array,
                                                                               None, None, None)
        # for i in range(3):
        #     df = pd.DataFrame(data=np.array(jnp.concatenate([t[:, None], orig_y[i], y[i]], axis=1)), columns=['t', 'orig_prey', 'orig_predator', 'iv_prey', 'iv_predator'])
        #     wandb.log({f'prey_{i}': wandb.plot.line_series(xs=t, ys=df[['orig_prey', 'iv_prey']].to_numpy().T,
        #                                               keys=['original', 'intervention'], xname='time', title='prey'),
        #                f'predator_{i}': wandb.plot.line_series(xs=t, ys=df[['orig_predator', 'iv_predator']].to_numpy().T,
        #                                               keys=['original', 'intervention'], xname='time', title='predator')})
        # exit()
        self.y = torch.FloatTensor(np.array(y))
        self.scale(is_scale)

        # Estimate the derivatives from the data (used for some models)
        self.dy = dict()
        # TODO: regenerate dataset since dy is now changed to be scaled (before, it was unscaled)
        self.dy["smooth"] = torch.FloatTensor(np.array(dydt) / self.max_for_scaling.numpy())
        # self.estimate_all_derivatives()
        self.save()

    def get_interventions(self, t, n_samples, intervention_type):
        if intervention_type == 'predator_shot':
            action_time_idx = np.random.randint(0, len(self.t), size=(n_samples, 3))
            action_time = t[action_time_idx]
            action_intensity = np.random.uniform(0, 0.3, size=(n_samples, 3))
            return action_time_idx, action_time, action_intensity

    def plan_predator_shot(self, t, iv_time, iv_intensity):
        # iv_time, iv_intensity: (1, )
        instant_effect = delta_func(t, iv_time, iv_intensity)
        relative_t = t - iv_time
        continuous_effect = lax.cond(relative_t > 0,
                                     lambda x: jnp.exp(-x),
                                     lambda x: 0.,
                                     relative_t)
        return instant_effect, continuous_effect

    @partial(jax.jit, static_argnums=(0, 1))
    def iv_effect_at(self, intervention_type, t, iv_time, iv_intensity):
        if intervention_type == 'predator_shot':
            # iv_time, iv_intensity: (n_interventions, )
            instant_effect, continuous_effect = jax.vmap(self.plan_predator_shot, (None, 0, 0))(t, iv_time,
                                                                                                iv_intensity)
            instant_effect = jnp.sum(instant_effect, axis=0)
            continuous_effect = jnp.sum(continuous_effect, axis=0)
            return instant_effect, continuous_effect
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 1))
    def lotka_volterra_ode_func(self, intervention_type, t, y, args):
        if intervention_type == "predator_shot":
            alpha, beta, gamma, delta, iv_time, iv_intensity = args
            prey, predator = y
            instant_effect, continuous_effect = self.iv_effect_at(intervention_type, t, iv_time, iv_intensity)
            # np.random.
            return jnp.array([
                alpha * prey - beta * prey * predator,
                delta / (continuous_effect + 1) * prey * predator - gamma * predator - instant_effect * predator,
            ])
        else:
            alpha, beta, gamma, delta = args
            prey, predator = y
            return jnp.array([
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ])

    # @partial(jax.jit, static_argnums=(0,)) if we don't use decorator, we don't need to mark 'self' as static_argnums
    def solve_ode(self, t, y0, alpha, beta, gamma, delta, intervention_type, iv_time, iv_intensity):
        vector_field = partial(self.lotka_volterra_ode_func, intervention_type)
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri8()
        args = (alpha, beta, gamma, delta, iv_time, iv_intensity) if intervention_type is not None else (
            alpha, beta, gamma, delta)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t[0],
            t[-1],
            None,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),  # , pcoeff=0.4, icoeff=0.3, dcoeff=0),
            # max_steps=None,
            saveat=diffrax.SaveAt(ts=t)
        )

        return sol.ys, jax.vmap(vector_field, in_axes=(0, 0, None))(t, sol.ys, args)

    @classmethod
    def get_standard_dataset(cls, root: str, datatype: int, n_samples: int = 100,
                             input_length_factor: int = 3, reload: bool = False):
        k = 5
        data_kfold = []
        T = 10
        nT = 10 * T
        t = np.linspace(0, T, nT)
        input_length = int(nT // input_length_factor)
        is_scale = True
        for seed in range(k):
            np.random.seed(seed)  # Set seed
            intervention_type = None
            if datatype == 1:
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = 0.1 * 12
                beta_id = beta_ood = 0.005 * 12
                gamma_id = gamma_ood = 0.04 * 12
                delta_id = delta_ood = 0.00004 * 12

            elif datatype == 2:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

            elif datatype == 3:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions & dynamical system parameters
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]

                alpha_id = (0.1 * 12, 0.2 * 12)
                beta_id = (0.005 * 12, 0.01 * 12)
                gamma_id = (0.04 * 12, 0.08 * 12)
                delta_id = (0.00004 * 12, 0.00008 * 12)
                alpha_ood = (0.2 * 12, 0.3 * 12)
                beta_ood = (0.01 * 12, 0.015 * 12)
                gamma_ood = (0.08 * 12, 0.12 * 12)
                delta_ood = (0.00008 * 12, 0.00012 * 12)

            elif datatype == 4:
                # 1. Single dynamical system parameter across all training curves
                # 2. OOD on initial conditions
                # 3. Intervention plans along the trajectory
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = 0.1 * 12
                beta_id = beta_ood = 0.005 * 12
                gamma_id = gamma_ood = 0.04 * 12
                delta_id = delta_ood = 0.00004 * 12

                intervention_type = 'predator_shot'

            elif datatype == 5:
                # 1. Multiple dynamical system parameter across training curves
                # 2. OOD on initial conditions
                # 3. Intervention plans along the trajectory
                y0_id = [(1000, 2000), (10, 20)]
                y0_ood = [(100, 200), (10, 20)]
                alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
                beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
                gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
                delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

                intervention_type = 'predator_shot'

            train_data = cls(
                int(n_samples * 0.8),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                root=root,
                reload=reload,
                intervention_type=intervention_type,
            )

            id_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_id,
                alpha=alpha_id,
                beta=beta_id,
                gamma=gamma_id,
                delta=delta_id,
                seed=seed,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                root=root,
                reload=reload,
                intervention_type=intervention_type,
            )

            ood_test_data = cls(
                int(n_samples * 0.2),
                t,
                input_length=input_length,
                y0=y0_ood,
                alpha=alpha_ood,
                beta=beta_ood,
                gamma=gamma_ood,
                delta=delta_ood,
                seed=seed,
                root=root,
                is_scale=is_scale,
                max_for_scaling=train_data.max_for_scaling,
                reload=reload,
                intervention_type=intervention_type,
            )
            data_kfold.append((train_data, id_test_data, ood_test_data))
        return data_kfold
        # return train_data


if __name__ == "__main__":
    args = docopt(__doc__)
    root = args["--root"]
    data = args["--data"]

    if data == "damped_pendulum":
        generate_all_datasets(DampedPendulumDataset, root, reload=True)
        # data_kfold = DampedPendulumDataset.get_standard_dataset(root=root, datatype=1, n_samples=1000, reload=True)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # train_data.plot()
        # ood_test_data.plot()
    elif data == "lotka_volterra":
        generate_all_datasets(LotkaVolterraDataset, root, reload=True)
        # data_kfold = LotkaVolterraDataset.get_standard_dataset(root=root, datatype=3, n_samples=1000, reload=False)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # ood_test_data.plot()
    elif data == "sir":
        generate_all_datasets(SIREpidemicDataset, root, reload=True)
        # data_kfold = SIREpidemicDataset.get_standard_dataset(root=root, datatype=3, n_samples=1000, reload=False)
        # train_data, id_test_data, ood_test_data = data_kfold[0]
        # ood_test_data.plot()
    else:
        raise NotImplementedError
