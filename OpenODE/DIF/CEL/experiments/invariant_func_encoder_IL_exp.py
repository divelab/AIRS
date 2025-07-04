import os
import pathlib
import time
from typing import Union, List, Literal

import equinox as eqx
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat, pack, unpack, reduce
from torch_scatter import scatter, scatter_std
from torchdiffeq import odeint as nn_odeint
from tqdm import tqdm

from CEL.data.data_manager import MyDataLoader
from CEL.networks.models.meta_model import ModelClass
from .meta_exp import ExpClass


class InvariantFuncEncILExperiment(ExpClass):

    def __init__(self, dataloader: MyDataLoader, model: ModelClass,
                 weight_decay: float = 0.0, lr: float = 1e-3,
                 max_epoch: int = 100, ctn_epoch: int = 0,
                 device: int = 0, log_interval: int = 20, wandb_logger: bool = True,
                 DEBUG: bool = False, num_sampling_per_batch: int = 1, partial_eval: bool = False,
                 lambda_reg: Union[float, List[float]] = 1.0, inference_out: str = 'inv', inv_type: str = 'VREx', adapt_steps: int = 10,
                 adapt_lr: float = 1e-2, **kwargs):
        r'''
        Invariant Physical Experiment
        Args:
            dataloader:
            model:
            weight_decay:
            lr:
            max_epoch:
            ctn_epoch:
            device:
            log_interval: Interval for wandb logging
            wandb_logger:
            DEBUG:
            num_sampling_per_batch: Each batch trajectory will be sampled this many times
            partial_eval: Whether to partially evaluate the model during testing phase because of the long computation time
            lambda_inv: Parameter for VREx or other invariant learning loss
            inference_out: During inference phases, whether to output the invariant or the combined derivative
        '''
        super().__init__()
        self.model = model
        self.loader = {set_name: dataloader.__getattribute__(set_name) for set_name in
                       ['train', 'test', 'id_test', 'ood_test'] if hasattr(dataloader, set_name)}
        self.pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                             'dynamic_ncols': True, 'ascii': '░▒█'}

        # exp configs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.ctn_epoch = ctn_epoch
        self.device = device
        self.log_interval = log_interval
        self.wandb_logger = wandb_logger
        self.DEBUG = DEBUG

        self.num_sampling_per_batch = num_sampling_per_batch
        self.partial_eval = partial_eval
        self.inference_out = inference_out

        # --- random seed generator ---
        self.sampling_generator = np.random.default_rng(0)

        # --- optimizer ---
        self.inv_type = inv_type
        self.lambda_reg = lambda_reg

        # --- adaptation ---
        self.adapt_steps = adapt_steps
        self.adapt_lr = adapt_lr


        # --- IRM ---
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(self.device)

    def inv_traj_inference(self, data, input_length, forecast_ode: Literal['inv', 'combine'], invariant_state_test: bool):

        states = data['states'].to(self.device)
        W = data['W'].to(self.device)
        t = data['t'].to(self.device)[0]
        dydt = data['dy'].to(self.device)

        # --- Infer derivative functions ---
        self.model(states, input_length, forecast_ode=forecast_ode)
        self.count = 0

        if invariant_state_test:
            # Replace the states with the invariant states for id_test invariant test.
            states = data['invariant_states'].to(self.device)

        with torch.no_grad():
            def inv_forward_ode_func(t, y):
                # print(f"\rCount: {self.count}", end="")
                self.count += 1
                pred_y = self.model(y, input_length)
                return pred_y

            def combine_forward_ode_func(t, y):
                pred_y = self.model(y, input_length)
                return pred_y

            forward_ode_func = {'inv': inv_forward_ode_func, 'combine': combine_forward_ode_func}

            try:
                tik = time.time()
                pred_y = nn_odeint(forward_ode_func['inv'], states[:, 0], t)
                print(f"\nODE integral time taken: {time.time() - tik} with {self.count} steps")  # about 5 mins
            except Exception as e:
                print(f"Exception {e}")
                pred_y = nn_odeint(forward_ode_func[self.inference_out], states[:, 0], t, method='rk4')

            nrmse = torch.sqrt(
                torch.mean((pred_y[input_length:] - states.transpose(0, 1)[input_length:]) ** 2)) / torch.std(
                states.transpose(0, 1)[input_length:])
        return nrmse.item()

    def train_batch(self, data, input_length) -> dict:
        r"""
        Train a batch. (Project use only)

        Returns:
            Calculated loss.
        """
        self.model.train()
        # data = data.to(self.config.device)
        states = data['states'].to(self.device)
        dydt = data['dy'].to(self.device)
        W = data['W'].to(self.device)
        t = data['t'].to(self.device)[0]
        env = data['env'].to(self.device)

        # --- Sample pairs ---
        loss = {}

        pred_dydt = self.model(states, input_length)

        if self.inv_type == 'IRM':
            from torch.autograd import grad
            pred_dydt = pred_dydt * self.dummy_w
            pred_loss = F.mse_loss(pred_dydt, dydt, reduction='none')

            loss_by_env = scatter(pred_loss, env, dim=0, reduce='mean')

            grad_by_env = [torch.norm(grad(l_e.mean(), self.dummy_w, create_graph=True)[0]) ** 2 for l_e in loss_by_env]
            grad_by_env: torch.Tensor = rearrange(grad_by_env, 'B -> B')

            irm_loss = grad_by_env.mean()
            pred_loss = pred_loss.mean()

            training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                             'irm': {'weight': self.lambda_reg, 'value': irm_loss}}
        elif self.inv_type == 'VREx':
            pred_loss = F.mse_loss(pred_dydt, dydt, reduction='none')

            loss_by_env = scatter(pred_loss, env, dim=0, reduce='mean')
            loss_by_env = reduce(loss_by_env, 'E T C -> E', 'mean')

            vrex_loss = loss_by_env.var(dim=0) # VREx loss across each environment
            pred_loss = pred_loss.mean()
            training_loss = {'pred': {'weight': 1.0, 'value': pred_loss},
                             'vrex': {'weight': self.lambda_reg, 'value': vrex_loss}}
        else:
            pred_loss = F.mse_loss(pred_dydt, dydt)
            training_loss = {'pred': {'weight': 1.0, 'value': pred_loss}}

        self.log_loss(loss, {key: loss_term['value'].item() for key, loss_term in training_loss.items()})

        # --- Backward ---
        self.optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        self.loss_wsum(training_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        # If RuntimeError: leaf variable has been moved into the graph interior, deepcopy the input model before deserializing.
        self.optimizer.step()

        return self.reduce_loss(loss)

    def reduce_loss(self, loss):
        return {key: np.mean(value) for key, value in loss.items()}

    def loss_wsum(self, loss):
        return sum(loss_term['weight'] * loss_term['value'] for loss_term in loss.values())

    def log_loss(self, loss, new_loss):
        for key, value in new_loss.items():
            if key not in loss:
                loss[key] = []
            loss[key].append(value)
        return loss

    def __call__(self, *args, **kwargs):
        # config model
        print('#D#Config model')
        self.config_model('train')

        # config optimizer
        print('#D#Config optimizer')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch)
        best_test_error = {
            'id_test':
                {
                    'inv': {'original_states': 1e10, 'invariant_states': 1e10},
                    'combine': {'original_states': 1e10, 'invariant_states': 1e10}
                },
            'ood_test':
                {'inv': 1e10, 'combine': 1e10}
        }
        # train the model
        for epoch in range(self.ctn_epoch, self.max_epoch + 1):
            self.epoch = epoch
            print(f'#IN#Epoch {epoch}:')

            loss_log = {}
            mean_loss = 0

            # self.ood_algorithm.stage_control(self.config)

            pbar = enumerate(self.loader['train']) if self.wandb_logger else tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **self.pbar_setting)
            for index, traj_batch in pbar:
                if self.unbalanced_env(traj_batch):
                    continue

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.max_epoch
                self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # train a batch
                try:
                    train_loss = self.train_batch(traj_batch, input_length=self.loader['train'].dataset.input_length)
                    loss_log = self.log_loss(loss_log, train_loss)
                    mean_loss = self.reduce_loss(loss_log)
                    if not self.wandb_logger:
                        pbar.set_description(
                            '|'.join([f'{loss_key}: {loss_value:.2e}' for loss_key, loss_value in mean_loss.items()]))
                except ValueError as e:
                    print(e)


            if epoch % 100 == 0:
                mean_test_error = dict()
                for split in best_test_error.keys():
                    mean_test_error[split] = dict()
                    for ode_type in best_test_error[split]:
                        if split == 'id_test':
                            mean_test_error[split][ode_type] = dict()
                            for state_type in best_test_error[split][ode_type]:
                                invariant_state_test = True if state_type == 'invariant_states' else False
                                mean_test_error[split][ode_type][state_type] = self.evaluate(split, forecast_ode=ode_type, invariant_state_test=invariant_state_test)
                                best_test_error[split][ode_type][state_type] = self.report_evaluation(best_test_error[split][ode_type][state_type],
                                                                                                  epoch,
                                                                                                  mean_test_error[split][ode_type][state_type],
                                                                                                  f'{split}_{state_type}',
                                                                                                  ode_type,
                                                                                                  kwargs)
                        else:
                            mean_test_error[split][ode_type] = self.evaluate(split, forecast_ode=ode_type)
                            best_test_error[split][ode_type] = self.report_evaluation(best_test_error[split][ode_type],
                                                                                      epoch,
                                                                                      mean_test_error[split][ode_type],
                                                                                      split=split, ode_type=ode_type,
                                                                                      kwargs=kwargs)

                print(f'lr: {self.optimizer.param_groups[0]["lr"]}')
                # --- Logging ---
                if epoch % self.log_interval == 0 and self.wandb_logger:
                    wandb.log({
                        'train_loss': mean_loss,
                        'test_NRMSE': mean_test_error,
                        'Best_NRMSE': best_test_error,
                        'lr': self.optimizer.param_groups[0]['lr'],
                    }, step=epoch)
            # self.scheduler.step(epoch)

        # self.explain_in_pysr(kwargs['exp_hash'])

        print('#IN#Training end.')

    def explain_in_pysr(self, exp_hash, epoch=None):
        checkpoint_path = pathlib.Path(os.environ['STORAGE_DIR']) / 'exp' / exp_hash

        if epoch:
            epoch = epoch
        else:
            # get the last epoch number in the checkpoint path
            saved_epochs = [int(file.stem) for file in checkpoint_path.iterdir() if file.stem.isdigit()]
            epoch = last_epoch = sorted(saved_epochs, reverse=True)[0]
        self.load_model(path=checkpoint_path, epoch=epoch)
        self.model.to(self.device)
        self.model.eval()
        self.evaluate('id_test', forecast_ode='inv', invariant_state_test=True)

        from pysr import PySRRegressor
        batch_data_list = list(iter(self.loader['id_test']))
        scaling = self.loader['id_test'].dataset.max_for_scaling.numpy()

        from sympy import Number
        def round_expr(expr, num_digits):
            return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})

        def model_fit_sym(data, input_length=1, forecast_ode='inv'):
            states = data['states'] # bs x T x Cy
            invariant_states = data['invariant_states']

            # use non-invariant states to predict invariant derivative function
            self.model(states.to(self.device), input_length, forecast_ode=forecast_ode)
            # use invariant states as the input distribution, to check derivatives from the predicted invariant derivative function
            pred_dy = self.model(invariant_states.to(self.device), input_length)
            W = repeat(data['W'], 'bs Cw -> bs T Cw', T=states.shape[1])
            yinv_W, ps = pack([invariant_states, W], 'bs T *') # bs T (Cy + Cw)

            # --- use symbolic model to explain the extracted invariant function ---
            sr_model = PySRRegressor(
                niterations=50,
                populations=30,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["cos", "exp", "sin"],
                model_selection="best",
                # complexity_of_constants=2,
                # complexity_of_operators={"sin": 2, "cos": 2, "exp": 2},
                batching=True
            )
            predicted_dy = pred_dy.cpu().detach().numpy()
            sr_model.fit(rearrange(yinv_W, 'bs T C -> (bs T) C').numpy(),
                         rearrange(predicted_dy, 'bs T Cy -> (bs T) Cy'))
            return sr_model

        inv_train_sr = model_fit_sym(batch_data_list[0], self.loader['id_test'].dataset.input_length, 'inv')
        combine_train_sr = model_fit_sym(batch_data_list[0], self.loader['id_test'].dataset.input_length, 'combine')

        def expr_selection(sr_model, equation_idx=0):
            sorted_eq = sr_model.equations_[equation_idx].sort_values(by='score', ascending=False)
            print(sorted_eq)
            min_loss = sorted_eq['loss'].min()
            for i in range(len(sorted_eq)):
                if sorted_eq.iloc[i]['loss'] < min_loss * 3:
                    return sorted_eq.iloc[i]

        print('=' * 80)
        print('Extracted Invariant Function:')
        invariant_function = [round_expr(inv_train_sr.sympy()[0], 1), round_expr(inv_train_sr.sympy()[1], 1)]
        # invariant_function = [round_expr(expr_selection(inv_train_sr, 0)['sympy_format'], 1), round_expr(expr_selection(inv_train_sr, 1)['sympy_format'], 1)]
        print(invariant_function[0])
        print(invariant_function[1])
        print('-' * 80)
        print('Predicted Function')
        combine_function = [round_expr(combine_train_sr.sympy()[0], 1), round_expr(combine_train_sr.sympy()[1], 1)]
        # combine_function = [round_expr(expr_selection(combine_train_sr, 0)['sympy_format'], 1), round_expr(expr_selection(combine_train_sr, 1)['sympy_format'], 1)]
        print(combine_function[0])
        print(combine_function[1])
        print('=' * 80)
        if self.wandb_logger:
            wandb.log({
                'Invariant Function': f"{invariant_function}",
                'Combined Function': f"{combine_function}"
            })


    def report_evaluation(self, best_error, epoch, mean_error, split, ode_type, kwargs):
        if mean_error < best_error:
            best_error = mean_error
            if ode_type == 'inv' and split == 'id_test_invariant_states':
                self.save_model(path=pathlib.Path(os.environ['STORAGE_DIR']) / 'exp' / kwargs['exp_hash'],
                                dumped_config=kwargs['dumped_config'],
                                epoch=epoch)
            print(f'Best {split} {ode_type} error: {best_error:.4f}')
        print(f'Test {split} {ode_type} error: {mean_error:.4f}, Best {split} {ode_type} error: {best_error:.4f}')
        return best_error

    def unbalanced_env(self, traj_batch):
        env = traj_batch['env'].to(self.device)
        valid_mask = scatter(torch.ones_like(env, device=self.device, dtype=int), env, dim=0, reduce='sum') > 2
        return not valid_mask.all()

    def evaluate(self, split: str, forecast_ode: str, invariant_state_test: bool = False):
        self.model.eval()
        test_errors = []
        input_length = self.loader[split].dataset.input_length
        for index, data in tqdm(enumerate(self.loader[split])):
            errors = self.inv_traj_inference(data, input_length, forecast_ode, invariant_state_test)
            test_errors.append(errors)
            print(np.mean(test_errors))
            if self.partial_eval and index >= 0:
                break
        mean_test_error = np.mean(test_errors)
        return mean_test_error

    def save_model(self, path: pathlib.Path, dumped_config: str, epoch: int):
        os.makedirs(path, exist_ok=True)
        with open(path / f'config.yml', 'w') as f:
            f.write(dumped_config)
        if issubclass(self.model.__class__, eqx.Module):
            with open(path / f'{epoch}.eqx', 'wb') as f:
                eqx.tree_serialise_leaves(f, self.model)
        else:
            torch.save(self.model.state_dict(), path / f'{epoch}.pt')

    def load_model(self, path: pathlib.Path, epoch: int):
        if issubclass(self.model.__class__, eqx.Module):
            with open(path / f'{epoch}.eqx', 'rb') as f:
                return eqx.tree_deserialise_leaves(f, self.model)
        else:
            self.model.load_state_dict(torch.load(path / f'{epoch}.pt'))

    def config_model(self, mode: str, load_param=False):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.device)
        self.model.train()

    def run(self):
        self.intervention.run(self.intervention_type, **self.intervention_kwargs)
        self.model.fit(self.data)
        self.results = self.model.evaluate(self.data)
