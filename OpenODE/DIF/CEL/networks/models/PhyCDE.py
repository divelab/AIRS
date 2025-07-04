import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.lax as lax
import diffrax
import jax.random as jr
from typing import Union

from typing import Literal

from .meta_model import ModelClass

class CDEFunc(eqx.Module):
    mlp: eqx.nn.MLP
    control_channels: int
    state_channels: int

    def __init__(self, control_channels, state_channels, depth_dfunc=2, width_dfunc=128, *, key):
        self.control_channels = control_channels
        self.state_channels = state_channels

        # self.linear1 = eqx.nn.Linear(hidden_channels, 128)
        # self.linear2 = eqx.nn.Linear(128, input_channels * hidden_channels)
        self.mlp = eqx.nn.MLP(
            in_size=state_channels,
            out_size=state_channels * control_channels,
            width_size=width_dfunc,
            depth=depth_dfunc,
            activation=jnn.softplus,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            final_activation=jnn.tanh,
            key=key,
        )


    def __call__(self, t, z, args):
        # jax.debug.print("t: {t}\n", t=t)
        z = self.mlp(z).reshape(self.state_channels, self.control_channels)
        # ┌                     ┐
        # │ d(y1)    d(y1)      │
        # │ ────     ────       │
        # │  d(x)    d(t)       │
        # │                     │
        # │ d(y2)    d(y2)      │
        # │ ────     ────       │
        # │  d(x)    d(t)       │
        # └                     ┘
        # z = jnp.matmul(z, jnp.diag(self.W))  # Uncomment if W is to be used

        return z


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    # bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)
        # self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry_hidden, input):
            carry = self.cell(input, carry_hidden)
            return carry, carry

        out, all_outs = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        # return jax.nn.sigmoid(self.linear(out) + self.bias)
        out_ys = jax.vmap(self.linear)(all_outs)
        return out, out_ys


class CDEModel(eqx.Module, ModelClass):
    interpolation: str
    def __init__(self, interpolation):
        super().__init__()
        self.interpolation = interpolation

    def interpolate(self, coeffs_x, t):
        if coeffs_x is not None:
            coeffs_x = jnp.concatenate((coeffs_x, t[:, None]), axis=1)
        else:
            coeffs_x = t[:, None]
        if self.interpolation == "cubic":
            control = diffrax.CubicInterpolation(t, diffrax.backward_hermite_coefficients(t, coeffs_x))
        elif self.interpolation == "linear":
            control = diffrax.LinearInterpolation(t, coeffs_x)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented.",
            )
        return control


class GRUCDE(CDEModel):
    initial: RNN
    cde_func: CDEFunc
    readout: eqx.nn.Linear
    dropout_layer: eqx.nn.Dropout


    def __init__(
            self,
            state_channels: int,
            X_channels: int,
            hidden_channels: int,
            output_channels: int,
            interpolation: str ="linear",
            *,
            key: int
    ):
        super(GRUCDE, self).__init__(interpolation)
        key = jr.PRNGKey(key)
        ikey, fkey, rkey = jr.split(key, 3)
        # self.initial = eqx.nn.Linear(input_channels_x, hidden_channels_x, key=ikey)
        self.initial = RNN(state_channels + X_channels, hidden_channels, hidden_channels, key=ikey)
        self.cde_func = CDEFunc(X_channels, hidden_channels, key=fkey)
        # self.combine_z = torch.nn.Linear(
        #     (hidden_channels_x + hidden_channels_x) // 2,
        #     hidden_channels_x * hidden_channels_x,
        # )
        self.readout = eqx.nn.Linear(hidden_channels, output_channels, key=rkey)
        # self.treatment = torch.nn.Linear(hidden_channels_x, 4)
        # self.softmax = torch.nn.Softmax(dim=4)
        # a jax dropout layer
        self.dropout_layer = eqx.nn.Dropout(0.1)


    def __call__(self, y_past, t, coeffs_x, input_length):

        t_past, t_future = t[:input_length], t[input_length:]
        control = self.interpolate(coeffs_x, t)
        term_future = diffrax.ControlTerm(self.cde_func, control).to_ode()
        solver = diffrax.Dopri5()

        z0, _ = self.initial(jnp.concatenate((y_past, jax.vmap(control.evaluate)(t_past)), axis=1))

        solution = diffrax.diffeqsolve(
            term_future,
            solver,
            t_future[0],
            t_future[-1],
            dt0=None,
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t_future)
        )

        # bs x [x, v] x T
        y_hat = jax.vmap(self.readout)(solution.ys)

        return y_hat


class HiddenCDE(CDEModel):
    initial: eqx.nn.Linear
    cde_func_past: CDEFunc
    cde_func_future: CDEFunc
    readout: eqx.nn.Linear
    dropout_layer: eqx.nn.Dropout

    def __init__(
            self,
            state_channels: int,
            X_channels: int,
            hidden_channels: int,
            output_channels: int,
            interpolation: str ="linear",
            *,
            key: int
    ):
        super(HiddenCDE, self).__init__(interpolation)
        key = jr.PRNGKey(key)
        ikey, f1key, f2key, rkey = jr.split(key, 4)
        self.initial = eqx.nn.Linear(state_channels + X_channels, hidden_channels, key=ikey)
        self.cde_func_past = CDEFunc(state_channels + X_channels, hidden_channels, key=f1key)
        self.cde_func_future = CDEFunc(X_channels, hidden_channels, key=f2key)
        # self.combine_z = torch.nn.Linear(
        #     (hidden_channels_x + hidden_channels_x) // 2,
        #     hidden_channels_x * hidden_channels_x,
        # )
        self.readout = eqx.nn.Linear(hidden_channels, output_channels, key=rkey)
        # self.treatment = torch.nn.Linear(hidden_channels_x, 4)
        # self.softmax = torch.nn.Softmax(dim=4)
        # a jax dropout layer
        self.dropout_layer = eqx.nn.Dropout(0.1)


    def __call__(self, y_past, t, coeffs_x, input_length):
        t_past, t_future = t[:input_length], t[input_length:]
        if coeffs_x is not None:
            X_past = coeffs_x[:input_length]
            X_future = coeffs_x[input_length:]
            yX_past = jnp.concatenate((y_past, X_past), axis=1)
        else:
            X_past = None
            X_future = None
            yX_past = y_past
        control_past = self.interpolate(yX_past, t_past)
        control_future = self.interpolate(X_future, t_future)
        term_past = diffrax.ControlTerm(self.cde_func_past, control_past).to_ode()
        term_future = diffrax.ControlTerm(self.cde_func_future, control_future).to_ode()
        solver = diffrax.Dopri5()

        z0 = self.initial(control_past.evaluate(t_past[0]))
        sol_past = diffrax.diffeqsolve(
            term_past,
            solver,
            t_past[0],
            t_past[-1],
            dt0=None,
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            # saveat=diffrax.SaveAt(ts=t_past[-1])
        )

        solution = diffrax.diffeqsolve(
            term_future,
            solver,
            t_future[0],
            t_future[-1],
            dt0=None,
            y0=sol_past.ys[-1],
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t_future)
        )

        # bs x [x, v] x T
        y_hat = jax.vmap(self.readout)(solution.ys)

        return y_hat


class DoubleControlCDE(CDEModel):
    zy_initial: eqx.nn.Linear
    num_zx: int
    zx_initial: eqx.nn.Linear
    zx_encoder: Union[CDEFunc, RNN]
    zx_readout: eqx.nn.Linear
    zy_encoder: Union[CDEFunc, RNN]
    zy_cde_future: CDEFunc
    readout: eqx.nn.Linear
    dropout_layer: eqx.nn.Dropout
    zx_encoder_option: Literal['cde', 'gru']
    zy_encoder_option: Literal['cde', 'gru']
    plan_option: Literal['full', 'partial']

    def __init__(
            self,
            state_channels: int,
            X_channels: int,
            num_zx: int,
            hidden_channels: int,
            output_channels: int,
            zx_encoder_option: Literal['cde', 'gru'],
            zy_encoder_option: Literal['cde', 'gru'],
            plan_option: Literal['full', 'partial'],
            interpolation: str ="linear",
            *,
            key: int
    ):
        super(DoubleControlCDE, self).__init__(interpolation)

        # --- Options ---
        self.zx_encoder_option = zx_encoder_option
        self.plan_option = plan_option
        self.zy_encoder_option = zy_encoder_option

        # --- Build modules ---
        key = jr.PRNGKey(key)
        ikey, zxkey, zxrkey, zy1key, zy2key, rkey = jr.split(key, 6)
        time_channel = 1

        # -- First stage --
        self.num_zx = num_zx
        self.zx_initial = eqx.nn.Linear(X_channels + time_channel, hidden_channels, key=ikey)
        if self.zx_encoder_option == 'cde':
            self.zx_encoder = CDEFunc(X_channels + time_channel, hidden_channels, key=zxkey)
        elif self.zx_encoder_option == 'gru':
            self.zx_encoder = RNN(X_channels + time_channel, self.num_zx, hidden_size=hidden_channels, key=ikey)
        self.zx_readout = eqx.nn.Linear(hidden_channels, self.num_zx, key=zxrkey)

        # -- Second stage --

        if self.plan_option == 'full':
            info_channels = state_channels + self.num_zx + time_channel
        elif self.plan_option == 'partial':
            info_channels = state_channels + X_channels + self.num_zx + time_channel

        self.zy_initial = eqx.nn.Linear(info_channels, hidden_channels, key=ikey)
        if self.zy_encoder_option == 'cde':
            self.zy_encoder = CDEFunc(info_channels, hidden_channels, key=zy1key)
        elif self.zy_encoder_option == 'gru':
            self.zy_encoder = RNN(info_channels, hidden_channels, hidden_size=hidden_channels, key=zy1key)

        self.zy_cde_future = CDEFunc(info_channels - state_channels, hidden_channels, key=zy2key)

        self.readout = eqx.nn.Linear(hidden_channels, output_channels, key=rkey)
        self.dropout_layer = eqx.nn.Dropout(0.1)


    def __call__(self, y_past, t, coeffs_x, input_length):
        r'''if self.effect_option == 'full': for the first stage
            # First CDE/GRU for intervention plan's instant effects + partial effects
            # Second CDE/GRU for explicit equation
        elif self.effect_option == 'partial':
            # First CDE/GRU for intervention plan's partial effects only
            # Second CDE/GRU for explicit equation + intervention plan's instant effects'''

        # First CDE/GRU for intervention plan's instant effects + partial effects [OR partial effects only]
        control_Xt = self.interpolate(coeffs_x, t)
        if self.zx_encoder_option == 'cde':
            term_zx = diffrax.ControlTerm(self.zx_encoder, control_Xt).to_ode()
            solver = diffrax.Dopri5()

            zx0 = self.zx_initial(control_Xt.evaluate(t[0]))
            sol_zx = diffrax.diffeqsolve(
                term_zx,
                solver,
                t[0],
                t[-1],
                dt0=None,
                y0=zx0,
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                saveat=diffrax.SaveAt(ts=t)
            )
            # CDE makes zx partial
            zx = jax.vmap(self.zx_readout)(sol_zx.ys)
        elif self.zx_encoder_option == 'gru':
            # GRU makes zx discrete
            _, zx = self.zx_encoder(jax.vmap(control_Xt.evaluate)(t))


        # Second CDE/GRU for explicit equation [OR + intervention plan's instant effects]

        t_past, t_future = t[:input_length], t[input_length:]
        if coeffs_x is not None:
            X_past = coeffs_x[:input_length]
            X_future = coeffs_x[input_length:]
            yX_past = jnp.concatenate((y_past, X_past), axis=1)
        else:
            X_past = None
            X_future = None
            yX_past = y_past
        if self.plan_option == 'full':
            yzx_past = jnp.concatenate((y_past, zx[: input_length]), axis=1) if zx is not None else yX_past
            control_past = self.interpolate(yzx_past, t_past)
            control_future = self.interpolate(zx[input_length: ], t_future)
        elif self.plan_option == 'partial':
            yXzx_past = jnp.concatenate((yX_past, zx[: input_length]), axis=1) if zx is not None else yX_past
            Xzx_future = jnp.concatenate((X_future, zx[input_length: ]), axis=1) if zx is not None else X_future
            control_past = self.interpolate(yXzx_past, t_past)
            control_future = self.interpolate(Xzx_future, t_future)

        # --- Parameter inference phase ---
        solver = diffrax.Dopri5()
        if self.zy_encoder_option == 'cde':
            term_past = diffrax.ControlTerm(self.zy_encoder, control_past).to_ode()

            zy0 = self.zy_initial(control_past.evaluate(t_past[0]))
            sol_past = diffrax.diffeqsolve(
                term_past,
                solver,
                t_past[0],
                t_past[-1],
                dt0=None,
                y0=zy0,
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                # saveat=diffrax.SaveAt(ts=t_past[-1])
            )

            zy_current = sol_past.ys[-1]
        elif self.zy_encoder_option == 'gru':
            zy_current, _ = self.zy_encoder(jax.vmap(control_past.evaluate)(t_past))


        # --- Future prediction phase ---
        term_future = diffrax.ControlTerm(self.zy_cde_future, control_future).to_ode()
        solution = diffrax.diffeqsolve(
            term_future,
            solver,
            t_future[0],
            t_future[-1],
            dt0=None,
            y0=zy_current,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t_future)
        )

        # solution.ys: T x hidden_channels
        # Do we really need this readout linear transformation?
        # The carry part of the CDE should include the information of physical states and the system parameters (for different tasks, meta learning?)
        y_hat = jax.vmap(self.readout)(solution.ys)

        return y_hat


class MetaCDE(CDEModel):
    initial: eqx.nn.Linear
    cde_func_past: CDEFunc
    cde_func_future: CDEFunc
    readout: eqx.nn.Linear
    dropout_layer: eqx.nn.Dropout

    def __init__(
            self,
            state_channels: int,
            X_channels: int,
            hidden_channels: int,
            output_channels: int,
            interpolation: str ="linear",
            *,
            key: int
    ):
        super(MetaCDE, self).__init__(interpolation)
        key = jr.PRNGKey(key)
        ikey, f1key, f2key, rkey = jr.split(key, 4)
        self.cde_func = CDEFunc(X_channels, state_channels, key=f1key)
        # self.treatment = torch.nn.Linear(hidden_channels_x, 4)
        # self.softmax = torch.nn.Softmax(dim=4)
        # a jax dropout layer
        self.dropout_layer = eqx.nn.Dropout(0.1)


    def __call__(self, y_past, t, coeffs_x, input_length):
        control = self.interpolate(coeffs_x, t)
        term = diffrax.ControlTerm(self.cde_func, control).to_ode()
        solver = diffrax.Dopri5()

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t[0],
            t[-1],
            dt0=None,
            y0=y_past,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t)
        )

        # bs x T x [x, v]
        y_hat = solution.ys

        return y_hat