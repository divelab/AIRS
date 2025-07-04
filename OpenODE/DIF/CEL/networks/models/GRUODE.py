import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.lax as lax
import diffrax
from CEL.utils.register import register
import jax.random as jr

from typing import Literal

from .meta_model import ModelClass

class ODEFunc(eqx.Module):
    mlp: eqx.nn.MLP
    input_channels: int
    hidden_channels: int

    def __init__(self, input_channels, hidden_channels, *, key):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.mlp = eqx.nn.MLP(
            in_size=hidden_channels,
            out_size=hidden_channels,
            width_size=128,
            depth=2,
            activation=jnn.softplus,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            final_activation=jnn.tanh,
            key=key,
        )


    def __call__(self, t, z, args):
        z = self.mlp(z)
        # z = jnp.matmul(z, jnp.diag(self.W))  # Uncomment if W is to be used

        return z

class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    # linear: eqx.nn.Linear
    # bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        # self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        # self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        # return jax.nn.sigmoid(self.linear(out) + self.bias)
        return out

# Note: Further integration with diffrax to create a CDE system would be needed.

@register.model_register
class GRUODE(eqx.Module, ModelClass):
    gru: RNN
    ode_func: ODEFunc
    readout: eqx.nn.Linear

    def __init__(
            self,
            input_channels_x: int,
            hidden_channels_x: int,
            output_channels: int,
            *,
            key: int
    ):
        super(GRUODE, self).__init__()

        key = jr.PRNGKey(key)
        gkey, fkey, rkey = jr.split(key, 3)
        self.gru = RNN(in_size=input_channels_x, out_size=hidden_channels_x,
                       hidden_size=hidden_channels_x, key=gkey)
        self.ode_func = ODEFunc(hidden_channels_x, hidden_channels_x, key=fkey)
        self.readout = eqx.nn.Linear(hidden_channels_x, output_channels, key=rkey)
        # self.dropout_layer = eqx.nn.Dropout(0.1)

        # logging.info(f"Interpolation type: {self.interpolation}")

    def __call__(self, y_past, t, coeffs_x, input_length):
        t_past, t_future = t[:input_length], t[input_length:]

        zc = self.gru(y_past)

        term = diffrax.ODETerm(self.ode_func)
        solver = diffrax.Dopri5()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t_future[0],
            t_future[-1],
            None,
            zc,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=t_future)
        )
        y_hat = jax.vmap(self.readout)(sol.ys)
        return y_hat



