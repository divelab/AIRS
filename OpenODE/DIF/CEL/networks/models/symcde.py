import diffrax
import equinox as eqx
import jax
# import jax.numpy as jnp
from jax import random as jr, numpy as jnp
import pysindy as ps
import sympy as sp
import sympy2jax as sp2j
import numpy as np

from CEL.networks.models.PhyCDE import CDEModel, RNN


@jax.custom_vjp
def ste_func(input):
    return (input > 0).astype(jnp.float32)

def ste_forward(input):
    return ste_func(input), jax.nn.sigmoid(input)

def ste_backward(sigmoid_input, grad_output):
    # Backward pass: use the saved sigmoid_input to compute the gradient
    grad_input = grad_output * sigmoid_input * (1 - sigmoid_input)
    return (grad_input,)

ste_func.defvjp(ste_forward, ste_backward)


def concrete_sample(rng_key, att_log_logit, temp=1.0, training=True):
    if training:
        # Split the RNG key
        rng_key, new_rng_key = jax.random.split(rng_key)

        # Generate uniform noise
        random_noise = jax.random.uniform(rng_key, att_log_logit.shape, minval=1e-10, maxval=1 - 1e-10)
        random_noise = jnp.log(random_noise) - jnp.log(1.0 - random_noise)

        # Compute the concrete sample
        att_bern = jax.nn.sigmoid((att_log_logit + random_noise) / temp)
    else:
        att_bern = jax.nn.sigmoid(att_log_logit)

    return att_bern

# Function to parse and convert feature names to SymPy expressions
def parse_feature_names(feature_names):
    sympy_expressions = []
    for feature in feature_names:
        # Replace ^ with ** for Python syntax
        feature = feature.replace("^", "**")

        # Replace space with * for multiplication
        feature = feature.replace(" ", "*")

        # Convert to SymPy expression
        expr = sp.sympify(feature)
        sympy_expressions.append(expr)
    return sympy_expressions

class SymbolicCDEFunc(eqx.Module):
    # mlp: eqx.nn.MLP
    control_channels: int
    # hidden_channels: int
    symbolic_model: sp2j.SymbolicModule
    state_channels: int
    feature_library: ps.ConcatLibrary
    sym_equation: sp.Expr
    xi: jnp.ndarray
    ckey: jax.random.PRNGKey

    def __init__(self, control_channels, state_channels, depth_dfunc=2, width_dfunc=128, *, key):
        self.control_channels = control_channels
        # self.hidden_channels = hidden_channels
        self.state_channels = state_channels

        # Sindy symbolic regression
        # polynomial_power = 3
        # polynomial_library = ps.PolynomialLibrary(degree=polynomial_power)
        # fourier_library = ps.FourierLibrary(n_frequencies=1)
        # self.feature_library = ps.ConcatLibrary([polynomial_library, fourier_library])
        self.feature_library = ps.PolynomialLibrary(degree=2)
        # self.feature_library = ps.PolynomialLibrary(degree=2, interaction_only=True)
        self.feature_library.fit(jnp.zeros((1, state_channels)))
        self.symbolic_model = sp2j.SymbolicModule(parse_feature_names(self.feature_library.get_feature_names()))
        self.sym_equation = self.symbolic_model.sympy()

        xikey, ckey = jr.split(key)
        self.xi = jax.random.normal(xikey, (state_channels * control_channels * self.feature_library.n_output_features_,))

        self.ckey = ckey

    def __call__(self, t, z, args):
        # inputs: W: state_channels x control_channels x size_feature_library, y: state_channels
        y = z
        W = args[0]
        # y, W = z[:self.state_channels], z[self.state_channels:]
        transformed_y = jnp.stack(self.symbolic_model(x0=y[0], x1=y[1]))
        dydx = (W * concrete_sample(self.ckey, self.xi)).reshape(self.state_channels, self.control_channels, -1) @ transformed_y
        # dydx = (W * ste_func(self.xi)).reshape(self.state_channels, self.control_channels, -1) @ transformed_y
        out = dydx
        # jax.debug.print("t: {t}\n"
        #                 "transformed_y: {transformed_y}\n"
        #                 "W: {W}\n"
        #                 # "xi: {xi}\n"
        #                 "dydx: {dydx}\n", transformed_y=transformed_y, dydx=dydx, t=t, W=W * ste_func(self.xi))
        # out = dydx.reshape(self.state_channels, self.control_channels)
        # dydx = transformed_y @ (self.W * jax.nn.sigmoid(self.xi))
        # z = self.mlp(z).reshape(self.state_channels, self.control_channels)
        # ┌                     ┐
        # │ d(y1)    d(y1)      │
        # │ ────     ────       │
        # │  d(x)    d(t)       │
        # │                     │
        # │ d(y2)    d(y2)      │
        # │ ────     ────       │
        # │  d(x)    d(t)       │
        # └                     ┘

        return out


class SymbolicCDE(CDEModel):
    rnn: RNN
    symcde_func: SymbolicCDEFunc
    output_channels: int
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
        super(SymbolicCDE, self).__init__(interpolation)
        key = jr.PRNGKey(key)
        ikey, fkey, rkey = jr.split(key, 3)
        time_channel = 1
        # self.initial = eqx.nn.Linear(input_channels_x, hidden_channels_x, key=ikey)
        self.symcde_func = SymbolicCDEFunc(X_channels + time_channel, state_channels, key=fkey)
        self.rnn = RNN(state_channels + X_channels + time_channel, state_channels * (X_channels + time_channel) * self.symcde_func.feature_library.n_output_features_, hidden_channels, key=ikey)
        self.output_channels = output_channels
        self.readout = eqx.nn.Linear(hidden_channels, output_channels, key=rkey)

        self.dropout_layer = eqx.nn.Dropout(0.1)


    def __call__(self, y_past, t, coeffs_x, input_length, prediction_length):
        # jax.debug.print('prediction_length: {prediction_length}\n', prediction_length=prediction_length)
        t_past, t_future = t[:input_length], t[input_length:]
        control = self.interpolate(coeffs_x, t)
        term_future = diffrax.ControlTerm(self.symcde_func, control).to_ode()
        solver = diffrax.Euler()

        hidden, Ws = self.rnn(jnp.concatenate((y_past, jax.vmap(control.evaluate)(t_past)), axis=1))

        # z0 = jnp.concatenate((y_past[-1], Ws[-1]))

        solution = diffrax.diffeqsolve(
            term_future,
            solver,
            t_past[-1],
            t_future[prediction_length - 1],
            dt0=t_future[0] - t_past[-1],
            y0=y_past[-1],
            args=(Ws[-1],),
            # stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
            saveat=diffrax.SaveAt(ts=t_future[:prediction_length]),
        )

        # T x [x, v, ...]
        # y_hat = jax.vmap(self.readout)(solution.ys)
        y_hat = solution.ys[:, :self.output_channels]

        # jax.debug.print('prediction_length: {prediction_length}', prediction_length=prediction_length)
        # jax.debug.print('prediction_length: {prediction_length}\n'
        #                 'y_hat: {y_hat}\n'
        #                 'W: {W}\n'
        #                 'xi: {xi}', prediction_length=prediction_length, y_hat=y_hat, W=jnp.abs(Ws[-1]).max(),
        #                 xi=jnp.abs(self.symcde_func.xi).max())
        y_out = jnp.concatenate((y_hat, jnp.zeros((t_future.shape[0] - prediction_length, y_hat.shape[1]))), axis=0)

        return y_out, Ws, self.symcde_func.xi

