import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config
config.update("jax_enable_x64", True)

def _value_step(step, val):
    """Performs a single step of the value MPS evaluation."""
    env, state, mps_params = val
    env = jnp.matmul(env, mps_params[state[step], :, :])
    return env, state, mps_params

def value(state, mps_params):
    """Performs the value MPS evaluation for the given state and parameters."""
    env = mps_params[state[0], :, :]
    val = (env, state, mps_params)
    val = fori_loop(1, len(state), _value_step, val)
    return jnp.log(jnp.trace(val[0])**2)
