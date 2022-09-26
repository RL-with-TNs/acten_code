import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config
config.update("jax_enable_x64", True)

def _value_step_func(state, mps_params):
    def _value_step(step, m):
        return jnp.matmul(m, mps_params[state[step], :, :])
    return _value_step

def value_closure(state, mps_params):
    env = mps_params[state[0], :, :]
    value_step = _value_step_func(state, mps_params)
    env = fori_loop(1, len(state), value_step, env)
    return jnp.log(jnp.trace(env)**2)

def _value_step(step, val):
    env, state, mps_params = val
    env = jnp.matmul(env, mps_params[state[step], :, :])
    return env, state, mps_params

def value_scan(state, mps_params):
    # chi_value_function = mps_params.shape[-1]
    # tensor = (jnp.array([jnp.eye(chi_value_function), jnp.eye(chi_value_function)]) + mps_params)
    tensor = mps_params
    env = tensor[state[0], :, :]
    val = (env, state, tensor)
    val = fori_loop(1, len(state), _value_step, val)
    # return jnp.log((jnp.trace(val[0])/chi_value_function)**2)
    return jnp.log(jnp.trace(val[0])**2)
