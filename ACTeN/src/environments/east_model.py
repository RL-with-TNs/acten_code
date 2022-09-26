import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config
config.update("jax_enable_x64", True)

def periodic_reward_func(state, action, environment_args):
    reward = cond(action == environment_args["L"],
                  lambda x: jnp.log(1 - jnp.sum(x)/environment_args["L"]),
                  lambda x: -environment_args["s"] - jnp.log(environment_args["L"]),
                  state)
    return reward

def _flip(state, site):
    state = state.at[site].set(-(state[site]-1))
    return state

def step(state, action, environment_args):
    state = cond(action == environment_args["L"],
                 lambda x: x,
                 lambda x: _flip(x, action),
                 state)
    return state

def periodic_original_sample(state, key, params):
    key, subkey = random.split(key)
    L = jnp.size(state)
    site_index = random.randint(subkey, (1,), 0, L)[0]
    action = cond(state[site_index],
                 lambda s: ((s + 1) % L),
                 lambda s: L,
                 site_index)
    probability = cond(action == L,
                       lambda s: 1 - jnp.sum(s)/L,
                       lambda s: 1/L,
                       state)
    return action, key, probability

def periodic_original_log_prob(state, action, params):
    return 0.0

def activity_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
    activity = cond(action == environment_args["L"], lambda: 0, lambda: 1)
    observation_state += learning_rate * (activity - observation_state)
    return observation_state

def kl_divergence_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
    kl_div = cond(action == environment_args["L"],
                  lambda x: jnp.log(1 - jnp.sum(x)/environment_args["L"]),
                  lambda x: -jnp.log(environment_args["L"]),
                  prior_state)
    observation_state += learning_rate * (kl_div - log_prob - observation_state)
    return observation_state

def init_obs_func(activity_learning_rate, kl_div_learning_rate):
    def obs_func(
        observation_state, step,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args
        ):
        observation_state = observation_state.at[0].set(average_reward)
        activity = cond(action == environment_args["L"], lambda: 0, lambda: 1)
        observation_state = observation_state.at[1].add(
            activity_learning_rate * (activity - observation_state[1]))
        kl_div = cond(action == environment_args["L"],
                    lambda x: jnp.log(1 - jnp.sum(x)/environment_args["L"]),
                    lambda x: -jnp.log(environment_args["L"]),
                    prior_state)
        observation_state = observation_state.at[2].add(
            kl_div_learning_rate * (kl_div - log_prob - observation_state[2]))
        return observation_state
    return obs_func
