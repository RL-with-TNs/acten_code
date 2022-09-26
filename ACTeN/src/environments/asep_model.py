import jax.numpy as jnp
from jax import grad, jit
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config
config.update("jax_enable_x64", True)

def _no_hop_step(site, val):
    """Performs a step of the calculation for the reward when no hop occurs."""
    prob, state, environment_args = val
    site2 = (site + 1) % environment_args["L"]
    prob -= environment_args["p"] * state[site] * (1- state[site2])
    prob -= (1-environment_args["p"]) * (1 - state[site]) * state[site2]
    return (prob, state, environment_args)

def _no_hop_reward(state, environment_args):
    """Returns the reward when no hop occurs."""
    val = (0, state, environment_args)
    val = fori_loop(0, environment_args["L"], _no_hop_step, val)
    return jnp.log(1 + val[0]/environment_args["N"])

def _hop_reward(state, action, environment_args):
    """Returns the reward when a hop occurs."""
    bias = environment_args["s"]
    hop_reward = cond(
        state[action] == 1,
        lambda: -bias + jnp.log(environment_args["p"]/environment_args["N"]),
        lambda: bias + jnp.log((1-environment_args["p"])/environment_args["N"]))
    return hop_reward

def reward_func(state, action, environment_args):
    """Returns the reward for performing action on state."""
    reward = cond(action == environment_args["L"],
                  lambda x: _no_hop_reward(x, environment_args),
                  lambda x: _hop_reward(x, action, environment_args),
                  state)
    return reward

def _flip(state, site):
    """Flips the spin in state at site."""
    state = state.at[site].set(-(state[site]-1))
    return state

def step(state, action, environment_args):
    """Calculates new state given action for the periodic ASEP.
    
    Flips the spins at sites action and action+1, hopping a particles from one site to 
    the other.
    """
    state = cond(action == environment_args["L"],
                 lambda x: x,
                 lambda x: _flip(_flip(x, action), (action+1) % environment_args["L"]),
                 state)
    return state

def activity_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
    """Returns an updated running average of the activity."""
    activity = cond(action == environment_args["L"], lambda: 0, lambda: 1)
    observation_state += learning_rate * (activity - observation_state)
    return observation_state

def _current_obs_internal(prior_state, action, environment_args):
    """Performs minimized calculation of the current for a single hop given by action."""
    activity = cond(
        action == environment_args["L"], 
        lambda: 0, 
        lambda: cond(
            prior_state[action] == 1,
            lambda: 1,
            lambda: -1))
    return activity

def current_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
    """Returns an updated running average of the current."""
    activity = _current_obs_internal(prior_state, action, environment_args)
    observation_state += learning_rate * (activity - observation_state)
    return observation_state

def _hop_log_prob(state, action, environment_args):
    """Calculates the logged probability of a hop."""
    hop_reward = cond(
        state[action] == 1,
        lambda: jnp.log(environment_args["p"]/environment_args["L"]),
        lambda: jnp.log(environment_args["q"]/environment_args["L"]))
    return hop_reward

def kl_divergence_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
    """Updates a running average of the KL divergence for the ASEP."""
    kl_div = cond(action == environment_args["L"],
                  lambda x: _no_hop_reward(x, environment_args),
                  lambda x: _hop_log_prob(x, action, environment_args),
                  state)
    observation_state += learning_rate * (kl_div - log_prob - observation_state)
    return observation_state

def init_obs_func(activity_learning_rate, kl_div_learning_rate):
    """Creates an observation function of activity and KL divergence for the ASEP."""
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
