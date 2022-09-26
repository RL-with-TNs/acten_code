import jax.numpy as jnp
from jax.lax import cond
from jax.config import config
config.update("jax_enable_x64", True)

def average_reward_observation(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
        return average_reward

def store_current_policy(
        observation_state, step, learning_rate,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args):
        return jnp.ravel(policy_params)

def observation_vec_to_func(observation_vec, learning_rates):
    def obs_func(
        observation_state, step,
        prior_state, policy_params, value_params, average_reward,
        log_prob, action, state, pol_grad, val_grad,
        prior_value, current_value, reward, td_error,
        environment_args
        ):
        i = 0
        for observation in observation_vec:
            observation_state = observation_state.at[i].set(
                observation(
            observation_state[i], step, learning_rates[i],
            prior_state, policy_params, value_params, average_reward,
            log_prob, action, state, pol_grad, val_grad,
            prior_value, current_value, reward, td_error,
            environment_args))
            i += 1
        return observation_state
    return obs_func
