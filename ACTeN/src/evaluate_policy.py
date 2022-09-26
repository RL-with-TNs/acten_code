import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.lax import cond, fori_loop
from numpy import average
from jax.config import config
config.update("jax_enable_x64", True)

def default_obs(
    observation_state, step,
    prior_state, average_reward,
    log_prob, action, state, pol_grad, val_grad,
    prior_value, reward,
    environment_args
    ):
    observation_state = observation_state.at[0].set(average_reward)
    return observation_state

def init_evaluate(
    reward_func,
    environment_step,
    environment_args,
    policy,
    observation_func = default_obs
    ):
    def algorithm_step(step, val):
        (prior_state, key, policy_params,
         average_reward, observation_state) = val
        log_prob, (action, key) = policy(prior_state, key, policy_params)
        reward = reward_func(prior_state, action, environment_args) - log_prob
        state = environment_step(prior_state, action, environment_args)
        observation_state = observation_func(
            observation_state, step,
            prior_state, policy_params, average_reward,
            log_prob, action, state, reward,
            environment_args
        )
        return (state, key, policy_params,
                average_reward, observation_state)
    return algorithm_step

def _init_eval_period(period, algorithm_step):
    def eval_period(step, val):
        (state, key, policy_params,
         average_reward, observation_state, observations) = val
        internal_val = (state, key, policy_params,
            average_reward, observation_state)
        internal_val = fori_loop(0, period, algorithm_step, internal_val)
        (state, key, policy_params,
        average_reward, observation_state) = internal_val
        observations = observations.at[step,:].set(observation_state)
        return (state, key, policy_params,
                average_reward, observation_state, observations)
    return eval_period

def evaluate(policy_params, state, key, algorithm_step,
          saves, initial_observations=jnp.array([0.0]), save_period=1,
          average_reward=0.0):
    observation_state = jnp.array(initial_observations)
    observations = jnp.zeros((saves, jnp.shape(initial_observations)[0]))
    observations = observations.at[0,:].set(initial_observations)
    eval_period = _init_eval_period(save_period, algorithm_step)
    val = (state, key, policy_params,
           average_reward, observation_state, observations)
    val = fori_loop(1, saves, eval_period, val)
    (state, key, policy_params,
     average_reward, observation_state, observations) = val
    return policy_params, state, key, average_reward, observations