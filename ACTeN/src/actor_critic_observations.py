import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.lax import cond, fori_loop
from numpy import average
from jax.config import config
config.update("jax_enable_x64", True)

def default_obs(
    observation_state, step,
    prior_state, policy_params, value_params, average_reward,
    log_prob, action, state, pol_grad, val_grad,
    prior_value, current_value, reward, td_error,
    environment_args
    ):
    observation_state = observation_state.at[0].set(average_reward)
    return observation_state

def init_actor_critic(
    reward_func,
    environment_step,
    environment_args,
    value,
    value_lr,
    policy,
    policy_lr,
    reward_lr,
    observation_func = default_obs
    ):
    policy_grad = value_and_grad(policy, argnums=2, has_aux=True)
    value_grad = value_and_grad(value, argnums=1)
    def algorithm_step(step, val):
        (prior_state, key, policy_params, value_params,
         average_reward, observation_state) = val
        prior_value, val_grad = value_grad(prior_state, value_params)
        (log_prob, (action, key)), pol_grad = policy_grad(prior_state, key, policy_params)
        reward = reward_func(prior_state, action, environment_args) - log_prob
        state = environment_step(prior_state, action, environment_args)
        current_value = value(state, value_params)
        td_error = current_value + reward - average_reward - prior_value
        policy_params += policy_lr * td_error * pol_grad
        value_params += value_lr * td_error * val_grad
        average_reward += reward_lr * td_error
        observation_state = observation_func(
            observation_state, step,
            prior_state, policy_params, value_params, average_reward,
            log_prob, action, state, pol_grad, val_grad,
            prior_value, current_value, reward, td_error,
            environment_args
        )
        return (state, key, policy_params, value_params,
                average_reward, observation_state)
    return algorithm_step

def _init_train_period(period, algorithm_step):
    def train_period(step, val):
        (state, key, policy_params, value_params,
         average_reward, observation_state, observations) = val
        internal_val = (state, key, policy_params, value_params,
            average_reward, observation_state)
        internal_val = fori_loop(0, period, algorithm_step, internal_val)
        (state, key, policy_params, value_params,
        average_reward, observation_state) = internal_val
        observations = observations.at[step,:].set(observation_state)
        return (state, key, policy_params, value_params,
                average_reward, observation_state, observations)
    return train_period

def train(state, key, value_params, policy_params, algorithm_step,
          saves, initial_observations=jnp.array([0.0]), save_period=1,
          average_reward=0.0):
    observation_state = jnp.array(initial_observations)
    observations = jnp.zeros((saves, jnp.shape(initial_observations)[0]))
    observations = observations.at[0,:].set(initial_observations)
    train_period = _init_train_period(save_period, algorithm_step)
    val = (state, key, policy_params, value_params,
           average_reward, observation_state, observations)
    val = fori_loop(1, saves, train_period, val)
    (state, key, policy_params, value_params,
     average_reward, observation_state, observations) = val
    return state, key, value_params, policy_params, average_reward, observations