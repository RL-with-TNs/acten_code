import jax.numpy as jnp
from jax import grad, jit, value_and_grad
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
    """Default observation function, simply returns the current average reward."""
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
    """Used to create a closure which performs a single step of an Actor-Critic algorithm.
    
    Wraps the learning rates, approximation functions, environment functions and 
    observation function.
    
    arguments:
        reward_func:
            the reward function of the environment
        environment_step:
            function implementing how the environments updates the state given the 
            current state and the action selected
        environment_args:
            contains any parameters used by the environment functions
        value:
            the value function approximation, takes the state as its first argument and 
            the parameters as its second. Must return a single float or single 
            element float array. Must be differentiable with respect to second argument.
        value_lr:
            the learning rate for updates of the value approximation parameters
        policy:
            the policy function approximation, takes the state, PRNG key and parameters 
            as its three arguments in that order. Must return the following signature:

                (log_probability_of_selected_action, (action_selected, new_prng_key))

            First return must be differentiable with respect to third argument
        policy_lr:
            the learning rate for updates of the policy approximation parameters
        reward_lr:
            the learning rate for the average_reward
        observation_func: 
            the function used to calculate observations. Takes all variables present at 
            the end of an algorithm step to update the observation state, see source for 
            details.
    return:
        function which takes two arguments and returns the new algorithm state
            arg_1: 
                the current step of the training
            arg_2: 
                the current algorithm state
            return:
                the new algorithm state
    """
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
    """Used to create a closure which performs training between two observation saves."""
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
    """Trains the approximations.
    
    Note this is not general, it is tailored to the single-step, sequential Actor-Critic
    algorithm constructed using the init_actor_critic function above.

    arguments:
        state: 
            the initial state of the system
        key: 
            the key used to generate random numbers
        value_params: 
            parameters for the value function approximation
        policy_params: 
            parameters for the policy approximations
        algorithm_step: 
            a function performing a single step of the training algorithm
        saves: 
            how many times observations are saved
        initial_observations: 
            an array used to initialize the observation state and buffer
        save_period: 
            how many training steps are performed between observation saves
        average_reward: 
            the initial value used for the average_reward
    
    returns:
        state: 
            the final state of the system after training
        key: 
            the final key after use for random number generation
        value_params: 
            the trained value approximation parameters
        policy_params: 
            the train policy approximation parameters
        average_reward: 
            the estimate of the average reward at the end of training
        observations: 
            an array of observations saved throughout training

    To JIT compile, arguments 4, 5 and 7 must be static.
    """
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