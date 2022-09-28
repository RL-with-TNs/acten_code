import jax.numpy as jnp
from jax import random
from jax.lax import cond, scan
from jax.config import config
config.update("jax_enable_x64", True)

def policy(state, key, weights):
    """Sample the L+1 actions of the east model

    Args:
        state (_type_): the vector s = (s_1, s_2, ..., s_L) of s_i \in {0,1}
        key : the rng key
        weights (array): the weights of the policy, a single tensor of shape (d, d, chi, chi)

    Returns:
        log_prob, (action, key): the log_prob of the sampled action along with the action and key as auxilary returns.
    """
    tensor = weights
    environments = _get_environments_east_model(state, tensor)
    probs, _ = _get_probs(environments)
    key, subkey = random.split(key)
    action = random.choice(subkey, jnp.array(list(range(len(state)+1))), p = probs)
    probability = probs[action]
    log_prob = jnp.log(probability)
    return log_prob, (action, key)

def _step_to_right(carry, input):
    """performs a step to the right for the east-model. If the state to the left is up, the current site can flip.
    """

    weights, mat, environments, state = carry
    site_idx = input

    cond_state_to_left_is_up = state[(site_idx-1)%len(state)] == 1

    def true_func(mat, weights, environments):
        mat_flip = jnp.matmul(mat, weights[1,state[site_idx],:,:]) #Attach the take-flip matrix
        environments = environments.at[site_idx,:,:].set(mat_flip) #this is the left_environment inclusive of the current site
        return environments

    def false_func(mat, weights, environments):
        return environments

    environments = cond(cond_state_to_left_is_up, true_func, false_func, mat, weights, environments)

    mat = jnp.matmul(mat, weights[0,state[site_idx],:,:]) #Attach the no-flip matrix to continue

    output = None

    return (weights, mat, environments, state), output

def _step_to_left(carry, input):

    weights, mat, environments, state = carry

    site_idx = input

    mat = jnp.matmul(weights[0,state[site_idx],:,:], mat) #Attach the no-flip matrix to complete the right environment of site_idx - 1

    completed_env = jnp.matmul(environments[(site_idx - 1)%(len(state)+1),:,:], mat) #complete the environment for the site to the left. If this was zero it will remain so

    environments = environments.at[(site_idx - 1)%(len(state)+1),:,:].set(completed_env) #-1 should be placed at the no-flip entry

    output = None

    return (weights, mat, environments, state), output

def _get_environments_east_model(state, weights):
    """form the environments for an action-state MPS subject to the east model dynamics (left facilitation and max single flip). Impossible actions are represented by a zero-matrix environment.
    """

    environments = jnp.zeros((len(state)+1,weights.shape[2],weights.shape[3]))

    mat = jnp.identity(weights.shape[2])

    cond_all_up = jnp.prod(state) #used to exclude the no-flip action case if true

    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(len(state)))
    carry, outputs = scan(_step_to_right, carry, inputs)

    weights, mat, environments, state = carry

    def true_func(environments):
        environments = environments.at[-1,:,:].set(jnp.identity(weights.shape[2]))
        return environments

    def false_func(environments):
        return environments

    environments = cond(cond_all_up != 1, true_func, false_func, environments)

    mat = jnp.identity(weights.shape[2])

    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(len(state)-1, -1, -1))
    carry, outputs = scan(_step_to_left, carry, inputs)

    weights, mat, environments, state = carry

    return environments

def _get_probs(environments):
    factors = jnp.trace(environments, axis1=1, axis2=2)
    factors_squared = jnp.square(factors)
    probs = factors_squared/jnp.sum(factors_squared)
    return probs, factors