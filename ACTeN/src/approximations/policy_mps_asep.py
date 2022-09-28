import jax.numpy as jnp
from jax import random
from jax.lax import cond, scan
from jax.config import config
config.update("jax_enable_x64", True)

def _step_to_right(carry, input):
    """performs a step to the right for the asep. If the current state and the next one are in 01 or 10, can flip both.
    If the local constraint condition is true, then the left-environment for the site is created and stored.
    Until reach a site where the constraint is satisfied, the environment just consists of no-flips.

    Args:
        carry (_type_): _description_
        input (_type_): _description_

    Returns:
        _type_: _description_
    """

    weights, mat, environments, state = carry
    site_idx = input

    cond_one_particle_in_pair = (state[(site_idx)%len(state)] != state[(site_idx+1)%len(state)]) #the local constraint to enable flips

    def true_func(mat, weights, environments):
        mat_flip = jnp.matmul(mat, weights[1,state[site_idx],:,:]) #Attach the take-flip matrix
        environments = environments.at[site_idx,:,:].set(mat_flip) #this is the left_environment inclusive of the current site
        return environments

    def false_func(mat, weights, environments):
        return environments #no updated to the environment without satisfying the constraint

    environments = cond(cond_one_particle_in_pair, true_func, false_func, mat, weights, environments) #update the left-environments

    mat = jnp.matmul(mat, weights[0,state[site_idx],:,:]) #Attach the no-flip matrix to continue

    output = None

    return (weights, mat, environments, state), output

def _add_zero_mat_step_to_right(carry, input):
    """multiples on the zero-mat, A0, for the current site_idx (the input).
        Used in a scan the final output mat will be a product of all zeros for the chosen inputs.
    Args:
        carry (_type_): _description_
        input (_type_): _description_

    Returns:
        _type_: _description_
    """

    weights, mat, environments, state = carry
    site_idx = input

    mat = jnp.matmul(mat, weights[0,state[site_idx],:,:]) #Attach the no-flip matrix to continue

    output = None

    return (weights, mat, environments, state), output

def _step_to_left(carry, input):

    weights, mat, environments, state = carry

    site_idx = input

    mat_flip = jnp.matmul(weights[1,state[(site_idx+1)%len(state)],:,:], mat) #Attach the flip matrix to complete the right-environment at site_idx - 1
    completed_env = jnp.matmul(environments[site_idx,:,:], mat_flip) #complete the environment for the site to the left. If this was zero it will remain so.
    environments = environments.at[site_idx,:,:].set(completed_env) #-1 should be placed at the no-flip entry

    mat = jnp.matmul(weights[0,state[(site_idx+1)%len(state)],:,:], mat) #Attach the no-flip matrix to continue the right environments

    output = None

    return (weights, mat, environments, state), output

def _get_environments_asep(state, weights):
    """form the environments for an action-state MPS subject to the asep dynamics (left/right particle hops).
    Impossible actions are represented by a zero-matrix environment.

    Args:
        state (_type_): _description_
        weights (_type_): _description_

    Returns:
        _type_: _description_
    """

    def cond_no_flip_possible(state):
        """The condition for the no-flip action to be possible.
        Here, no flip can occur unless no particle has a neighbour"""
        return 1. - jnp.allclose(jnp.sum(state*jnp.roll(state, 1)), 0)

    ## Sweep to right
    environments = jnp.zeros((len(state)+1,weights.shape[2],weights.shape[3]))

    mat = jnp.identity(weights.shape[2])

    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(1,len(state))) #skip the first site at the action-matrix here depends on the final site
    carry, outputs = scan(_step_to_right, carry, inputs) #construct the left-environments via a rightward sweep

    weights, mat, environments, state = carry ## mat from the rightward sweep is the no-flip factor but missing the first (site=0) matrix

    ## If No Flip is possible, set environment != 0

    def no_flip_possible_true_func(environments):
        environments = environments.at[-1,:,:].set(jnp.matmul(weights[0,state[0],:,:],mat)) #add the first no-flip matrix to complete env.
        return environments

    def no_flip_possible_false_func(environments):
        environments = environments.at[-1,:,:].set(jnp.zeros(environments.shape[1::])) #add the first no-flip matrix to complete env.
        return environments

    environments = cond(cond_no_flip_possible(state), no_flip_possible_true_func, no_flip_possible_false_func, environments)

    ## Sweep to left
    mat = jnp.identity(weights.shape[2])

    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(len(state)-1, 0, -1)) #Exclude site=0 as need explicit construction (was skipped in right-sweep)
    carry, outputs = scan(_step_to_left, carry, inputs)

    weights, mat, environments, state = carry

    ## Site-0 prob still needs computing: perform manually

    cond_zero_site_can_flip = (state[0] != state[1]) #the local constraint to enable flip of last state and first in PBC, requires special handling

    def true_func(environments): #flip on final site and 0 is possible
        environments = environments.at[0,:,:].set(jnp.identity(weights.shape[2]))
        return environments

    def false_func(environments): #cannot flip 0 and 1
        return environments

    environments = cond(cond_zero_site_can_flip, true_func, false_func, environments) #

    mat = jnp.identity(weights.shape[2])
    inputs = jnp.array(range(2, len(state))) #for these inputs the final mat with have zeros for all sites except the left and right boundaries
    carry, outputs = scan(_add_zero_mat_step_to_right, carry, inputs)
    weights, mat, _, state = carry #mat is the product of 0s in the bulk
    flip_mat_01 = jnp.matmul(weights[1,state[0],:,:], weights[1,state[1],:,:])
    completed_01_flip_environment = jnp.matmul(flip_mat_01,mat)
    completed_01_environment = jnp.matmul(environments[0,:,:], completed_01_flip_environment)
    environments = environments.at[0,:,:].set(completed_01_environment)

    return environments

def _get_probs(environments):
    factors = jnp.trace(environments, axis1=1, axis2=2)
    factors_squared = jnp.square(factors)
    probs = factors_squared/jnp.sum(factors_squared)
    return probs, factors

def policy_sample(state, key, weights):
    environments = _get_environments_asep(state, weights)
    probs, factors = _get_probs(environments)
    key, subkey = random.split(key)
    action = random.choice(subkey, jnp.array(list(range(len(state)+1))), p = probs) #Use shape?
    probability = probs[action]
    return action, key, probability

def policy(state, key, weights):
    tensor = weights
    environments = _get_environments_asep(state, tensor)
    probs, _ = _get_probs(environments)
    key, subkey = random.split(key)
    action = random.choice(subkey, jnp.array(list(range(len(state)+1))), p = probs)
    probability = probs[action]
    log_prob = jnp.log(probability)
    return log_prob, (action, key)

def policy_log_prob(state, action, weights):
    environments = _get_environments_asep(state, weights)
    probs, factors = _get_probs(environments)
    probability = probs[action]
    log_prob = jnp.log(probability)
    return log_prob

def policy_probs(state, weights):
    environments = _get_environments_asep(state, weights)
    probs, _ = _get_probs(environments)
    return probs