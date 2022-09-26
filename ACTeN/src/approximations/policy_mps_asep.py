import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.lax import cond, fori_loop, scan
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

def _step_to_right_boundary(carry, input):
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

    cond_first_site = (site_idx == 0)

    def true_func(mat, weights, environments):
        mat = jnp.matmul(mat, weights[1,state[site_idx],:,:]) #Attach the take-flip matrix
        return mat

    def false_func(mat, weights, environments):
        mat = jnp.matmul(mat, weights[0,state[site_idx],:,:]) #Attach the no-flip
        return mat

    mat = cond(cond_first_site, true_func, false_func, mat, weights, environments) #update the left-environments

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

    environments = jnp.zeros((len(state)+1,weights.shape[2],weights.shape[3]))

    mat = jnp.identity(weights.shape[2])


    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(len(state)))
    carry, outputs = scan(_step_to_right, carry, inputs) #construct the left-environments via a rightward sweep

    weights, mat, environments, state = carry

    #Add condition for no flip to be impossibe

    # cond_all_up = jnp.prod(state) #used to exclude the no-flip action case if true
    # no_flip_is_possible = cond_all_up != 1
    # no_flip_is_possible = True

    # def true_func(environments): #no flip is possible
    #     environments = environments.at[-1,:,:].set(jnp.identity(weights.shape[2]))
    #     return environments

    # def false_func(environments): #no flip is impossible. keep environment as zero.
    #     return environments

    # environments = cond(no_flip_is_possible, true_func, false_func, environments)

    environments = environments.at[-1,:,:].set(jnp.identity(weights.shape[2])) #no flip always possible for bond asep

    mat = jnp.identity(weights.shape[2])

    carry = (weights, mat, environments, state)

    inputs = jnp.array(range(len(state)-2, -1, -1)) #resolve special case for site=len(state)-1 last
    carry, outputs = scan(_step_to_left, carry, inputs)

    weights, mat, environments, state = carry

    #Finish no-flip

    environment_completed_no_flip =  jnp.matmul(weights[0,state[0],:,:], mat) #Attach the no-flip matrix to complete the no-flip environment
    environments = environments.at[-1,:,:].set(environment_completed_no_flip)

    #Simply calculate special case manually

    cond_last_site_can_flip = (state[-1] != state[0]) #the local constraint to enable flip of last state and first in PBC, requires special handling

    def true_func(environments): #flip on final site and 0 is possible
        environments = environments.at[-2,:,:].set(jnp.identity(weights.shape[2])) #-1 is no flip, so -2 is the final flip. Eg. for L = 4; a = (1,0,0,1)
        return environments

    def false_func(environments): #cannot flip final site and 0
        return environments

    environments = cond(cond_last_site_can_flip, true_func, false_func, environments) # reset environment for -2 as rightward sweep was incorrect in this case

    mat = jnp.identity(weights.shape[2])
    inputs = jnp.array(range(len(state)-1))
    carry, outputs = scan(_step_to_right_boundary, carry, inputs)
    weights, mat, _, state = carry
    environment_completed_boundary_flip =  jnp.matmul(mat, weights[1,state[-1],:,:]) #Attach the no-flip matrix to complete the no-flip environment
    environments = environments.at[-2,:,:].set(environment_completed_boundary_flip)

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

# def policy(state, key, weights):
#     chi_policy = weights.shape[-1]
#     tensor = (jnp.array([[jnp.eye(chi_policy), jnp.eye(chi_policy)],
# 							 [jnp.eye(chi_policy), jnp.eye(chi_policy)]]) + weights)
#     environments = _get_environments_east_model(state, tensor)
#     probs, _ = _get_probs(environments)
#     key, subkey = random.split(key)
#     action = random.choice(subkey, jnp.array(list(range(len(state)+1))), p = probs) #Use shape?
#     probability = probs[action]
#     log_prob = jnp.log(probability)
#     return log_prob, (action, key)

def policy(state, key, weights):
    # chi_policy = weights.shape[-1]
    # tensor = (jnp.array([[jnp.eye(chi_policy), jnp.eye(chi_policy)],
	# 						 [jnp.eye(chi_policy), jnp.eye(chi_policy)]]) + weights)
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
