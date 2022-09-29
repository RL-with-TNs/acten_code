# ACTeN, Actor Critic with Tensor Networks
This repository contains some of the code used in producing results for the paper introducing ACTeN (https://arxiv.org/abs/2209.14089).

Specifically:
* exact_diagonalization contains code used to produce the exact results for the ASEP model.
* dmrg contains code used to produce the values for the East model with which we compare our approach.
* ACTeN contains the code used for the ACTeN training runs.
  * src/environments contains the code used to describe the two problems, the activity statistics of the East model and the current statistics of the ASEP model, in the usual MDP format of an RL problem.
  * src/approximations contains the TN approximations used for the value functions and policies, with specialized policy approximations for the two different constraints present in the two models.
  * src contains the actor critic code, along with additional code for storing observations throughout the training process and evaluating the policy after training is complete.

We have not provided code for the parallel training and annealing that we used to on top of this ACTeN code. For details of this please see the paper.
