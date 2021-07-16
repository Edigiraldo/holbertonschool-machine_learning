#!/usr/bin/env python3
import numpy as np
"""Monte Carlo Policy Gradient."""


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.

    - matrix is a set of observations.
    - weight is the parameters of the policy.

    Returns: Probabilities of actions (= policy).
    """
    xw = np.dot(matrix, weight)
    e = np.exp(xw)

    soft = e / e.sum()

    return soft


def policy_gradient(state, weight):
    """
    Function that computes the Monte-Carlo policy gradient based
    on a state and a weight matrix.

    - state is matrix representing the current observation of the environment.
    - weight is a matrix of random weight.

    Returns: the action and the gradient (in this order).
    """
    # applying policy
    actions_prob = policy(state, weight)  # dims -> (1, actions)
    num_actions = actions_prob.shape[1]

    action = np.random.choice(num_actions, p=actions_prob[0])

    # policy gradient -> soft(j) * (delta(i,j) - soft(i))
    grad_policy = -actions_prob[action] * actions_prob
    grad_policy[action] += 1

    # gradient(log(policy)) -> grad(policy) / prob(action)
    grad_log_policy = grad_policy / actions_prob[0, action]  # (1, actions)

    # Full gradient
    gradient = np.dot(state.T, grad_log_policy)

    return action, gradient
