#!/usr/bin/env/python3
"""Function epsilon_greedy."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action.

    - Q is a numpy.ndarray containing the q-table.
    - state is the current state.
    - epsilon is the epsilon to use for the calculation.

    Returns: the next action index.
    """
    n_actions = Q.shape[1]

    random = np.random.uniform(0, 1)
    if epsilon >= random:  # explore.
        action_idx = np.random.randint(0, n_actions)
    else:  # exploit.
        action_idx = Q.argmax(axis=1)[state]

    return action_idx
