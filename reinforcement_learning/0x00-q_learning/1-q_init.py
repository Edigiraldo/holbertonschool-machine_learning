#!/usr/bin/env/python3
"""Function q_init."""
import gym
import numpy as np


def q_init(env):
    """
    Function that initializes the Q-table.

    - env is the FrozenLakeEnv instance.

    - Returns: the Q-table as a numpy.ndarray of zeros.
    """
    n_actions = env.action_space.n
    n_states = env.observation_space.n

    Q_table = np.zeros((n_states, n_actions))

    return Q_table
