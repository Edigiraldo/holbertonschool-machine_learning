#!/usr/bin/env python3
"""Function play."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode.

    - env is the FrozenLakeEnv instance.
    - Q is a numpy.ndarray containing the Q-table.
    - max_steps is the maximum number of steps in the episode.
    - Each state of the board should be displayed via the console.

    - Returns: the total rewards for the episode.
    """
    state = env.reset()
    env.render()

    done = False
    for step in range(max_steps):
        action = np.argmax(Q[state, :])

        state, reward, done, _ = env.step(action)
        env.render()

        if done is True:
            break

    return reward
