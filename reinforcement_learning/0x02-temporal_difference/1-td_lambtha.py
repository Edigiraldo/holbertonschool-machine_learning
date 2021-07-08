#!/usr/bin/env python3
"""Function td_lambtha."""
import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Function that performs the TD(λ) algorithm.

    - env is the openAI environment instance.
    - V is a numpy.ndarray of shape (s,) containing the value estimate.
    - policy is a function that takes in a state and returns the next
      action to take.
    - lambtha is the eligibility trace factor.
    - episodes is the total number of episodes to train over.
    - max_steps is the maximum number of steps per episode.
    - alpha is the learning rate.
    - gamma is the discount rate.

    Returns: V, the updated value estimate.
    """
    ET = np.zeros(V.shape)
    for _ in range(episodes):
        last_state = env.reset()
        for _ in range(max_steps):
            action = policy(last_state)
            state, reward, finished, _ = env.step(action)

            ET *= gamma * lambtha
            ET[last_state] = 1
            delta = reward + gamma * V[state] - V[last_state]

            V[last_state] = V[last_state] + alpha * delta * ET[last_state]

            last_state = state

            if finished:
                break

    return V
