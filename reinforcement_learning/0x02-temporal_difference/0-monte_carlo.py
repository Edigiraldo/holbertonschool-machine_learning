#!/usr/bin/env python3
"""Function monte carlo."""
import numpy as np
import gym


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Function that performs the Monte Carlo algorithm.

    - env is the openAI environment instance.
    - V is a numpy.ndarray of shape (s,) containing the value estimate.
    - policy is a function that takes in a state and returns the next action
      to take.
    - episodes is the total number of episodes to train over.
    - max_steps is the maximum number of steps per episode.
    - alpha is the learning rate.
    - gamma is the discount rate.

    Returns: V, the updated value estimate.
    """
    env.reset()
    space_n = env.observation_space.n
    discount_factor = gamma ** np.arange(max_steps)

    for _ in range(episodes):
        states = []  # State at ith timestep.
        rewards = []  # Reward got passing from states[i] state to next state.

        init_state = env.reset()
        states.append(init_state)
        for _ in range(max_steps):
            action = policy(states[-1])
            state, reward, finished, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            if finished:
                break
        for idx in range(len(states)):
            state = states[idx]
            # If last state (There is no reward from here).
            if idx == len(rewards):
                break
            rews_from_state = np.array(rewards)
            num_rews = rews_from_state.shape[0]
            disc_facts = discount_factor[:num_rews]
            disc_reward = (rews_from_state * disc_facts).sum()

            V[state] = V[state] + alpha * (disc_reward - V[state])

    return V
