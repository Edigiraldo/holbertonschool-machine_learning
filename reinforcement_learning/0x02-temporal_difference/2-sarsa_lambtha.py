#!/usr/bin/env python3
"""Function sarsa_lambtha."""
import numpy as np
import gym


def greedy_policy_action(epsilon, state, Q):
    """
    Function to manage explotation exploration trade-off.

    - epsilon is a value between threshold to determine how much to explore.
    - state is the current state.
    - Q is a matrix with expected rewards.

    Returns: Next action to take.
    """
    num_actions = Q.shape[1]
    r = np.random.uniform(0.1)

    if r >= epsilon:
        return Q[state].argmax()
    else:
        return np.random.randint(0, num_actions)


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs SARSA(λ).

    - env is the openAI environment instance.
    - Q is a numpy.ndarray of shape (s,a) containing the Q table.
    - lambtha is the eligibility trace factor.
    - episodes is the total number of episodes to train over.
    - max_steps is the maximum number of steps per episode.
    - alpha is the learning rate.
    - gamma is the discount rate.
    - epsilon is the initial threshold for epsilon greedy.
    - min_epsilon is the minimum value that epsilon should decay to.
    - epsilon_decay is the decay rate for updating epsilon between episodes.

    Returns: Q, the updated Q table.
    """
    epsilon_0 = epsilon
    ET = np.zeros(Q.shape)
    for episode in range(episodes):
        last_state = env.reset()
        last_action = greedy_policy_action(epsilon, last_state, Q)
        for _ in range(max_steps):
            state, reward, finished, _ = env.step(last_action)

            action = greedy_policy_action(epsilon, last_state, Q)

            ET *= gamma * lambtha
            ET[last_state, last_action] = 1

            delta = (reward +
                     gamma * Q[state, action] - Q[last_state, last_action])

            Q[last_state, last_action] = (Q[last_state, last_action] +
                                          alpha * delta *
                                          ET[last_state, last_action])

            last_state = state
            last_action = action

            if finished:
                break

        epsilon = (min_epsilon + (epsilon_0 - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
