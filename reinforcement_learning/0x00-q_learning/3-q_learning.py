#!/usr/bin/env python3
"""Function train."""
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


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning.

    - env is the FrozenLakeEnv instance.
    - Q is a numpy.ndarray containing the Q-table.
    - episodes is the total number of episodes to train over.
    - max_steps is the maximum number of steps per episode.
    - alpha is the learning rate.
    - gamma is the discount rate.
    - epsilon is the initial threshold for epsilon greedy.
    - min_epsilon is the minimum value that epsilon should decay to.
    - epsilon_decay is the decay rate for updating epsilon between episodes.

    Returns: Q, total_rewards.
        - Q is the updated Q-table.
        - total_rewards is a list containing the rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        reward_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done:  # Falling in a hole.
                reward = -1

            Q[state, action] = (Q[state, action] * (1 - alpha) + alpha *
                                (reward + gamma * np.max(Q[new_state, :])))

            state = new_state
            reward_episode += reward

            if done is True:
                break

        epsilon = (min_epsilon +
                   (1 - min_epsilon) * np.exp(-epsilon_decay * episode))
        total_rewards.append(reward_episode)

    return Q, total_rewards
