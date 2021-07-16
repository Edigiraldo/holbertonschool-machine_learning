#!/usr/bin/env python3
"""Function train."""
import gym
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Function to train an agent applying monte carlo policy gradient.

    - env is the initial environment
    - nb_episodes is the number of episodes used for training
    - alpha is the learning rate
    - gamma is the discount factor

    Returns: all values of the score (sum of all rewards during
    one episode loop).
    """
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_weights = np.random.rand(n_obs, n_actions)

    scores = []
    for episode in range(nb_episodes * 10):
        state = env.reset()[None,:]
        gradients = []
        rewards = []
        score = 0

        # Run an episode
        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, gradient = policy_gradient(state, policy_weights)
            state, reward, done, _ = env.step(action)
            state = state[None, :]

            gradients.append(gradient)
            rewards.append(reward)
            score += reward

        scores.append(score)

        # Policy update
        num_steps = len(gradients)
        discount_factor = gamma ** np.arange(num_steps)
        for i in range(num_steps):
            rews_after_step = rewards[i:]
            discount_factors = discount_factor[:len(rews_after_step)]
            disc_reward = np.sum(rews_after_step * discount_factors)

            policy_weights += alpha * gradients[i] * disc_reward
        print("{}: {}".format(episode, score), end="\r", flush=False)

    return scores
