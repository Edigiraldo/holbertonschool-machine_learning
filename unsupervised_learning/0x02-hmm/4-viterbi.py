#!/usr/bin/env python3
"""Function viterbi."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function that calculates the most likely sequence of
    hidden states for a hidden markov model.

    - Observation is a numpy.ndarray of shape (T,) that contains
      the index of the observation.
        - T is the number of observations.
    - Emission is a numpy.ndarray of shape (N, M) containing the
      emission probability of a specific observation given a
      hidden state.
        - Emission[i, j] is the probability of observing j given
          the hidden state i.
        - N is the number of hidden states.
        - M is the number of all possible observations.
    - Transition is a 2D numpy.ndarray of shape (N, N) containing
      the transition probabilities.
        - Transition[i, j] is the probability of transitioning from
          the hidden state i to j.
    - Initial a numpy.ndarray of shape (N, 1) containing the
      probability of starting in a particular hidden state.

    Returns: path, P, or None, None on failure.
    - path is the a list of length T containing the most likely
      sequence of hidden states.
    - P is the probability of obtaining the path sequence.
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    T = Observation.shape[0]

    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    N, M = Emission.shape

    if (type(Transition) is not np.ndarray or Transition.ndim != 2 or
       Transition.shape[0] != Transition.shape[1] or
       Transition.shape[0] != N):
        return None, None

    if (type(Initial) is not np.ndarray or Initial.ndim != 2 or
       Initial.shape != (N, 1)):
        return None, None

    # mu[j, k] = max_(X_0:k){P[X_0:k, Y_0:K]} for x_k = j k E [0, T]
    mu = np.zeros((N, T))
    prev_state = np.zeros((N, T))

    for j in range(0, N):
        mu[j, 0] = Initial[j] * Emission[j, Observation[0]]

    for k in range(1, T):
        for j in range(0, N):
            mu[j, k] = -1
            for l in range(0, N):
                val = (mu[l, k - 1] * Transition[l, j] *
                       Emission[j, Observation[k]])
                if val > mu[j, k]:
                    mu[j, k] = val
                    prev_state[j, k] = l

    path = [0] * T
    path[T - 1] = np.argmax(mu[:, T - 1])
    for k in range(T - 2, -1, -1):
        path[k] = int(prev_state[int(path[k + 1]), k + 1])

    P = mu[int(path[T - 1]), T - 1]

    return path, P
