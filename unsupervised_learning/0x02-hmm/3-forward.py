#!/usr/bin/env python3
"""Function forward."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm
    for a hidden markov model.

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
        - Transition[i, j] is the probability of transitioning
          from the hidden state i to j.
    - Initial a numpy.ndarray of shape (N, 1) containing the
      probability of starting in a particular hidden state.

    - Returns: P, F, or None, None on failure.
        - P is the likelihood of the observations given the model.
        - F is a numpy.ndarray of shape (N, T) containing
          the forward path probabilities.
        - F[i, j] is the probability of being in hidden state i
          at time j given the previous observations.
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

    # Obs.  (T) -> x1, ..., xT (xt E [1, 2, ..., M], t E [1, ..., T])
    # Emis. (N, M)   -> P(xt=j|zt=i) (xt E [1, ..., M], zt E [1, ..., N])
    # Trans. (N, N)  -> P(zt+1=j|zt=i) (zt E [1, ..., N])
    # Initial. (N, 1)-> z1
    # F[i, t] = P(zt=i|Obs[t]) (i E [1, ..., N], t E [1, ..., T])
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        state = F[:, i - 1].T @ Transition
        F[:, i] = state * Emission[:, Observation[i]]

    P = np.sum(F[:, -1])

    return P, F
