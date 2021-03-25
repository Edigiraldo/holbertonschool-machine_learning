0x07. Bayesian Probability
==========================

Learning Objectives
-------------------

-   What is Bayesian Probability?
-   What is Bayes' rule and how do you use it?
-   What is a base rate?
-   What is a prior?
-   What is a posterior?
-   What is a likelihood?

Tasks
-----

### 0\. Likelihood

You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and `x` patients develop severe side effects. You can assume that `x` follows a binomial distribution.

Write a function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

-   `x` is the number of patients that develop severe side effects
-   `n` is the total number of patients observed
-   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
-   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
-   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
-   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
-   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
-   If any value in `P` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in P must be in the range [0, 1]`
-   Returns: a 1D `numpy.ndarray` containing the likelihood of obtaining the data, `x` and `n`, for each probability in `P`, respectively

### 1\. Intersection

Based on `0-likelihood.py`, write a function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:

-   `x` is the number of patients that develop severe side effects
-   `n` is the total number of patients observed
-   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
-   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
-   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
-   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
-   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
-   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
-   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
-   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
-   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1` **Hint: use [numpy.isclose](https://intranet.hbtn.io/rltoken/7pptg2vy0_-c0qQ9MnZu1w "numpy.isclose")**
-   All exceptions should be raised in the above order
-   Returns: a 1D `numpy.ndarray` containing the intersection of obtaining `x` and `n` with each probability in `P`, respectively

### 2\. Marginal Probability

Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data:

-   `x` is the number of patients that develop severe side effects
-   `n` is the total number of patients observed
-   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of patients developing severe side effects
-   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs about `P`
-   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
-   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
-   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
-   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
-   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
-   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
-   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
-   All exceptions should be raised in the above order
-   Returns: the marginal probability of obtaining `x` and `n`

### 3\. Posterior

Based on `2-marginal.py`, write a function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:

-   `x` is the number of patients that develop severe side effects
-   `n` is the total number of patients observed
-   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
-   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
-   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
-   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
-   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
-   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
-   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
-   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
-   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
-   All exceptions should be raised in the above order
-   Returns: the posterior probability of each probability in `P` given `x` and `n`, respectively
