0x00. Q-learning
================

Learning Objectives
-------------------

-   What is a Markov Decision Process?
-   What is an environment?
-   What is an agent?
-   What is a state?
-   What is a policy function?
-   What is a value function? a state-value function? an action-value function?
-   What is a discount factor?
-   What is the Bellman equation?
-   What is epsilon greedy?
-   What is Q-learning?

Tasks
-----

### 0\. Load the Environment

Write a function `def load_frozen_lake(desc=None, map_name=None, is_slippery=False):` that loads the pre-made `FrozenLakeEnv` evnironment from OpenAI's `gym`:

-   `desc` is either `None` or a list of lists containing a custom description of the map to load for the environment
-   `map_name` is either `None` or a string containing the pre-made map to load
-   *Note: If both `desc` and `map_name` are `None`, the environment will load a randomly generated 8x8 map*
-   `is_slippery` is a boolean to determine if the ice is slippery
-   Returns: the environment

### 1\. Initialize Q-table

Write a function `def q_init(env):` that initializes the Q-table:

-   `env` is the `FrozenLakeEnv` instance
-   Returns: the Q-table as a `numpy.ndarray` of zeros

### 2\. Epsilon Greedy

Write a function `def epsilon_greedy(Q, state, epsilon):` that uses epsilon-greedy to determine the next action:

-   `Q` is a `numpy.ndarray` containing the q-table
-   `state` is the current state
-   `epsilon` is the epsilon to use for the calculation
-   You should sample `p` with `numpy.random.uniformn` to determine if your algorithm should explore or exploit
-   If exploring, you should pick the next action with `numpy.random.randint` from all possible actions
-   Returns: the next action index

### 3\. Q-learning

Write the function `def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs Q-learning:

-   `env` is the `FrozenLakeEnv` instance
-   `Q` is a `numpy.ndarray` containing the Q-table
-   `episodes` is the total number of episodes to train over
-   `max_steps` is the maximum number of steps per episode
-   `alpha` is the learning rate
-   `gamma` is the discount rate
-   `epsilon` is the initial threshold for epsilon greedy
-   `min_epsilon` is the minimum value that `epsilon` should decay to
-   `epsilon_decay` is the decay rate for updating `epsilon` between episodes
-   When the agent falls in a hole, the reward should be updated to be `-1`
-   Returns: `Q, total_rewards`
    -   `Q` is the updated Q-table
    -   `total_rewards` is a list containing the rewards per episode

### 4\. Play

Write a function `def play(env, Q, max_steps=100):` that has the trained agent play an episode:

-   `env` is the `FrozenLakeEnv` instance
-   `Q` is a `numpy.ndarray` containing the Q-table
-   `max_steps` is the maximum number of steps in the episode
-   Each state of the board should be displayed via the console
-   You should always exploit the Q-table
-   Returns: the total rewards for the episode
