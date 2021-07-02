0x01. Deep Q-learning
=====================

Tasks
-----

### 0\. Breakout

Write a python script `train.py` that utilizes `keras`, `keras-rl`, and `gym` to train an agent that can play Atari's Breakout:

-   Your script should utilize `keras-rl`'s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`
-   Your script should save the final policy network as `policy.h5`

Write a python script `play.py` that can display a game played by the agent trained by `train.py`:

-   Your script should load the policy network saved in `policy.h5`
-   Your agent should use the `GreedyQPolicy`


Executed with
-------------

-    python 3.6.13
-    tensorflow 1.14.0
-    keras 2.2.4
-    keras-rl 0.4.2
-    numpy 1.19.5
-    gym 0.18.3
-    h5py 2.10.0
