#!/usr/bin/env python3
"""Script to train an agent to play atari breakout with deep Q-learning."""
import numpy as np
import gym

from PIL import Image

import keras as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# number of context images for the model to evaluate current state.
window_length = 4
# size of resized images for keras model.
input_shape = (84, 84)

def create_model(window_length, input_shape):
    """
    Function to create model to be used as policy.
    """
    model_inp = (window_length,) + input_shape

    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=model_inp))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    return model


class AtariProcessor(Processor):
    """
    Class that acts as a coupling mechanism between an `Agent` and its `Env`.
    This class processes input images to be fed to a model and clips
    the gradient.
    """
    def process_observation(self, observation):
        """
        Method to process images, resizing to 84x84 and
        converting from RGB to grayscale images.

        observation: RGB image with shape (height, width, channels).
        """
        img = Image.fromarray(observation)
        # resize and convert to gray scale.
        img = img.resize(input_shape).convert('L')

        # convert to np.array and store pixels in uint8 to save space.
        processed_img = np.array(img).astype('uint8')

        return processed_img

    def process_state_batch(self, batch):
        """
        Method to rescale image pixels to be between 0 and 1.
        """
        batch = batch.astype('float32') / 255.0

        return batch

    def process_reward(self, reward):
        """
        Method to perform gradient clipping and avoid exploding gradient.

        reward: reward obtaing by model after an action.
        """

        return np.clip(reward, -1., 1.)


env = gym.make('BreakoutDeterministic-v4')
nb_actions = env.action_space.n


model = K.models.load_model('policy.h5')
# define policy to handle exploitation vs exploration trade-off.
policy = GreedyQPolicy()

# manage memory to store batches of sequences of images for training.
memory = SequentialMemory(limit=1000000, window_length=window_length)
# process images: resizes and converts them from RGB format to grayscale.

processor = AtariProcessor()

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.99,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


# Finally, evaluate our algorithm for 10 episodes.
dqn.test(env, nb_episodes=10, visualize=True)
