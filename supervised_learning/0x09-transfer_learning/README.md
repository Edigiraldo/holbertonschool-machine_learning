0x09. Transfer Learning
=======================

Learning Objectives
-------------------

### General

-   What is a transfer learning?
-   What is fine-tuning?
-   What is a frozen layer? How and why do you freeze a layer?
-   How to use transfer learning with Keras applications

Tasks
-----

#### 0\. Transfer Knowledge

Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

-   Your script must save your trained model in the current working directory as `cifar10.h5`
-   Your saved model should be compiled
-   Your saved model should have a validation accuracy of 87% or higher
-   Your script should not run when the file is imported
-   **Hint1:** *The training and tweaking of hyperparameters may take a while so start early!*
-   **Hint2:** *The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct size*
-   **Hint3:** *You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.*

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

-   `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
-   `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
-   Returns: `X_p, Y_p`
    -   `X_p` is a `numpy.ndarray` containing the preprocessed `X`
    -   `Y_p` is a `numpy.ndarray` containing the preprocessed `Y`
