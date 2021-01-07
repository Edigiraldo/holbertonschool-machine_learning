0x02. Tensorflow
================

Learning Objectives
-------------------

### General

-   What is tensorflow?
-   What is a session? graph?
-   What are tensors?
-   What are variables? constants? placeholders? How do you use them?
-   What are operations? How do you use them?
-   What are namespaces? How do you use them?
-   How to train a neural network in tensorflow
-   What is a checkpoint?
-   How to save/load a model with tensorflow
-   What is the graph collection?
-   How to add and get variables from the collection


Tasks
-----

#### 0\. Placeholders

Write the function `def create_placeholders(nx, classes):` that returns two placeholders, `x` and `y`, for the neural network:

-   `nx`: the number of feature columns in our data
-   `classes`: the number of classes in our classifier
-   Returns: placeholders named `x` and `y`, respectively
    -   `x` is the placeholder for the input data to the neural network
    -   `y` is the placeholder for the one-hot labels for the input data

#### 1\. Layers

Write the function `def create_layer(prev, n, activation):`

-   `prev` is the tensor output of the previous layer
-   `n` is the number of nodes in the layer to create
-   `activation` is the activation function that the layer should use
-   use `tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")` to implement `He et. al` initialization for the layer weights
-   each layer should be given the name `layer`
-   Returns: the tensor output of the layer

#### 2\. Forward Propagation

Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network:

-   `x` is the placeholder for the input data
-   `layer_sizes` is a list containing the number of nodes in each layer of the network
-   `activations` is a list containing the activation functions for each layer of the network
-   Returns: the prediction of the network in tensor form
-   For this function, you should import your `create_layer` function with `create_layer = __import__('1-create_layer').create_layer`

#### 3\. Accuracy

Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction:

-   `y` is a placeholder for the labels of the input data
-   `y_pred` is a tensor containing the network's predictions
-   Returns: a tensor containing the decimal accuracy of the prediction

#### 4\. Loss

Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction:

-   `y` is a placeholder for the labels of the input data
-   `y_pred` is a tensor containing the network's predictions
-   Returns: a tensor containing the loss of the prediction

#### 5\. Train_Op

Write the function `def create_train_op(loss, alpha):` that creates the training operation for the network:

-   `loss` is the loss of the network's prediction
-   `alpha` is the learning rate
-   Returns: an operation that trains the network using gradient descent

#### 6\. Train

Write the function `def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):` that builds, trains, and saves a neural network classifier:

-   `X_train` is a `numpy.ndarray` containing the training input data
-   `Y_train` is a `numpy.ndarray` containing the training labels
-   `X_valid` is a `numpy.ndarray` containing the validation input data
-   `Y_valid` is a `numpy.ndarray` containing the validation labels
-   `layer_sizes` is a list containing the number of nodes in each layer of the network
-   `activations` is a list containing the activation functions for each layer of the network
-   `alpha` is the learning rate
-   `iterations` is the number of iterations to train over
-   `save_path` designates where to save the model
-   Add the following to the graph's collection
    -   placeholders `x` and `y`
    -   tensors `y_pred`, `loss`, and `accuracy`
    -   operation `train_op`
-   After every 100 iterations, the 0th iteration, and `iterations` iterations, print the following:
    -   `After {i} iterations:` where i is the iteration
    -   `\tTraining Cost: {cost}` where `{cost}` is the training cost
    -   `\tTraining Accuracy: {accuracy}` where `{accuracy}` is the training accuracy
    -   `\tValidation Cost: {cost}` where `{cost}` is the validation cost
    -   `\tValidation Accuracy: {accuracy}` where `{accuracy}` is the validation accuracy
-   *Reminder: the 0th iteration represents the model before any training has occurred*
-   After training has completed, save the model to `save_path`
-   You may use the following imports:
    -   `calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy`
    -   `calculate_loss = __import__('4-calculate_loss').calculate_loss`
    -   `create_placeholders = __import__('0-create_placeholders').create_placeholders`
    -   `create_train_op = __import__('5-create_train_op').create_train_op`
    -   `forward_prop = __import__('2-forward_prop').forward_prop`
-   You are not allowed to use `tf.saved_model`
-   Returns: the path where the model was saved

#### 7\. Evaluate

Write the function `def evaluate(X, Y, save_path):` that evaluates the output of a neural network:

-   `X` is a `numpy.ndarray` containing the input data to evaluate
-   `Y` is a `numpy.ndarray` containing the one-hot labels for `X`
-   `save_path` is the location to load the model from
-   You are not allowed to use `tf.saved_model`
-   Returns: the network's prediction, accuracy, and loss, respectively

