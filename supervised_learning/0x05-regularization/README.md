0x05. Regularization
====================

Learning Objectives
-------------------

### General

-   What is regularization? What is its purpose?
-   What is are L1 and L2 regularization? What is the difference between the two methods?
-   What is dropout?
-   What is early stopping?
-   What is data augmentation?
-   How do you implement the above regularization methods in Numpy? Tensorflow?
-   What are the pros and cons of the above regularization methods?

Tasks
-----

#### 0\. L2 Regularization Cost

Write a function `def l2_reg_cost(cost, lambtha, weights, L, m):` that calculates the cost of a neural network with L2 regularization:

-   `cost` is the cost of the network without L2 regularization
-   `lambtha` is the regularization parameter
-   `weights` is a dictionary of the weights and biases (`numpy.ndarray`s) of the neural network
-   `L` is the number of layers in the neural network
-   `m` is the number of data points used
-   Returns: the cost of the network accounting for L2 regularization

#### 1\. Gradient Descent with L2 Regularization

Write a function `def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):` that updates the weights and biases of a neural network using gradient descent with L2 regularization:

-   `Y` is a one-hot `numpy.ndarray` of shape `(classes, m)` that contains the correct labels for the data
    -   `classes` is the number of classes
    -   `m` is the number of data points
-   `weights` is a dictionary of the weights and biases of the neural network
-   `cache` is a dictionary of the outputs of each layer of the neural network
-   `alpha` is the learning rate
-   `lambtha` is the L2 regularization parameter
-   `L` is the number of layers of the network
-   The neural network uses `tanh` activations on each layer except the last, which uses a `softmax` activation
-   The weights and biases of the network should be updated in place

#### 2\. L2 Regularization Cost

Write the function `def l2_reg_cost(cost):` that calculates the cost of a neural network with L2 regularization:

-   `cost` is a tensor containing the cost of the network without L2 regularization
-   Returns: a tensor containing the cost of the network accounting for L2 regularization

#### 3\. Create a Layer with L2 Regularization

Write a function `def l2_reg_create_layer(prev, n, activation, lambtha):` that creates a `tensorflow` layer that includes L2 regularization:

-   `prev` is a tensor containing the output of the previous layer
-   `n` is the number of nodes the new layer should contain
-   `activation` is the activation function that should be used on the layer
-   `lambtha` is the L2 regularization parameter
-   Returns: the output of the new layer

#### 4\. Forward Propagation with Dropout

Write a function `def dropout_forward_prop(X, weights, L, keep_prob):` that conducts forward propagation using Dropout:

-   `X` is a `numpy.ndarray` of shape `(nx, m)` containing the input data for the network
    -   `nx` is the number of input features
    -   `m` is the number of data points
-   `weights` is a dictionary of the weights and biases of the neural network
-   `L` the number of layers in the network
-   `keep_prob` is the probability that a node will be kept
-   All layers except the last should use the `tanh` activation function
-   The last layer should use the `softmax` activation function
-   Returns: a dictionary containing the outputs of each layer and the dropout mask used on each layer (see example for format)

#### 5\. Gradient Descent with Dropout

Write a function `def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):` that updates the weights of a neural network with Dropout regularization using gradient descent:

-   `Y` is a one-hot `numpy.ndarray` of shape `(classes, m)` that contains the correct labels for the data
    -   `classes` is the number of classes
    -   `m` is the number of data points
-   `weights` is a dictionary of the weights and biases of the neural network
-   `cache` is a dictionary of the outputs and dropout masks of each layer of the neural network
-   `alpha` is the learning rate
-   `keep_prob` is the probability that a node will be kept
-   `L` is the number of layers of the network
-   All layers use the`tanh` activation function except the last, which uses the `softmax` activation function
-   The weights of the network should be updated in place

#### 6\. Create a Layer with Dropout

Write a function `def dropout_create_layer(prev, n, activation, keep_prob):` that creates a layer of a neural network using dropout:

-   `prev` is a tensor containing the output of the previous layer
-   `n` is the number of nodes the new layer should contain
-   `activation` is the activation function that should be used on the layer
-   `keep_prob` is the probability that a node will be kept
-   Returns: the output of the new layer

#### 7\. Early Stopping

Write the function `def early_stopping(cost, opt_cost, threshold, patience, count):` that determines if you should stop gradient descent early:

-   Early stopping should occur when the validation cost of the network has not decreased relative to the optimal validation cost by more than the threshold over a specific patience count
-   `cost` is the current validation cost of the neural network
-   `opt_cost` is the lowest recorded validation cost of the neural network
-   `threshold` is the threshold used for early stopping
-   `patience` is the patience count used for early stopping
-   `count` is the count of how long the threshold has not been met
-   Returns: a boolean of whether the network should be stopped early, followed by the updated count
