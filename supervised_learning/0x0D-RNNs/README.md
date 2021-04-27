0x0D. RNNs
==========

Tasks
-----

### 0\. RNN Cell

Create the class `RNNCell` that represents a cell of a simple RNN:

-   class constructor `def __init__(self, i, h, o):`
    -   `i` is the dimensionality of the data
    -   `h` is the dimensionality of the hidden state
    -   `o` is the dimensionality of the outputs
    -   Creates the public instance attributes `Wh`, `Wy`, `bh`, `by` that represent the weights and biases of the cell
        -   `Wh` and `bh` are for the concatenated hidden state and input data
        -   `Wy` and `by` are for the output
    -   The weights should be initialized using a random normal distribution in the order listed above
    -   The weights will be used on the right side for matrix multiplication
    -   The biases should be initialized as zeros
-   public instance method `def forward(self, h_prev, x_t):` that performs forward propagation for one time step
    -   `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data input for the cell
        -   `m` is the batche size for the data
    -   `h_prev` is a `numpy.ndarray` of shape `(m, h)` containing the previous hidden state
    -   The output of the cell should use a softmax activation function
    -   Returns: `h_next, y`
        -   `h_next` is the next hidden state
        -   `y` is the output of the cell

### 1\. RNN

Write the function `def rnn(rnn_cell, X, h_0):` that performs forward propagation for a simple RNN:

-   `rnn_cell` is an instance of `RNNCell` that will be used for the forward propagation
-   `X` is the data to be used, given as a `numpy.ndarray` of shape `(t, m, i)`
    -   `t` is the maximum number of time steps
    -   `m` is the batch size
    -   `i` is the dimensionality of the data
-   `h_0` is the initial hidden state, given as a `numpy.ndarray` of shape `(m, h)`
    -   `h` is the dimensionality of the hidden state
-   Returns: `H, Y`
    -   `H` is a `numpy.ndarray` containing all of the hidden states
    -   `Y` is a `numpy.ndarray` containing all of the outputs

### 2\. GRU Cell

Create the class `GRUCell` that represents a gated recurrent unit:

-   class constructor `def __init__(self, i, h, o):`
    -   `i` is the dimensionality of the data
    -   `h` is the dimensionality of the hidden state
    -   `o` is the dimensionality of the outputs
    -   Creates the public instance attributes `Wz`, `Wr`, `Wh`, `Wy`, `bz`, `br`, `bh`, `by` that represent the weights and biases of the cell
        -   `Wz`and `bz` are for the update gate
        -   `Wr`and `br` are for the reset gate
        -   `Wh`and `bh` are for the intermediate hidden state
        -   `Wy`and `by` are for the output
    -   The weights should be initialized using a random normal distribution in the order listed above
    -   The weights will be used on the right side for matrix multiplication
    -   The biases should be initialized as zeros
-   public instance method `def forward(self, h_prev, x_t):` that performs forward propagation for one time step
    -   `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data input for the cell
        -   `m` is the batche size for the data
    -   `h_prev` is a `numpy.ndarray` of shape `(m, h)` containing the previous hidden state
    -   The output of the cell should use a softmax activation function
    -   Returns: `h_next, y`
        -   `h_next` is the next hidden state
        -   `y` is the output of the cell

### 3\. LSTM Cell

Create the class `LSTMCell` that represents an LSTM unit:

-   class constructor `def __init__(self, i, h, o):`
    -   `i` is the dimensionality of the data
    -   `h` is the dimensionality of the hidden state
    -   `o` is the dimensionality of the outputs
    -   Creates the public instance attributes `Wf`, `Wu`, `Wc`, `Wo`, `Wy`, `bf`, `bu`, `bc`, `bo`, `by` that represent the weights and biases of the cell
        -   `Wf`and `bf` are for the forget gate
        -   `Wu`and `bu` are for the update gate
        -   `Wc`and `bc` are for the intermediate cell state
        -   `Wo`and `bo` are for the output gate
        -   `Wy`and `by` are for the outputs
    -   The weights should be initialized using a random normal distribution in the order listed above
    -   The weights will be used on the right side for matrix multiplication
    -   The biases should be initialized as zeros
-   public instance method `def forward(self, h_prev, c_prev, x_t):` that performs forward propagation for one time step
    -   `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data input for the cell
        -   `m` is the batche size for the data
    -   `h_prev` is a `numpy.ndarray` of shape `(m, h)` containing the previous hidden state
    -   `c_prev` is a `numpy.ndarray` of shape `(m, h)` containing the previous cell state
    -   The output of the cell should use a softmax activation function
    -   Returns: `h_next, c_next, y`
        -   `h_next` is the next hidden state
        -   `c_next` is the next cell state
        -   `y` is the output of the cell

### 4\. Deep RNN

Write the function `def deep_rnn(rnn_cells, X, h_0):` that performs forward propagation for a deep RNN:

-   `rnn_cells` is a list of `RNNCell` instances of length `l` that will be used for the forward propagation
    -   `l` is the number of layers
-   `X` is the data to be used, given as a `numpy.ndarray` of shape `(t, m, i)`
    -   `t` is the maximum number of time steps
    -   `m` is the batch size
    -   `i` is the dimensionality of the data
-   `h_0` is the initial hidden state, given as a `numpy.ndarray` of shape `(l, m, h)`
    -   `h` is the dimensionality of the hidden state
-   Returns: `H, Y`
    -   `H` is a `numpy.ndarray` containing all of the hidden states
    -   `Y` is a `numpy.ndarray` containing all of the outputs

### 5\. Bidirectional Cell Forward

Create the class `BidirectionalCell` that represents a bidirectional cell of an RNN:

-   class constructor `def __init__(self, i, h, o):`
    -   `i` is the dimensionality of the data
    -   `h` is the dimensionality of the hidden states
    -   `o` is the dimensionality of the outputs
    -   Creates the public instance attributes `Whf`, `Whb`, `Wy`, `bhf`, `bhb`, `by` that represent the weights and biases of the cell
        -   `Whf` and `bhf`are for the hidden states in the forward direction
        -   `Whb` and `bhb`are for the hidden states in the backward direction
        -   `Wy` and `by`are for the outputs
    -   The weights should be initialized using a random normal distribution in the order listed above
    -   The weights will be used on the right side for matrix multiplication
    -   The biases should be initialized as zeros
-   public instance method `def forward(self, h_prev, x_t):` that calculates the hidden state in the forward direction for one time step
    -   `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data input for the cell
        -   `m` is the batch size for the data
    -   `h_prev` is a `numpy.ndarray` of shape `(m, h)` containing the previous hidden state
    -   Returns: `h_next`, the next hidden state

### 6\. Bidirectional Cell Backward

Update the class `BidirectionalCell`, based on `5-bi_forward.py`:

-   public instance method `def backward(self, h_next, x_t):` that calculates the hidden state in the backward direction for one time step
    -   `x_t` is a `numpy.ndarray` of shape `(m, i)` that contains the data input for the cell
        -   `m` is the batch size for the data
    -   `h_next` is a `numpy.ndarray` of shape `(m, h)` containing the next hidden state
    -   Returns: `h_pev`, the previous hidden state

### 7\. Bidirectional Output

Update the class `BidirectionalCell`, based on `6-bi_backward.py`:

-   public instance method `def output(self, H):` that calculates all outputs for the RNN:
    -   `H` is a `numpy.ndarray` of shape `(t, m, 2 * h)` that contains the concatenated hidden states from both directions, excluding their initialized states
        -   `t` is the number of time steps
        -   `m` is the batch size for the data
        -   `h` is the dimensionality of the hidden states
    -   Returns: `Y`, the outputs

### 8\. Bidirectional RNN

Write the function `def bi_rnn(bi_cell, X, h_0, h_t):` that performs forward propagation for a bidirectional RNN:

-   `bi_cell` is an instance of `BidirectinalCell` that will be used for the forward propagation
-   `X` is the data to be used, given as a `numpy.ndarray` of shape `(t, m, i)`
    -   `t` is the maximum number of time steps
    -   `m` is the batch size
    -   `i` is the dimensionality of the data
-   `h_0` is the initial hidden state in the forward direction, given as a `numpy.ndarray` of shape `(m, h)`
    -   `h` is the dimensionality of the hidden state
-   `h_t` is the initial hidden state in the backward direction, given as a `numpy.ndarray` of shape `(m, h)`
-   Returns: `H, Y`
    -   `H` is a `numpy.ndarray` containing all of the concatenated hidden states
    -   `Y` is a `numpy.ndarray` containing all of the outputs
