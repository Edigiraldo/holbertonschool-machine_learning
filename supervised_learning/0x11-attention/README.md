0x11. Attention
===============

Learning Objectives
-------------------

### General

-   What is the attention mechanism?
-   How to apply attention to RNNs
-   What is a transformer?
-   How to create an encoder-decoder transformer model
-   What is GPT?
-   What is BERT?
-   What is self-supervised learning?
-   How to use BERT for specific NLP tasks
-   What is SQuAD? GLUE?


Tasks
-----

### 0\. RNN Encoder

Create a class `RNNEncoder` that inherits from `tensorflow.keras.layers.Layer` to encode for machine translation:

-   Class constructor `def __init__(self, vocab, embedding, units, batch):`
    -   `vocab` is an integer representing the size of the input vocabulary
    -   `embedding` is an integer representing the dimensionality of the embedding vector
    -   `units` is an integer representing the number of hidden units in the RNN cell
    -   `batch` is an integer representing the batch size
    -   Sets the following public instance attributes:
        -   `batch` - the batch size
        -   `units` - the number of hidden units in the RNN cell
        -   `embedding` - a `keras` Embedding layer that converts words from the vocabulary into an embedding vector
        -   `gru` - a `keras` GRU layer with `units` units
            -   Should return both the full sequence of outputs as well as the last hidden state
            -   Recurrent weights should be initialized with `glorot_uniform`
-   Public instance method `def initialize_hidden_state(self):`
    -   Initializes the hidden states for the RNN cell to a tensor of zeros
    -   Returns: a tensor of shape `(batch, units)`containing the initialized hidden states
-   Public instance method `def call(self, x, initial):`
    -   `x` is a tensor of shape `(batch, input_seq_len)` containing the input to the encoder layer as word indices within the vocabulary
    -   `initial` is a tensor of shape `(batch, units)` containing the initial hidden state
    -   Returns: `outputs, hidden`
        -   `outputs` is a tensor of shape `(batch, input_seq_len, units)`containing the outputs of the encoder
        -   `hidden` is a tensor of shape `(batch, units)` containing the last hidden state of the encoder

### 1\. Self Attention

Create a class `SelfAttention` that inherits from `tensorflow.keras.layers.Layer` to calculate the attention for machine translation based on [this paper](https://intranet.hbtn.io/rltoken/YlDIODUFbkYQbRL3a5CwEQ "this paper"):

-   Class constructor `def __init__(self, units):`
    -   `units` is an integer representing the number of hidden units in the alignment model
    -   Sets the following public instance attributes:
        -   `W` - a Dense layer with `units` units, to be applied to the previous decoder hidden state
        -   `U` - a Dense layer with `units` units, to be applied to the encoder hidden states
        -   `V` - a Dense layer with `1` units, to be applied to the tanh of the sum of the outputs of `W` and `U`
-   Public instance method `def call(self, s_prev, hidden_states):`
    -   `s_prev` is a tensor of shape `(batch, units)` containing the previous decoder hidden state
    -   `hidden_states` is a tensor of shape `(batch, input_seq_len, units)`containing the outputs of the encoder
    -   Returns: `context, weights`
        -   `context` is a tensor of shape `(batch, units)` that contains the context vector for the decoder
        -   `weights` is a tensor of shape `(batch, input_seq_len, 1)` that contains the attention weights

### 2\. Self Attention

Create a class `SelfAttention` that inherits from `tensorflow.keras.layers.Layer` to calculate the attention for machine translation based on [this paper](https://intranet.hbtn.io/rltoken/YlDIODUFbkYQbRL3a5CwEQ "this paper"):

-   Class constructor `def __init__(self, units):`

    -   `units` is an integer representing the number of hidden units in the alignment model

    -   Sets the following public instance attributes:

        -   `W` - a Dense layer with `units` units, to be applied to the previous decoder hidden state

        -   `U` - a Dense layer with `units` units, to be applied to the encoder hidden states

        -   `V` - a Dense layer with `1` units, to be applied to the tanh of the sum of the outputs of `W` and `U`

-   Public instance method `def call(self, s_prev, hidden_states):`

    -   `s_prev` is a tensor of shape `(batch, units)` containing the previous decoder hidden state

    -   `hidden_states` is a tensor of shape `(batch, input_seq_len, units)`containing the outputs of the encoder

    -   Returns: `context, weights`

        -   `context` is a tensor of shape `(batch, units)` that contains the context vector for the decoder

        -   `weights` is a tensor of shape `(batch, input_seq_len, 1)` that contains the attention weights

### 4\. Positional Encoding

Write the function `def positional_encoding(max_seq_len, dm):` that calculates the positional encoding for a transformer:

-   `max_seq_len` is an integer representing the maximum sequence length
-   `dm` is the model depth
-   Returns: a `numpy.ndarray` of shape `(max_seq_len, dm)` containing the positional encoding vectors
-   You should use `import numpy as np`

### 5\. Scaled Dot Product Attention

Write the function `def sdp_attention(Q, K, V, mask=None)` that calculates the scaled dot product attention:

-   `Q` is a tensor with its last two dimensions as `(..., seq_len_q, dk)` containing the query matrix
-   `K` is a tensor with its last two dimensions as `(..., seq_len_v, dk)` containing the key matrix
-   `V` is a tensor with its last two dimensions as `(..., seq_len_v, dv)` containing the value matrix
-   `mask` is a tensor that can be broadcast into `(..., seq_len_q, seq_len_v)` containing the optional mask, or defaulted to `None`
    -   if `mask` is not `None`, multiply `-1e9` to the mask and add it to the scaled matrix multiplication
-   The preceding dimensions of `Q`, `K`, and `V` are the same
-   Returns: `output, weights`
    -   `output`a tensor with its last two dimensions as `(..., seq_len_q, dv)` containing the scaled dot product attention
    -   `weights` a tensor with its last two dimensions as `(..., seq_len_q, seq_len_v)` containing the attention weights

### 6\. Multi Head Attention

Create a class `MultiHeadAttention` that inherits from `tensorflow.keras.layers.Layer` to perform multi head attention:

-   Class constructor `def __init__(self, dm, h):`
    -   `dm` is an integer representing the dimensionality of the model
    -   `h` is an integer representing the number of heads
    -   `dm` is divisible by `h`
    -   Sets the following public instance attributes:
        -   `h` - the number of heads
        -   `dm` - the dimensionality of the model
        -   `depth` - the depth of each attention head
        -   `Wq` - a Dense layer with `dm` units, used to generate the query matrix
        -   `Wk` - a Dense layer with `dm` units, used to generate the key matrix
        -   `Wv` - a Dense layer with `dm` units, used to generate the value matrix
        -   `linear` - a Dense layer with `dm` units, used to generate the attention output
-   Public instance method `def call(self, Q, K, V, mask):`
    -   `Q` is a tensor of shape `(batch, seq_len_q, dk)` containing the input to generate the query matrix
    -   `K` is a tensor of shape `(batch, seq_len_v, dk)` containing the input to generate the key matrix
    -   `V` is a tensor of shape `(batch, seq_len_v, dv)` containing the input to generate the value matrix
    -   `mask` is always `None`
    -   Returns: `output, weights`
        -   `output`a tensor with its last two dimensions as `(..., seq_len_q, dm)` containing the scaled dot product attention
        -   `weights` a tensor with its last three dimensions as `(..., h, seq_len_q, seq_len_v)` containing the attention weights
-   You should use `sdp_attention = __import__('5-sdp_attention').sdp_attention`

### 7\. Transformer Encoder Block

Create a class `EncoderBlock` that inherits from `tensorflow.keras.layers.Layer` to create an encoder block for a transformer:

-   Class constructor `def __init__(self, dm, h, hidden, drop_rate=0.1):`
    -   `dm` - the dimensionality of the model
    -   `h` - the number of heads
    -   `hidden` - the number of hidden units in the fully connected layer
    -   `drop_rate` - the dropout rate
    -   Sets the following public instance attributes:
        -   `mha` - a `MultiHeadAttention` layer
        -   `dense_hidden` - the hidden dense layer with `hidden` units and `relu` activation
        -   `dense_output` - the output dense layer with `dm` units
        -   `layernorm1` - the first layer norm layer, with `epsilon=1e-6`
        -   `layernorm2` - the second layer norm layer, with `epsilon=1e-6`
        -   `dropout1` - the first dropout layer
        -   `dropout2` - the second dropout layer
-   Public instance method `call(self, x, training, mask=None):`
    -   `x` - a tensor of shape `(batch, input_seq_len, dm)`containing the input to the encoder block
    -   `training` - a boolean to determine if the model is training
    -   `mask` - the mask to be applied for multi head attention
    -   Returns: a tensor of shape `(batch, input_seq_len, dm)` containing the block's output
-   You should use `MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention`

### 8\. Transformer Decoder Block

Create a class `DecoderBlock` that inherits from `tensorflow.keras.layers.Layer` to create an encoder block for a transformer:

-   Class constructor `def __init__(self, dm, h, hidden, drop_rate=0.1):`
    -   `dm` - the dimensionality of the model
    -   `h` - the number of heads
    -   `hidden` - the number of hidden units in the fully connected layer
    -   `drop_rate` - the dropout rate
    -   Sets the following public instance attributes:
        -   `mha1` - the first `MultiHeadAttention` layer
        -   `mha2` - the second `MultiHeadAttention` layer
        -   `dense_hidden` - the hidden dense layer with `hidden` units and `relu` activation
        -   `dense_output` - the output dense layer with `dm` units
        -   `layernorm1` - the first layer norm layer, with `epsilon=1e-6`
        -   `layernorm2` - the second layer norm layer, with `epsilon=1e-6`
        -   `layernorm3` - the third layer norm layer, with `epsilon=1e-6`
        -   `dropout1` - the first dropout layer
        -   `dropout2` - the second dropout layer
        -   `dropout3` - the third dropout layer
-   Public instance method `def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):`
    -   `x` - a tensor of shape `(batch, target_seq_len, dm)`containing the input to the decoder block
    -   `encoder_output` - a tensor of shape `(batch, input_seq_len, dm)`containing the output of the encoder
    -   `training` - a boolean to determine if the model is training
    -   `look_ahead_mask` - the mask to be applied to the first multi head attention layer
    -   `padding_mask` - the mask to be applied to the second multi head attention layer
    -   Returns: a tensor of shape `(batch, target_seq_len, dm)` containing the block's output
-   You should use `MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention`

### 9\. Transformer Encoder

Create a class `Encoder` that inherits from `tensorflow.keras.layers.Layer` to create the encoder for a transformer:

-   Class constructor `def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):`
    -   `N` - the number of blocks in the encoder
    -   `dm` - the dimensionality of the model
    -   `h` - the number of heads
    -   `hidden` - the number of hidden units in the fully connected layer
    -   `input_vocab` - the size of the input vocabulary
    -   `max_seq_len` - the maximum sequence length possible
    -   `drop_rate` - the dropout rate
    -   Sets the following public instance attributes:
        -   `N` - the number of blocks in the encoder
        -   `dm` - the dimensionality of the model
        -   `embedding` - the embedding layer for the inputs
        -   `positional_encoding` - a `numpy.ndarray` of shape `(max_seq_len, dm)` containing the positional encodings
        -   `blocks` - a list of length `N` containing all of the `EncoderBlock`'s
        -   `dropout` - the dropout layer, to be applied to the positional encodings
-   Public instance method `call(self, x, training, mask):`
    -   `x` - a tensor of shape `(batch, input_seq_len, dm)`containing the input to the encoder
    -   `training` - a boolean to determine if the model is training
    -   `mask` - the mask to be applied for multi head attention
    -   Returns: a tensor of shape `(batch, input_seq_len, dm)` containing the encoder output
-   You should use `positional_encoding = __import__('4-positional_encoding').positional_encoding` and `EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock`

### 10\. Transformer Decoder

Create a class `Decoder` that inherits from `tensorflow.keras.layers.Layer` to create the decoder for a transformer:

-   Class constructor `def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):`
-   `N` - the number of blocks in the encoder
    -   `dm` - the dimensionality of the model
    -   `h` - the number of heads
    -   `hidden` - the number of hidden units in the fully connected layer
    -   `target_vocab` - the size of the target vocabulary
    -   `max_seq_len` - the maximum sequence length possible
    -   `drop_rate` - the dropout rate
    -   Sets the following public instance attributes:
        -   `N` - the number of blocks in the encoder
        -   `dm` - the dimensionality of the model
        -   `embedding` - the embedding layer for the targets
        -   `positional_encoding` - a `numpy.ndarray` of shape `(max_seq_len, dm)` containing the positional encodings
        -   `blocks` - a list of length `N` containing all of the `DecoderBlock`'s
        -   `dropout` - the dropout layer, to be applied to the positional encodings
-   Public instance method `def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):`
    -   `x` - a tensor of shape `(batch, target_seq_len, dm)`containing the input to the decoder
    -   `encoder_output` - a tensor of shape `(batch, input_seq_len, dm)`containing the output of the encoder
    -   `training` - a boolean to determine if the model is training
    -   `look_ahead_mask` - the mask to be applied to the first multi head attention layer
    -   `padding_mask` - the mask to be applied to the second multi head attention layer
    -   Returns: a tensor of shape `(batch, target_seq_len, dm)` containing the decoder output
-   You should use `positional_encoding = __import__('4-positional_encoding').positional_encoding` and `DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock`

### 11\. Transformer Network

Create a class `Transformer` that inherits from `tensorflow.keras.Model` to create a transformer network:

-   Class constructor `def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):`
    -   `N` - the number of blocks in the encoder and decoder
    -   `dm` - the dimensionality of the model
    -   `h` - the number of heads
    -   `hidden` - the number of hidden units in the fully connected layers
    -   `input_vocab` - the size of the input vocabulary
    -   `target_vocab` - the size of the target vocabulary
    -   `max_seq_input` - the maximum sequence length possible for the input
    -   `max_seq_target` - the maximum sequence length possible for the target
    -   `drop_rate` - the dropout rate
    -   Sets the following public instance attributes:
        -   `encoder` - the encoder layer
        -   `decoder` - the decoder layer
        -   `linear` - a final Dense layer with `target_vocab` units
-   Public instance method `def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):`
    -   `inputs` - a tensor of shape `(batch, input_seq_len)`containing the inputs
    -   `target` - a tensor of shape `(batch, target_seq_len)`containing the target
    -   `training` - a boolean to determine if the model is training
    -   `encoder_mask` - the padding mask to be applied to the encoder
    -   `look_ahead_mask` - the look ahead mask to be applied to the decoder
    -   `decoder_mask` - the padding mask to be applied to the decoder
    -   Returns: a tensor of shape `(batch, target_seq_len, target_vocab)` containing the transformer output
-   You should use `Encoder = __import__('9-transformer_encoder').Encoder` and `Decoder = __import__('10-transformer_decoder').Decoder`
