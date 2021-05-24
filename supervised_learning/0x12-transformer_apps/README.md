0x12. Transformer Applications
==============================

Learning Objectives
-------------------

### General

-   How to use Transformers for Machine Translation
-   How to write a custom train/test loop in Keras
-   How to use Tensorflow Datasets

Tasks
-----

### 0\. Dataset

Create the class `Dataset` that loads and preps a dataset for machine translation:

-   Class constructor `def __init__(self):`
    -   creates the instance attributes:
        -   `data_train`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `train` split, loaded `as_supervided`
        -   `data_valid`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `validate` split, loaded `as_supervided`
        -   `tokenizer_pt` is the Portuguese tokenizer created from the training set
        -   `tokenizer_en` is the English tokenizer created from the training set
-   Create the instance method `def tokenize_dataset(self, data):` that creates sub-word tokenizers for our dataset:
    -   `data` is a `tf.data.Dataset` whose examples are formatted as a tuple `(pt, en)`
        -   `pt` is the `tf.Tensor` containing the Portuguese sentence
        -   `en` is the `tf.Tensor` containing the corresponding English sentence
    -   The maximum vocab size should be set to `2**15`
    -   Returns: `tokenizer_pt, tokenizer_en`
        -   `tokenizer_pt` is the Portuguese tokenizer
        -   `tokenizer_en` is the English tokenizer

### 1\. Encode Tokens

Update the class `Dataset`:

-   Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens:
    -   `pt` is the `tf.Tensor` containing the Portuguese sentence
    -   `en` is the `tf.Tensor` containing the corresponding English sentence
    -   The tokenized sentences should include the start and end of sentence tokens
    -   The start token should be indexed as `vocab_size`
    -   The end token should be indexed as `vocab_size + 1`
    -   Returns: `pt_tokens, en_tokens`
        -   `pt_tokens` is a `np.ndarray` containing the Portuguese tokens
        -   `en_tokens` is a `np.ndarray.` containing the English tokens

### 2\. TF Encode

Update the class `Dataset`:

-   Create the instance method `def tf_encode(self, pt, en):` that acts as a `tensorflow` wrapper for the `encode` instance method
    -   Make sure to set the shape of the `pt` and `en` return tensors
-   Update the class constructor `def __init__(self):`
    -   update the `data_train` and `data_validate` attributes by tokenizing the examples

### 3\. Pipeline

Update the class `Dataset` to set up the data pipeline:

-   Update the class constructor `def __init__(self, batch_size, max_len):`
    -   `batch_size` is the batch size for training/validation
    -   `max_len` is the maximum number of tokens allowed per example sentence
    -   update the `data_train` attribute by performing the following actions:
        -   filter out all examples that have either sentence with more than `max_len` tokens
        -   cache the dataset to increase performance
        -   shuffle the entire dataset
        -   split the dataset into padded batches of size `batch_size`
        -   prefetch the dataset using `tf.data.experimental.AUTOTUNE` to increase performance
    -   update the `data_validate` attribute by performing the following actions:
        -   filter out all examples that have either sentence with more than `max_len` tokens
        -   split the dataset into padded batches of size `batch_size`

### 4\. Create Masks

Create the function `def create_masks(inputs, target):` that creates all masks for training/validation:

-   `inputs` is a tf.Tensor of shape `(batch_size, seq_len_in)` that contains the input sentence
-   `target` is a tf.Tensor of shape `(batch_size, seq_len_out)` that contains the target sentence
-   This function should only use `tensorflow` operations in order to properly function in the training step
-   Returns: `encoder_mask`, `look_ahead_mask`, `decoder_mask`
    -   `encoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
    -   `look_ahead_mask` is the `tf.Tensor` look ahead mask of shape `(batch_size, 1, seq_len_out, seq_len_out)` to be applied in the decoder
    -   `decoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the decoder

### 5\. Train

Take your implementation of a transformer from our [previous project](https://intranet.hbtn.io/rltoken/xFGAKD-jaUWnsvOXMTPcvw "previous project") and save it to the file `5-transformer.py`. Note, you may need to make slight adjustments to this model to get it to functionally train.

Write a the function `def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):` that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:

-   `N` the number of blocks in the encoder and decoder
-   `dm` the dimensionality of the model
-   `h` the number of heads
-   `hidden` the number of hidden units in the fully connected layers
-   `max_len` the maximum number of tokens per sequence
-   `batch_size` the batch size for training
-   `epochs` the number of epochs to train for
-   You should use the following imports:
    -   `Dataset = __import__('3-dataset').Dataset`
    -   `create_masks = __import__('4-create_masks').create_masks`
    -   `Transformer = __import__('5-transformer').Transformer`
-   Your model should be trained with Adam optimization with `beta_1=0.9`, `beta_2=0.98`, `epsilon=1e-9`
    -   The learning rate should be scheduled using the following equation with `warmup_steps=4000`:
    -   ![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/9/39ceb6fefc25283cd8ee7a3f302ae799b6051bcd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210524T144328Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e8546527f7e6bec823391d33a24cf2ad3e37597d7cb68fe2464c0e661ed3b91b)
-   Your model should use sparse categorical crossentropy loss, ignoring padded tokens
-   Your model should print the following information about the training:
    -   Every 50 batches, you should print `Epoch {Epoch number}, batch {batch_number}: loss {training_loss} accuracy {training_accuracy}`
    -   Every epoch, you should print `Epoch {Epoch number}: loss {training_loss} accuracy {training_accuracy}`
-   Returns the trained model
