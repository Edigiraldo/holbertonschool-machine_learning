#!/usr/bin/env python3
"""Function train_model."""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient
    descent, to analyze validation data, trains the model
    using early stopping and learning rate decay and also
    saves the best iteration of the model."""
    def scheduler(epoch):
        """
        - epochs is the number of passes through data for mini-batch
        gradient descent.
        - alpha is the initial learning rate
        """
        lr = alpha / (1 + (epoch*decay_rate))
        return lr

    cbacks = []
    if validation_data is not None and early_stopping:
        es = K.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       verbose=verbose,
                                       patience=patience)
        cbacks.append(es)

    if validation_data is not None and learning_rate_decay:
        lrd = K.callbacks.LearningRateScheduler(schedule=scheduler,
                                                verbose=1)
        cbacks.append(lrd)

    if save_best:
        sbest = K.callbacks.ModelCheckpoint(filepath,
                                            monitor='val_loss')
        cbacks.append(sbest)

    hist = network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=cbacks)

    return hist
