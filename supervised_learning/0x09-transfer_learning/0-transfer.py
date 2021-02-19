#!/usr/bin/env python3
"""Script to train the CIFAR10 dataset."""
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Function that pre-processes the data for your model.

       - X is a numpy.ndarray of shape (m, 32, 32, 3) containing
         the CIFAR 10 data, where m is the number of data points.
       - Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
         labels for X.

       Returns: X_p, Y_p

           - X_p is a numpy.ndarray containing the preprocessed X.
           - Y_p is a numpy.ndarray containing the preprocessed Y.
    """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


if __name__ == "__main__":
    he_normal = K.initializers.he_normal()
    (x_train, y_train), (x_val, y_val) = K.datasets.cifar10.load_data()

    x_t, y_t = preprocess_data(x_train, y_train)
    x_v, y_v = preprocess_data(x_val, y_val)

    base_model = K.applications.ResNet50(weights='imagenet',
                                         include_top=False,
                                         input_shape=(224, 224, 3))
    base_model.trainable = False

    input_tensor = K.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(lambda image: tf.image.resize(image,
                        size=(224, 224)))(input_tensor)

    x = base_model(x, training=False)
    x = K.layers.Flatten()(x)

    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(32, activation='relu', kernel_initializer=he_normal)(x)
    x = K.layers.Dropout(0.4)(x)

    x = K.layers.BatchNormalization()(x)
    output = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=he_normal)(x)

    model = K.models.Model(inputs=input_tensor, outputs=output)

    opt = K.optimizers.Adam(lr=0.00001)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=['accuracy'])

    save_best = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                            monitor='val_acc',
                                            mode='max',
                                            save_best_only=True)
    model.fit(x=x_t, y=y_t, batch_size=256, epochs=20,
              verbose=1, callbacks=[save_best],
              validation_data=(x_v, y_v), shuffle=True)
