from __future__ import annotations

from .mnist_classifier_interface import MnistClassifierInterface
import tensorflow as tf


class FeedForwardNNClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28), num_classes: int = 10):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, X_train, y_train, epochs: int = 5, batch_size: int = 128, **kwargs):
        verbose = kwargs.get("verbose", 1)
        validation_split = kwargs.get("validation_split", 0.1)
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
        )
        return self

    def predict(self, X_test):
        probabilities = self.model.predict(X_test, verbose=0)
        return probabilities.argmax(axis=1)
