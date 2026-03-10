from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """Common API for all MNIST classifiers."""

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Fit the model on the provided training data."""

    @abstractmethod
    def predict(self, X_test):
        """Return class predictions for the provided samples."""
