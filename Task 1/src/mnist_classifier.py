from cnn_classifier import CNNClassifier
from feed_forward_nn_classifier import FeedForwardNNClassifier
from random_forest_classifier import RandomForestClassifier


class MnistClassifier:
    """Wrapper with a stable API for every supported algorithm."""

    def __init__(self, algorithm: str, **model_kwargs):
        algorithms = {
            "rf": RandomForestClassifier,
            "nn": FeedForwardNNClassifier,
            "cnn": CNNClassifier,
        }
        try:
            model_class = algorithms[algorithm]
        except KeyError as exc:
            raise ValueError("Unknown algorithm. Use one of: rf, nn, cnn.") from exc
        self.algorithm = algorithm
        self.model = model_class(**model_kwargs)

    def train(self, X_train, y_train, **kwargs):
        return self.model.train(X_train, y_train, **kwargs)

    def predict(self, X_test):
        return self.model.predict(X_test)
