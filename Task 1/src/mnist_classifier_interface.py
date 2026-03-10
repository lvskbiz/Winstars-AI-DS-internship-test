# Interface for MNIST classifiers
class MnistClassifierInterface:
    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError
