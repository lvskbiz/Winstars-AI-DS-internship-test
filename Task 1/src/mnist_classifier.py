# MnistClassifier wrapper class
from random_forest_classifier import RandomForestClassifier
from feed_forward_nn_classifier import FeedForwardNNClassifier
from cnn_classifier import CNNClassifier

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestClassifier()
        elif algorithm == 'nn':
            self.model = FeedForwardNNClassifier()
        elif algorithm == 'cnn':
            self.model = CNNClassifier()
        else:
            raise ValueError('Unknown algorithm')

    def train(self, X_train, y_train):
        return self.model.train(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
