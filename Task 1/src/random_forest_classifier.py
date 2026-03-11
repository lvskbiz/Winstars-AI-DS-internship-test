from __future__ import annotations

from mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier


class RandomForestClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 150, random_state: int = 42):
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)
