from __future__ import annotations

from mnist_classifier_interface import MnistClassifierInterface

try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
except ImportError as exc:  # pragma: no cover
    SklearnRandomForestClassifier = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


class RandomForestClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 150, random_state: int = 42):
        if SklearnRandomForestClassifier is None:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for RandomForestClassifier. "
                "Install dependencies from Task 1/requirements.txt."
            ) from SKLEARN_IMPORT_ERROR
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
