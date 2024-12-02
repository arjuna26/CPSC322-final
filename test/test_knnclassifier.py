import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from package.classifiers import MyKNNClassifier

# Create sample data for testing
@pytest.fixture
def sample_data():
    # Simple dataset with two classes
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 7],
        [7, 8],
        [8, 8]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])  # Labels
    return X, y

@pytest.fixture
def knn_classifier():
    return MyKNNClassifier(n_neighbors=3)

# Test for the fit method
def test_fit(knn_classifier, sample_data):
    X, y = sample_data
    knn_classifier.fit(X, y)
    assert hasattr(knn_classifier, "X_train"), "X_train not set after fit"
    assert hasattr(knn_classifier, "y_train"), "y_train not set after fit"
    assert knn_classifier.X_train.shape == X.shape, "Training data shape mismatch"
    assert len(knn_classifier.y_train) == len(y), "Training labels shape mismatch"

# Test for the predict method
def test_predict(knn_classifier, sample_data):
    X, y = sample_data
    knn_classifier.fit(X, y)
    predictions = knn_classifier.predict(X)
    assert len(predictions) == len(y), "Prediction length mismatch"
    assert set(predictions).issubset(set(y)), "Predictions contain invalid class labels"

# Test for correct prediction on a simple case
def test_predict_accuracy(knn_classifier, sample_data):
    X, y = sample_data
    knn_classifier.fit(X, y)
    predictions = knn_classifier.predict(X)
    assert np.array_equal(predictions, y), "Predictions do not match actual labels on training data"

# Test for the score method
def test_score(knn_classifier, sample_data):
    X, y = sample_data
    knn_classifier.fit(X, y)
    accuracy = knn_classifier.score(X, y)
    expected_accuracy = accuracy_score(y, y)  # Training on the same data, expect 100% accuracy
    assert accuracy == expected_accuracy, "Score method returned incorrect accuracy"

