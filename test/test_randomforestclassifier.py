import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from package.classifiers import MyRandomForestClassifier, MyDecisionTreeClassifier
from package.utils import preprocess_data

# Test: Initialization
def test_initialization():
    rf = MyRandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=3, max_features='sqrt', bootstrap=False)
    assert rf.n_estimators == 50
    assert rf.max_depth == 10
    assert rf.min_samples_split == 3
    assert rf.max_features == 'sqrt'
    assert rf.bootstrap is False
    assert rf.trees == []
    
# Test: fit - Check number of trees and feature subset size
def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    rf = MyRandomForestClassifier(n_estimators=3, max_features=1, bootstrap=True)
    rf.fit(X, y)
    
    assert len(rf.trees) == 3, "Number of trees does not match n_estimators"
    for tree, feature_indices in rf.trees:
        assert isinstance(tree, MyDecisionTreeClassifier), "Tree is not a MyDecisionTreeClassifier instance"
        assert len(feature_indices) == 1, "Number of features used in tree does not match max_features"

# Test: predict - Basic majority vote
def test_predict_majority_vote():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 1, 1, 1, 1])
    rf = MyRandomForestClassifier(n_estimators=5)
    rf.fit(X, y)

    predictions = rf.predict(X)
    assert np.array_equal(predictions, y), "Predictions do not match expected majority vote"

# Test: predict - Ensure correct feature subset is used
def test_predict_feature_subset():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    rf = MyRandomForestClassifier(n_estimators=1, max_features=1)
    rf.fit(X, y)
    
    tree, feature_indices = rf.trees[0]
    predictions = rf.predict(X)
    # Ensure predictions are based only on the selected feature
    assert tree is not None
    assert len(feature_indices) == 1, "Prediction did not respect max_features constraint"

# Test: Consistency with deterministic random seed
def test_random_seed_consistency():
    np.random.seed(42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    rf1 = MyRandomForestClassifier(n_estimators=3, max_features=1, bootstrap=True)
    rf1.fit(X, y)
    predictions1 = rf1.predict(X)
    
    np.random.seed(42)
    rf2 = MyRandomForestClassifier(n_estimators=3, max_features=1, bootstrap=True)
    rf2.fit(X, y)
    predictions2 = rf2.predict(X)
    
    assert np.array_equal(predictions1, predictions2), "Random seed does not ensure consistent results"

# Mock decision tree for majority voting test
class MockDecisionTree:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return np.array(self.predictions)

# Test: Majority voting with diverse predictions
def test_majority_voting():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 1])
    rf = MyRandomForestClassifier(n_estimators=3)
    rf.fit(X, y)
    
    # Mock diverse predictions from individual trees
    rf.trees[0] = (MockDecisionTree([0, 1, 1]), [0])
    rf.trees[1] = (MockDecisionTree([1, 0, 0]), [0])
    rf.trees[2] = (MockDecisionTree([1, 1, 0]), [0])
    
    predictions = rf.predict(X)
    assert np.array_equal(predictions, [1, 1, 0]), "Majority voting is incorrect"

# Test Random Forest basic functionality
def test_random_forest_basic():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = MyRandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.6, f"Test failed with accuracy: {accuracy}"

# Test overfitting
def test_random_forest_overfitting():
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = MyRandomForestClassifier(n_estimators=50, max_depth=None, max_features='sqrt', bootstrap=True)
    rf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    assert train_accuracy > 0.9, f"Train accuracy too low: {train_accuracy}"
    assert test_accuracy > 0.75, f"Test accuracy too low: {test_accuracy}"

# Test hyperparameter tuning
@pytest.mark.parametrize("n_estimators, max_features", [
    (5, 'sqrt'),
    (20, None)
])
def test_random_forest_hyperparameters(n_estimators, max_features):
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = MyRandomForestClassifier(n_estimators=n_estimators, max_depth=10, max_features=max_features, bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.55, f"Test failed with n_estimators={n_estimators}, max_features={max_features}, accuracy={accuracy}"

# Test Random Forest with real data
def test_random_forest_real_data():
    data = pd.read_csv('../data/premier_league_data2021-24.csv') 
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    rf = MyRandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', bootstrap=True)
    rf.fit(X_train.values, y_train.values)
    y_pred = rf.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.6, f"Test failed with accuracy: {accuracy}"