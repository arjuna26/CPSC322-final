import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from package.classifiers import MyDecisionTreeClassifier
from package.utils import preprocess_data

# Test: Initialization
def test_initialization():
    dt = MyDecisionTreeClassifier(max_depth=5, min_samples_split=3)
    assert dt.max_depth == 5
    assert dt.min_samples_split == 3
    assert dt.tree is None
    
# Test: _calculate_gini
def test_calculate_gini():
    dt = MyDecisionTreeClassifier()
    
    left_y = np.array([1, 1, 0])
    right_y = np.array([0, 0, 1, 1])
    gini = dt._calculate_gini(left_y, right_y)
    
    assert np.isclose(gini, 0.47619047619047616), f"Gini index incorrect, got {gini}"
    
# Test: fit
def test_fit():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    dt = MyDecisionTreeClassifier(max_depth=2)
    dt.fit(X, y)
    
    assert dt.tree is not None, "Tree not built after fit"
    assert dt.tree['feature'] == 0, "Incorrect feature selected"
    assert dt.tree['threshold'] == 2, "Incorrect threshold selected"
    
# Test: predict
def test_predict():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    dt = MyDecisionTreeClassifier(max_depth=2)
    dt.fit(X, y)
    
    predictions = dt.predict(X)
    assert np.array_equal(predictions, y), "Predictions do not match expected values"
    
# Test: _predict_sample
def test_predict_sample():
    tree = {
        'feature': 0,
        'threshold': 3,
        'left': {'label': 0},
        'right': {'label': 1}
    }
    dt = MyDecisionTreeClassifier()
    sample = np.array([4])
    prediction = dt._predict_sample(sample, tree)
    assert prediction == 1, "Prediction for sample incorrect"

# Test Decision Tree basic functionality
def test_decision_tree_basic():
    # Create a simple synthetic dataset for testing
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the decision tree
    tree = MyDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Test failed with accuracy: {accuracy}"

# Test: Depth constraint
def test_max_depth():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    dt = MyDecisionTreeClassifier(max_depth=1)
    dt.fit(X, y)
    
    assert 'label' in dt.tree['left'], "Tree depth exceeded max_depth constraint"
    assert 'label' in dt.tree['right'], "Tree depth exceeded max_depth constraint"

# Test overfitting: The decision tree should perform reasonably well on test data
def test_decision_tree_overfitting():
    # Generate a synthetic dataset with more complex relationships
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a decision tree
    tree = MyDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # Check training accuracy
    y_train_pred = tree.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Check test accuracy
    y_test_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Ensure the training accuracy is high and test accuracy is reasonable
    assert train_accuracy > 0.9, f"Train accuracy too low: {train_accuracy}"
    assert test_accuracy > 0.7, f"Test accuracy too low: {test_accuracy}"

# Test handling of missing values
def test_decision_tree_missing_values():
    # Create a simple dataset with missing values
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X[0, 0] = np.nan  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree
    tree = MyDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Test failed with accuracy: {accuracy}"

# Test decision tree with real data
def test_decision_tree_real_data():
    # Load real data
    data = pd.read_csv('../data/premier_league_data2021-24.csv')  
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the decision tree
    tree = MyDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X_train.values, y_train.values)
    
    # Make predictions
    y_pred = tree.predict(X_test.values)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Test failed with accuracy: {accuracy}"