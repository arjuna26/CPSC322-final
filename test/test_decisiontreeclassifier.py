import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from package.classifiers import MyDecisionTreeClassifier
from package.utils import preprocess_data

# Test Initialization
def test_initialization():
    dt = MyDecisionTreeClassifier(max_depth=5, min_samples_split=3)
    assert dt.max_depth == 5
    assert dt.min_samples_split == 3
    assert dt.tree is None

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
    X[0, 0] = np.nan  # Introduce a missing value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree
    tree = MyDecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.7, f"Test failed with accuracy: {accuracy}"

# Test decision tree with real data (e.g., Premier League dataset)
def test_decision_tree_real_data():
    # Load real data
    data = pd.read_csv('../data/premier_league_data2021-24.csv')  # Replace with your data file path
    
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

