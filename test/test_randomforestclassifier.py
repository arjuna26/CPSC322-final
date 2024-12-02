import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from package.classifiers import MyRandomForestClassifier 
from package.utils import preprocess_data


# 1. Test Random Forest basic functionality
def test_random_forest_basic():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = MyRandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', bootstrap=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.6, f"Test failed with accuracy: {accuracy}"

# 2. Test overfitting
def test_random_forest_overfitting():
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = MyRandomForestClassifier(n_estimators=50, max_depth=None, max_features='sqrt', bootstrap=True)
    rf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    assert train_accuracy > 0.9, f"Train accuracy too low: {train_accuracy}"
    assert test_accuracy > 0.75, f"Test accuracy too low: {test_accuracy}"

# 3. Test hyperparameter tuning
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
    assert accuracy > 0.6, f"Test failed with n_estimators={n_estimators}, max_features={max_features}, accuracy={accuracy}"

# 4. Test Random Forest with real data
def test_random_forest_real_data():
    data = pd.read_csv('../data/premier_league_data2021-24.csv')  # Replace with your data path
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    rf = MyRandomForestClassifier(n_estimators=10, max_depth=5, max_features='sqrt', bootstrap=True)
    rf.fit(X_train.values, y_train.values)
    y_pred = rf.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.6, f"Test failed with accuracy: {accuracy}"

# To run the tests
if __name__ == "__main__":
    pytest.main()
