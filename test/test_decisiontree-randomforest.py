from package.classifiers import MyDecisionTreeClassifier, MyRandomForestClassifier
import pytest
import numpy as np


# Helper Data for Tests
X_simple = np.array([[0], [1], [2], [3], [4]])
y_simple = np.array([0, 0, 1, 1, 1])

X_multi = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
y_multi = np.array([0, 0, 1, 1, 1])

def test_decision_tree_fit_predict():
    """
    Test fitting and predicting with MyDecisionTreeClassifier on simple data.
    """
    clf = MyDecisionTreeClassifier()
    clf.fit(X_simple, y_simple)
    
    # Predict and check correctness
    predictions = clf.predict(X_simple)
    assert np.array_equal(predictions, y_simple), "Decision Tree predictions are incorrect."

def test_decision_tree_unfitted():
    """
    Test that MyDecisionTreeClassifier raises an exception if predict is called before fitting.
    """
    clf = MyDecisionTreeClassifier()
    with pytest.raises(Exception, match="Tree has not been fitted yet."):
        clf.predict(X_simple)

def test_random_forest_fit_predict():
    """
    Test fitting and predicting with MyRandomForestClassifier on simple data.
    """
    clf = MyRandomForestClassifier(n_trees=3, max_features=1)
    clf.fit(X_simple, y_simple)
    
    # Predict and check if majority class is returned
    predictions = clf.predict(X_simple)
    assert len(predictions) == len(y_simple), "Random Forest predictions have incorrect length."
    assert set(predictions).issubset(set(y_simple)), "Random Forest predictions have invalid classes."

def test_random_forest_unfitted():
    """
    Test that MyRandomForestClassifier raises an exception if predict is called before fitting.
    """
    clf = MyRandomForestClassifier()
    with pytest.raises(ValueError, match="The model has not been fitted yet."):
        clf.predict(X_simple)

def test_random_forest_consistency():
    """
    Test that MyRandomForestClassifier gives consistent results on simple data.
    """
    clf = MyRandomForestClassifier(n_trees=5, max_features=1)
    clf.fit(X_simple, y_simple)
    predictions_1 = clf.predict(X_simple)
    predictions_2 = clf.predict(X_simple)
    assert np.array_equal(predictions_1, predictions_2), "Random Forest predictions are not consistent."

def test_decision_tree_edge_case_pure_class():
    """
    Test MyDecisionTreeClassifier on edge case with all targets the same.
    """
    clf = MyDecisionTreeClassifier()
    y_pure = np.array([1, 1, 1, 1])
    X_pure = np.array([[0], [1], [2], [3]])
    clf.fit(X_pure, y_pure)
    
    # Predict and verify all predictions match the single class
    predictions = clf.predict(X_pure)
    assert np.array_equal(predictions, y_pure), "Decision Tree failed on pure class case."

def test_decision_tree_majority_class():
    """
    Test MyDecisionTreeClassifier on a dataset where majority voting is needed.
    """
    clf = MyDecisionTreeClassifier()
    X_majority = np.array([[0], [1], [2], [3]])
    y_majority = np.array([0, 0, 1, 1])
    clf.fit(X_majority, y_majority)
    
    # Verify the majority class is predicted
    predictions = clf.predict(X_majority)
    assert set(predictions).issubset(set(y_majority)), "Decision Tree failed majority class test."

def test_random_forest_different_features():
    """
    Test Random Forest with max_features=2 on a multi-feature dataset.
    """
    clf = MyRandomForestClassifier(n_trees=3, max_features=2)
    clf.fit(X_multi, y_multi)
    predictions = clf.predict(X_multi)
    assert len(predictions) == len(y_multi), "Random Forest predictions have incorrect length."
    assert set(predictions).issubset(set(y_multi)), "Random Forest predictions have invalid classes."