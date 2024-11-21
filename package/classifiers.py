"""
classifiers.py
@author arjuna26
date: 11-20-2024
"""

import numpy as np
import random
from .utils import bootstrap_sample


class MyDecisionTreeClassifier:
    """
    A simple implementation of a decision tree classifier.
    """
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        # Implement a decision tree fitting algorithm (e.g., ID3 with entropy)
        pass

    def predict(self, X):
        # Implement prediction logic based on the fitted tree
        pass


class MyRandomForestClassifier:
    """
    A basic implementation of a Random Forest Classifier.
    """
    def __init__(self, n_trees=10, max_features=2):
        """
        Args:
            n_trees (int): Number of decision trees to build in the forest.
            max_features (int): Number of features to consider at each split.
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []  # List to hold (tree, selected_attributes)
        self.classes = None

    def _select_attributes(self, available_attributes):
        """
        Randomly selects a subset of attributes for splitting.

        Args:
            available_attributes (list): List of available attribute indices.

        Returns:
            list: Randomly selected attribute indices.
        """
        return random.sample(available_attributes, self.max_features)

    def _build_tree(self, X, y):
        """
        Builds a single decision tree using bootstrapped data and random feature selection.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.

        Returns:
            tuple: Trained decision tree and selected attributes.
        """
        tree = MyDecisionTreeClassifier()
        selected_attributes = self._select_attributes(list(range(X.shape[1])))
        X_selected = X[:, selected_attributes]
        tree.fit(X_selected, y)
        return tree, selected_attributes

    def _bootstrap_sample(self, X, y):
        """
        Generates a bootstrap sample of the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            tuple: Bootstrapped feature matrix and target vector.
        """
        return bootstrap_sample(X, y)

    def fit(self, X, y):
        """
        Fits the random forest model to the data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target vector.
        """
        self.classes = np.unique(y)
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            tree, selected_attributes = self._build_tree(X_bootstrap, y_bootstrap)
            self.trees.append((tree, selected_attributes))

    def _majority_vote(self, predictions):
        """
        Determines the class prediction using majority voting.

        Args:
            predictions (list): List of class predictions from trees.

        Returns:
            int/float: Predicted class.
        """
        counts = np.bincount(predictions)
        return np.argmax(counts)

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.trees:
            raise ValueError("The model has not been fitted yet.")

        predictions = []
        for instance in X:
            tree_votes = []
            for tree, selected_attributes in self.trees:
                instance_subset = instance[selected_attributes]
                tree_votes.append(tree.predict([instance_subset])[0])
            predictions.append(self._majority_vote(tree_votes))
        return np.array(predictions)
