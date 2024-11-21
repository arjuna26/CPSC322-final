"""
classifiers.py
@author arjuna26
"""

import numpy as np
import random
from .utils import bootstrap_sample

class MyDecisionTreeClassifier:
    def __init__(self):
        # Initialization for the decision tree
        pass

    def fit(self, X, y):
        # Train the decision tree on X and y
        pass

    def predict(self, X):
        # Predict classes for X
        pass


class MyRandomForestClassifier:
    def __init__(self, n_trees=10, max_features=2):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def _select_attributes(self, available_attributes):
        return random.sample(available_attributes, self.max_features)

    def _build_tree(self, X, y):
        """
        Builds a single decision tree using bootstrapped data and random feature selection.
        """
        tree = MyDecisionTreeClassifier()
        selected_attributes = self._select_attributes(list(range(X.shape[1])))
        X_selected = X[:, selected_attributes]
        tree.fit(X_selected, y)
        return tree, selected_attributes

    def _bootstrap_sample(self, X, y):
        return bootstrap_sample(X, y)

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            tree, selected_attributes = self._build_tree(X_bootstrap, y_bootstrap)
            self.trees.append((tree, selected_attributes))

    def predict(self, X):
        # Implement majority voting using the trained trees
        pass
