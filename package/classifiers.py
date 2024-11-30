"""
classifiers.py
@author arjuna26
date: 11-20-2024
"""

import numpy as np
import random
from .utils import bootstrap_sample
from collections import Counter 


import numpy as np
from collections import Counter

class MyDecisionTreeClassifier:
    """
    A simple implementation of a decision tree classifier using ID3 with entropy.
    """
    def __init__(self):
        self.tree = None

    def _entropy(self, y):
        """
        Calculate the entropy of a dataset.

        Args:
            y (array-like): Target labels.

        Returns:
            float: Entropy of the target labels.
        """
        counts = Counter(y)
        total = len(y)
        return -sum((count / total) * np.log2(count / total) for count in counts.values())

    def _information_gain(self, X_col, y, split_value):
        """
        Calculate the information gain for a split on a specific feature.

        Args:
            X_col (array-like): Column of feature values.
            y (array-like): Target labels.
            split_value (float): Split point for the feature.

        Returns:
            float: Information gain of the split.
        """
        left_mask = X_col <= split_value
        right_mask = X_col > split_value

        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        total_entropy = self._entropy(y)

        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return total_entropy - weighted_entropy

    def _best_split(self, X, y):
        """
        Find the best feature and split value to split on.

        Args:
            X (np.ndarray): Feature matrix.
            y (array-like): Target labels.

        Returns:
            tuple: (best_feature, best_split_value, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_split_value = None

        for feature_idx in range(X.shape[1]):
            X_col = X[:, feature_idx]
            unique_values = np.unique(X_col)

            for split_value in unique_values:
                gain = self._information_gain(X_col, y, split_value)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_split_value = split_value

        return best_feature, best_split_value, best_gain

    def _build_tree(self, X, y):
        """
        Recursively build the decision tree.

        Args:
            X (np.ndarray): Feature matrix.
            y (array-like): Target labels.

        Returns:
            dict: The decision tree structure.
        """
        # Base case: pure set or no features left
        if len(set(y)) == 1:
            return y[0]  # Return the class label
        if X.shape[1] == 0:
            return Counter(y).most_common(1)[0][0]  # Return majority class

        # Find the best split
        best_feature, best_split_value, best_gain = self._best_split(X, y)

        if best_gain == 0:
            return Counter(y).most_common(1)[0][0]  # Return majority class

        # Partition data
        left_mask = X[:, best_feature] <= best_split_value
        right_mask = X[:, best_feature] > best_split_value

        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask])
        right_tree = self._build_tree(X[right_mask], y[right_mask])

        return {
            "feature": best_feature,
            "split_value": best_split_value,
            "left": left_tree,
            "right": right_tree
        }

    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Args:
            X (np.ndarray): Training feature matrix.
            y (array-like): Target labels.
        """
        self.tree = self._build_tree(X, y)

    def _predict_instance(self, instance, tree):
        if isinstance(tree, dict):  # Internal node
            feature = tree.get("feature")
            split_value = tree.get("split_value")

            if instance[feature] <= split_value:
                return self._predict_instance(instance, tree["left"])
            else:
                return self._predict_instance(instance, tree["right"])
        elif isinstance(tree, (int, float)):  # Leaf node
            return tree
        else:
            raise ValueError(f"Unexpected tree node: {tree}")

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.tree is None:
            raise Exception("Tree has not been fitted yet.")  
        return np.array([self._predict_instance(instance, self.tree) for instance in X])



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
