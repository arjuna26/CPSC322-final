import numpy as np

# ---------------------------------------------- Decision tree and Random Forest ----------------------------------------------

class MyDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1:
            return {'label': y[0]}
        if n_samples < self.min_samples_split:
            return {'label': np.bincount(y).argmax()}
        if self.max_depth is not None and depth >= self.max_depth:
            return {'label': np.bincount(y).argmax()}

        best_gini = float('inf')
        best_split = None
        best_left_y = None
        best_right_y = None
        best_left_X = None
        best_right_X = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = self._calculate_gini(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)
                    best_left_y = left_y
                    best_right_y = right_y
                    best_left_X = X[left_mask]
                    best_right_X = X[right_mask]

        if best_split is None:
            return {'label': np.bincount(y).argmax()}

        left_tree = self._build_tree(best_left_X, best_left_y, depth + 1)
        right_tree = self._build_tree(best_right_X, best_right_y, depth + 1)

        return {'feature': best_split[0],
                'threshold': best_split[1],
                'left': left_tree,
                'right': right_tree}

    def _calculate_gini(self, left_y, right_y):
        total_size = len(left_y) + len(right_y)
        left_size = len(left_y) / total_size
        right_size = len(right_y) / total_size

        def gini_impurity(y):
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)

        return left_size * gini_impurity(left_y) + right_size * gini_impurity(right_y)

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if 'label' in tree:
            return tree['label']

        feature_val = sample[tree['feature']]
        if feature_val <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])

class MyRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        # If max_features is 'sqrt', set it to the square root of the number of features
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(X.shape[1]))
        elif self.max_features is None:
            self.max_features = X.shape[1]  # Use all features if not specified

        # Train multiple decision trees
        for _ in range(self.n_estimators):
            # Sample the data
            if self.bootstrap:
                indices = np.random.choice(range(len(X)), size=len(X), replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y

            # Select a random subset of features
            feature_indices = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]

            # Train a decision tree
            tree = MyDecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append((tree, feature_indices))  # Save tree and the features it was trained on

    def predict(self, X):
        # Collect predictions from each tree
        predictions = []
        for tree, feature_indices in self.trees:
            X_selected = X[:, feature_indices]  # Use only the features selected by this tree
            predictions.append(tree.predict(X_selected))
        
        # Majority voting
        predictions = np.array(predictions)
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])  # Majority vote across trees

# ---------------------------------------------------------------------------------------------------------------------------------------