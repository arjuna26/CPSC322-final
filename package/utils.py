"""
utils.py
@author arjuna26
"""

import numpy as np
import random

def bootstrap_sample(X, y):
    """
    Generate a bootstrap sample from the dataset.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        tuple: Bootstrapped (X, y) datasets.
    """
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


# =============================================================================

