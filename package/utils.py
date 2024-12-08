"""
utils.py
@author arjuna26
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


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

def preprocess_data(data: pd.DataFrame):
    # Drop the first empty column and 'notes' column
    data = data.drop(columns=[data.columns[0], 'notes'])
    data = data.drop(columns=['date', 'time', 'comp', 'round', 'referee', 'match report'])
    
    # Fill missing values in numeric columns with the median and categorical with mode
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Convert categorical columns to numerical
    categorical_cols = ['day', 'venue', 'captain', 'formation', 'opp formation', 'team', 'opponent', 'result']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Separate features (X) and target (y)
    X = data.drop(columns=['result'])
    y = data['result']

    # Feature scaling for numeric columns
    numerical_cols = ['gf', 'ga', 'xg', 'xga', 'poss', 'attendance', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50) #42 gives 98.9% accuracy
    
    return X_train, X_test, y_train, y_test
