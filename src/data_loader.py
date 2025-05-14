from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os


# Poprawione dane treningowe
def preprocessing_first():
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42
    )
    y = 2 * y - 1  # zamiana etykiet na {-1, +1}
    X = StandardScaler().fit_transform(X)
    return X, y

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_svm_file(file_path, zero_based=True):
    """
    Load data from SVMlight/LIBLINEAR format file.

    Args:
        file_path: Path to the .binary file
        zero_based: Whether feature indices start at 0 (True) or 1 (False)

    Returns:
        X: scipy.sparse.csr_matrix of features
        y: numpy array of labels
    """
    labels = []
    rows = []
    cols = []
    data = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            labels.append(float(parts[0]))

            for feat in parts[1:]:
                idx, val = feat.split(':')
                idx = int(idx) - (0 if zero_based else 1)
                rows.append(i)
                cols.append(idx)
                data.append(float(val))

    # Create sparse matrix
    X = csr_matrix((data, (rows, cols)))
    y = np.array(labels)

    return X, y


def preprocess_data(X, y, test_size=0.2, scale=True):
    """
    Split and scale data.

    Args:
        X: Feature matrix (sparse or dense)
        y: Labels
        test_size: Proportion for test split
        scale: Whether to standardize features

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if scale:
        scaler = StandardScaler(with_mean=False)  # Preserve sparsity (dividing by sd)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_dataset(file_path):
    """Combined loader and preprocessor"""
    X, y = load_svm_file(file_path)
    return preprocess_data(X, y)
