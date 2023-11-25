import numpy as np
from sklearn.metrics import pairwise_distances


def featurenormalize(feature_vector):
    # Normalize the feature vector to have unit length
    if np.isnan(feature_vector).any():
        # Handle the presence of NaN values (e.g., replace with zeros)
        feature_vector = np.nan_to_num(feature_vector)

    # Calculate the L2 norm of the feature_vector
    norm = np.linalg.norm(feature_vector)
    return feature_vector / norm if norm != 0 else feature_vector

def euclidean_distance_calculator(x, y):
    return np.linalg.norm(x - y)


def get_pairwise_dissimilarities(X):
    X_normalized = featurenormalize(X)
    dissimilarities = pairwise_distances(X_normalized) ##TODO Implement this.

    return dissimilarities

def pairwise_dissimilarities(X):
    pass

def calculate_stress(X_data):
    pass

### Calculating Pairwise Distances
### One hot decoding
### Stress Calculation
### PCA
### Maximum Likelihood estimation.