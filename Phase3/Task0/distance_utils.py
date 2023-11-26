import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


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


def calculate_stress(X_original, X_latent_dim):
    """
        Funtion to find out the average stress between data in original space and feature reduced space.
        Args:
            X_original(np.array): dataset represent in original space.
        Return:
            X_latent_dim(np.array): dataset represented in feature reduced latent space.
    """

    # Ensure X_original and X_latent_dim are numpy arrays
    X_original = np.array(X_original)
    X_latent_dim = np.array(X_latent_dim)

    n_samples = X_original.shape[0]
    stress_sum = 0.0
    distance_sum = 0.0

    for i in tqdm(range(n_samples)):
        for j in range(i + 1, n_samples):
            # Original pairwise dissimilarity
            d_ij = np.linalg.norm(X_original[i] - X_original[j])

            # Euclidean distance in low-dimensional representation
            delta_ij = np.linalg.norm(X_latent_dim[i] - X_latent_dim[j])

            stress_sum += (d_ij - delta_ij)**2
            distance_sum += d_ij**2

    stress = np.sqrt(stress_sum / distance_sum)

    return stress

def pca_mle(X):
    """
    Perform PCA for dimensionality reduction using maximum likelihood estimation.

    Parameters:
     X(np.array): The original dataset.

    Returns:
    - X_reduced(np.array): Feature Reduced dataset.
    - components: numpy array, shape (n_features, n_components)
      The principal components.
    """

    # Ensure X is a numpy array
    X = np.array(X)

    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Determine the number of components to keep (using explained variance)
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Determine the number of components to keep based on a threshold (e.g., 95% variance)
    threshold = 0.99
    num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

    # Select the first num_components columns of eigenvectors
    components = eigenvectors[:, :num_components]

    # Project the data onto the principal components
    X_reduced = np.dot(X_centered, components)

    return X_reduced

