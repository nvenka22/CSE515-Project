import numpy as np
import math

from sklearn.metrics.pairwise import euclidean_distances

def PCA(X, n_componenets):
    """
        Method to compute and return Principal compoment analysis of the given matrix X
        Args:
            X(np.array): Original feature matrix for the images.
            n_components(float): Percentage of original features to keep in the latet space.

        Return:
            np.array of the latent dimention corresponding to Image set X. 
    """

    X_mean = X - np.mean(X , axis = 0)
    cov = np.cov(X_mean, rowvar=False)
    eigen_val, eigen_vec = np.linalg.eigh(cov)

    sorted_index = np.argsort(eigen_val)[::-1]
    sorted_eigen_val = eigen_val[sorted_index]
    sorted_eigen_vec = eigen_vec[: ,sorted_index]

    _num_components = math.ceil(n_componenets*sorted_eigen_vec.shape[1])

    _eigen_vec = sorted_eigen_vec[:, 0:_num_components]

    X_red = np.dot(_eigen_vec.transpose(), X_mean.transpose()).transpose()

    return X_red

def calculate_stress(dissimilarities, max_iter = 300, metric = True):


    for it in range(max_iter):
        
        dis = euclidean_distances(X)
        if metric:
            disparities = dissimilarities

        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        if normalized_stress:
            stress = np.sqrt(stress / ((disparities.ravel() ** 2).sum() / 2))
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1.0 / n_samples * np.dot(B, X)

        dis = np.sqrt((X**2).sum(axis=1)).sum()
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print("breaking at iteration %d with stress %s" % (it, stress))
                break
        old_stress = stress / dis