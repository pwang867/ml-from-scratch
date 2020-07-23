"""
PCA is a popular machine learning algorithm useful for dimension reduction and
information compression. It works by projecting the original data into a lower
dimensional (k) space while preserving as much information as possible (largest
 variances) in the projected dimensions. The projection can be realized using
matrix product C = f(X) = XD and decoding X' = g(C) = XDD_{T}. Because we want to
preserve the information, the decoded data X' should be as close as possilbe to
the original data matrix X. This can be measured using the Frobenius norm of 
(X - X') as a loss, and the optimal D can be obtained by minimizing it. 

Mathematically:
    D = argmin_D {||X - X'||_F2} 
      = argmin_D {||X - XDD_{T}||^2} 
      = argmin_D {(X - XDD_{T})_T dot (X - XDD_{T})}
It can be proved that D is the eigen vectors corresponding to the k largest eigen 
values of X_{T}X

Programmatically, PCA can be computed using the following procedure:
    1. Normalize the data X to have zero mean
    2. Compute covariance matrix of X, cov(X) = X_{T}.dot(X) / (n - 1)
    3. Compute the eigen values and eigen vectors of cov(X)
    4. Sort the eigen vectors reversed using eigen values and select top K vectors.
    5. Project data using C = (X - mean).dot(D) 

Main purpose of using PCA:
    1. Dimension reduction of features to speed up training downstream ML model.
    2. Reduce noise by only keeping most relevant information (principal components).
    3. Make visualization possible for high dimensional data.

PCA should not be used for solving over-fitting issue.
"""

import unittest
import numpy as np


class PCA:
    def __init__(self, n_components = 2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mu = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mu
        cov = np.dot(X_centered.T, X_centered) / (n - 1)
        eigen_vals, eigen_vecs = np.linalg.eig(cov)
        sorted_indexes = eigen_vals.argsort()[::-1][:self.n_components]
        self.eigen_vecs = eigen_vecs[:, sorted_indexes]
        self.eigen_vals = eigen_vals[sorted_indexes]
        self.total_vars = eigen_vals.sum()
        self.explained_vars = self.eigen_vals
        self.explained_vars_ratio = self.eigen_vals / self.total_vars
        return self

    def transform(self, X):
        return np.dot(X - self.mu, self.eigen_vecs)


class TestPCA(unittest.TestCase):
    def test(self):
        " validate result with sklearn.decomposition.PCA "
        from sklearn.decomposition import PCA as skPCA
        X = np.random.normal(3.2, 5.1, size=(20, 8))
        pca = PCA(3).fit(X)
        skpca = skPCA(3).fit(X)
        output = skpca.transform(X)
        self.assertTrue(np.allclose(np.abs(pca.transform(X)), np.abs(output)), "Should be equal")

if __name__ == "__main__":
    unittest.main()