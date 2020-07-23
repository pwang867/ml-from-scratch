"""
Kmeans is a unsupervised machine learning algorithms. Given the number k and 
a set of training examples, it divides the data into k different clusters 
that are close to each other.

The Kmeans algorithm can be trained using the following procedure.
    1. Initialize k different centroids (u1, u2, ..., uk) randomly
        Kmeans may run into local optimals, to avoid it we can try multiple
        random initilizations and choose the one with smallest converged loss.
    2. Compute distances between each training example to the centoids, and
       assign the closest centroid id to each training example.
    3. Update centroids using the training examples that have the 
       corresponding cluster id.
    4. Repeat step 2 & 3 until converge. Convergence can be measured using 
       within total cluster variance: Var(t+1) - Var(t) < epsilon.
       var = sum((xij - cj)^2)

How to choose the optimal number of clusters?

1. Elbow method
Varying the number of cluster K and calculate within cluster sum of squared
errors (WSS). Plot WSS vs K and select the best K as the point where WSS curve
starts to turn flat (elbow point).


2. Silhouette method
The silhouette value measures how similar a point is to its own cluster 
(cohesion) compared to other clusters (separation).

s(i) = (a(i) - b(i)) / max{a(i), b(i)} if |Ci| > 1
s(i) = 0 if |Ci| == 1

a(i) measures how similar point i is to its own cluster, it is defined as
the average distance it is to the other points in the same cluster as i.
    a(i) = sum(dist(i, j)) / (|Ci| - 1))

b(i) measures how dissimilar point i is to points in other cluster.
    b(i) = sum(dist(i, j)) / |Cj|
"""

import numpy as np


class KMeans:
    def __init__(self, k=2, tol=0.00001, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = 100

    def fit(self, X):
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, size=self.k), :]
        for i in range(self.max_iter):
            cluster_ids = self.predict(X)
            new_centroids = np.zeros(self.centroids.shape)
            for c in range(self.k):
                new_centroids[c, :] = X[[_ for _ in range(n_samples) if cluster_ids[_]==c], :].mean(axis=0)
            if np.sum((self.centroids - new_centroids) ** 2) <= self.tol:
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        dist = self.calc_dist(X, self.centroids)
        cluster_ids = np.argmin(dist, axis=1)
        return cluster_ids

    def calc_dist(self, X, Y):
        """ 
        compute pairwise squared Euclidean distance between X and Y 
        X shape is (m, p) and Y shape is (n, p), the results D should be shape 
        (m, n). 
        Dij = sum_k(Xik - Yjk) ** 2 
            = sum_k(Xik**2 + Yjk**2 - 2 * Xik * Yjk)
        """
        P = np.add.outer(np.sum(X**2, axis=1), np.sum(Y**2, axis=1))
        N = np.dot(X, Y.T)
        return P - 2 * N

    def wss(self, X):
        """ within cluster sum of squares """
        if not hasattr(self, "centroids"):
            self.fit(X)
        cluster_ids = self.predict(X)
        return np.sum((X - self.centroids[cluster_ids, :]) ** 2) / X.shape[0]


def test():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler

    n_features = 4
    n_clusters = 5
    cluster_std = 1.2
    n_samples = 200
    X = make_blobs(n_samples=n_samples, n_features=n_features, 
        centers=n_clusters, cluster_std=cluster_std)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[0])
    plt.style.use("ggplot")

    scores = []
    for k in range(2, 11):
        kmeans = KMeans(k).fit(X_scaled)
        scores.append(kmeans.wss(X_scaled))
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.plot(np.arange(2, 11), scores, "ro-", alpha=0.5)
    fig.show()

if __name__ == "__main__":
    test()