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

s(i) = (b(i) - a(i)) / max{a(i), b(i)} if |Ci| > 1
s(i) = 0 if |Ci| == 1

a(i) measures how similar point i is to its own cluster, it is defined as
the average distance it is to the other points in the same cluster as i.
    a(i) = sum(dist(i, j)) / (|Ci| - 1))

b(i) measures how dissimilar point i is to points in other cluster.
    b(i) = sum(dist(i, j)) / |Cj| for j != i


3. Cross Validation
Use Silhouette or other metrics to grid search for best k value.


4. Extension to categorical features
    4.1 Problem with convert categorical feature to one hot encoded vector? 
        The main problem is that the categorical features doesn't have a 
        well defined scale and thus not suitable for calculating Euclidean 
        distance. We need to find another way to measure how similar or how
        far two points are from each other.
    4.2 
"""

import numpy as np


class KMeans:
    def __init__(self, k=2, tol=1e-6, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, size=self.k), :]
        for i in range(self.max_iter):
            cluster_ids = self.predict(X)
            new_centroids = np.zeros(self.centroids.shape)
            for c in range(self.k):
                indexes = [_ for _ in range(n_samples) if cluster_ids[_] == c]
                if len(indexes) == 0:
                    new_centroids[c, :] = self.centroids[c, :]
                else:
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
        return np.sum((X - self.centroids[cluster_ids, :]) ** 2)

    def silhouette(self, X):
        if not hasattr(self, "centroids"):
            self.fit(X)
        n_samples = X.shape[0]
        cluster_ids = self.predict(X)
        D = self.calc_dist(X, X)
        def s(i):
            c = cluster_ids[i]
            Ci = [_ for _ in range(n_samples) if cluster_ids[_] == c]
            if len(Ci) == 1:
                return 0.0
            a = np.sum(D[i, Ci] / (len(Ci) - 1))
            b = float("inf")
            for ck in range(self.k):
                if ck != c:
                    Ck = [_ for _ in range(n_samples) if cluster_ids[_] == ck]
                    b = min(b, np.sum(D[i, Ck]) / len(Ck))
            return (b - a) / max(a, b)
        return np.mean([s(i) for i in range(n_samples)])


def test():
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans as skKmeans
    from sklearn.metrics import silhouette_score

    n_features = 4
    n_clusters = 5
    cluster_std = 1.2
    n_samples = 200
    X = make_blobs(n_samples=n_samples, n_features=n_features, 
        centers=n_clusters, cluster_std=cluster_std)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[0])
    plt.style.use("ggplot")

    wss = []
    silhouette = []
    sk_wss = []
    sk_silhouette = []
    for k in range(2, 11):
        kmeans = KMeans(k).fit(X_scaled)
        skkmeans = skKmeans(k).fit(X_scaled)
        wss.append(kmeans.wss(X_scaled))
        silhouette.append(kmeans.silhouette(X_scaled))
        sk_wss.append(-skkmeans.score(X_scaled))
        sk_silhouette.append(silhouette_score(X_scaled, skkmeans.predict(X_scaled)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(np.arange(2, 11), wss, "ro-", alpha=0.5, label="from scratch")
    ax1.plot(np.arange(2, 11), sk_wss, "bo-", alpha=0.5, label="from sklearn")
    ax1.legend(fontsize=14)
    ax1.set_xlabel("k", fontsize=14)
    ax1.set_ylabel("wss", fontsize=14)

    ax2.plot(np.arange(2, 11), silhouette, "ro-", alpha=0.5, label="from scratch")
    ax2.plot(np.arange(2, 11), sk_silhouette, "bo-", alpha=0.5, label="from sklearn")
    ax2.legend(fontsize=14)
    ax2.set_xlabel("k", fontsize=14)
    ax2.set_ylabel("silhouette", fontsize=14)
    fig.show()

if __name__ == "__main__":
    test()