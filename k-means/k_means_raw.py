import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.cluster_assignments = None

    def _assign_clusters(self, X):
        X_expanded = X[:, np.newaxis]
        differences = X_expanded - self.centroids
        distances = np.linalg.norm(differences, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = X[self.cluster_assignments == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
        self.centroids = new_centroids

    def fit(self, X):
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids = X[indices]
        for i in range(self.max_iters):
            self.cluster_assignments = self._assign_clusters(X)
            old_centroids = self.centroids.copy()
            self._update_centroids(X)
            if np.all(np.abs(self.centroids - old_centroids) < self.tolerance):
                break
        return self

    def predict(self, X):
        return self._assign_clusters(X)

    def plot_clusters(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.cluster_assignments, cmap="viridis", marker="o", edgecolor="k")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="red", marker="X", s=100, label="Centroids")
        plt.legend()
        plt.title("K-Means Clustering Results")
        plt.show()