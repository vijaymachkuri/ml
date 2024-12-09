import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters, max_iters=100):  # Fixed `__init__` method name
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

# Generate dataset and use KMeans
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans._assign_labels(X)

# Print results
print("Cluster Assignments:", labels)
print("Final Centroids:", kmeans.centroids)
