from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _knn_hypergraph(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(features)
    indices = neighbors.kneighbors(return_distance=False)
    h = np.zeros((n, n), dtype=np.float64)
    for center, members in enumerate(indices):
        h[members, center] = 1.0
        h[center, center] = 1.0
    return h, np.arange(n, dtype=int)


def _clustering_hypergraph(
    features: np.ndarray, n_clusters: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit(features)
    h = np.zeros((features.shape[0], n_clusters), dtype=np.float64)
    centroids = np.zeros(n_clusters, dtype=int)
    for cluster in range(n_clusters):
        members = np.flatnonzero(model.labels_ == cluster)
        h[members, cluster] = 1.0
        distances = np.linalg.norm(features[members] - model.cluster_centers_[cluster], axis=1)
        centroids[cluster] = members[np.argmin(distances)]
    return h, centroids


def _representation_hypergraph(features: np.ndarray, k: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge self-representation, followed by top-k coefficient selection."""
    x = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    gram = x @ x.T
    coefficients = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), gram)
    np.fill_diagonal(coefficients, 0.0)
    n = x.shape[0]
    h = np.zeros((n, n), dtype=np.float64)
    for center in range(n):
        selected = np.argpartition(np.abs(coefficients[:, center]), -min(k, n - 1))[-min(k, n - 1):]
        h[selected, center] = 1.0
        h[center, center] = 1.0
    return h, np.arange(n, dtype=int)


def build_hypergraph(
    features: np.ndarray,
    method: str,
    *,
    k: int = 10,
    n_clusters: int | None = None,
    ridge: float = 1e-2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN, clustering, or ridge-representation hyperedges."""
    x = np.asarray(features, dtype=np.float64)
    if method == "knn":
        return _knn_hypergraph(x, k)
    if method == "clustering":
        if n_clusters is None:
            raise ValueError("n_clusters is required for clustering")
        return _clustering_hypergraph(x, n_clusters, random_state)
    if method == "representation":
        return _representation_hypergraph(x, k, ridge)
    raise ValueError(f"unknown hypergraph method: {method}")


def propagation_matrix(incidence: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(incidence, dtype=np.float64)
    vertex_degree = h.sum(axis=1)
    edge_degree = h.sum(axis=0)
    dv = 1.0 / np.sqrt(np.maximum(vertex_degree, eps))
    de = 1.0 / np.maximum(edge_degree, eps)
    return (dv[:, None] * h * de[None, :]) @ (h.T * dv[None, :])


def hypergraph_label_propagation(
    incidence: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    n_classes: int,
    regularization: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form minimizer of Equations (14)--(16)."""
    n = incidence.shape[0]
    y = np.full((n, n_classes), 0.5, dtype=np.float64)
    y[train_indices] = 0.0
    y[train_indices, labels[train_indices]] = 1.0
    laplacian = np.eye(n) - propagation_matrix(incidence)
    scores = np.linalg.solve(laplacian + regularization * np.eye(n), regularization * y)
    return scores.argmax(axis=1), scores
