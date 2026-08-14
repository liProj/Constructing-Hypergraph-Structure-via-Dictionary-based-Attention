from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import MiniBatchDictionaryLearning


def neighborhood_incidence(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Create one node-centred hyperedge from each closed citation neighborhood."""
    n = adjacency.shape[0]
    support = adjacency.copy().astype(np.float64)
    support.data[:] = 1.0
    return (support + sp.eye(n, format="csr")).astype(bool).astype(np.float64).tocsc()


def global_dictionary_attention(
    features: sp.csr_matrix,
    incidence: sp.csc_matrix,
    *,
    n_components: int = 50,
    alpha: float = 2 ** -4,
    max_iter: int = 30,
    random_state: int = 0,
    eps: float = 1e-8,
    weight_floor: float = 0.5,
) -> sp.csr_matrix:
    """Shared-dictionary interpretation of Equations (4)--(6).

    This graph-scale implementation learns one dictionary over all vertices,
    then computes centroid/member code similarities within every hyperedge.
    The paper does not state whether B is shared globally or refit per edge;
    this scalable choice is therefore labelled as an independent variant.
    """
    dense = features.toarray().astype(np.float32)
    learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        batch_size=256,
        transform_algorithm="threshold",
        transform_alpha=alpha,
        random_state=random_state,
        n_jobs=-1,
    )
    codes = learner.fit_transform(dense)
    codes /= np.maximum(np.linalg.norm(codes, axis=1, keepdims=True), eps)

    weighted = incidence.copy().astype(np.float64)
    for edge in range(weighted.shape[1]):
        start, end = weighted.indptr[edge], weighted.indptr[edge + 1]
        members = weighted.indices[start:end]
        centroid = edge
        similarities = np.clip(codes[members] @ codes[centroid], 0.0, 1.0)
        # A floor avoids disconnecting nodes whose global sparse codes have
        # disjoint support. This safeguard is explicit because the paper does
        # not define how non-positive inner products enter degree matrices.
        weighted.data[start:end] = weight_floor + (1.0 - weight_floor) * similarities
    return weighted.tocsr()


def propagation_sparse(incidence: sp.spmatrix, eps: float = 1e-12) -> sp.csr_matrix:
    h = incidence.tocsr().astype(np.float64)
    dv = np.asarray(h.sum(axis=1)).ravel()
    de = np.asarray(h.sum(axis=0)).ravel()
    dv_inv = sp.diags(1.0 / np.sqrt(np.maximum(dv, eps)))
    de_inv = sp.diags(1.0 / np.maximum(de, eps))
    return (dv_inv @ h @ de_inv @ h.T @ dv_inv).tocsr()
