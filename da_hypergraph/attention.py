from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _soft_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


@dataclass
class DictionaryAttention:
    """Equation (4)--(6) dictionary attention with documented safeguards.

    The paper does not state how negative/zero similarities are handled before
    forming degree matrices. ``stabilize=True`` clips cosine-normalized code
    similarities to [eps, 1], which guarantees a valid nonnegative incidence
    matrix. Set it to False to obtain the literal sparse-code inner products.
    """

    n_components: int = 50
    alpha: float = 2 ** -4
    rho: float = 1.0
    dual_step: float = 1.0
    max_iter: int = 12
    tol: float = 1e-5
    eps: float = 1e-8
    stabilize: bool = True
    random_state: int = 0

    def _fit_codes(self, x_edge: np.ndarray, edge_seed: int) -> np.ndarray:
        """Learn a dictionary and codes for one d-by-m hyperedge matrix."""
        d, m = x_edge.shape
        rng = np.random.default_rng(edge_seed)
        picks = rng.integers(0, m, size=self.n_components)
        dictionary = x_edge[:, picks].copy()
        dictionary += rng.normal(0.0, 1e-4, size=dictionary.shape)
        dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), self.eps)

        codes = np.zeros((self.n_components, m), dtype=np.float64)
        auxiliary = codes.copy()
        multiplier = codes.copy()
        previous = np.inf

        for _ in range(self.max_iter):
            lhs = dictionary.T @ dictionary + self.rho * np.eye(self.n_components)
            rhs = dictionary.T @ x_edge + self.rho * auxiliary - multiplier
            codes = np.linalg.solve(lhs, rhs)
            auxiliary = _soft_threshold(codes + multiplier / self.rho, self.alpha / self.rho)
            multiplier += self.dual_step * (codes - auxiliary)

            for atom in range(self.n_components):
                row = codes[atom]
                row_energy = float(row @ row)
                if row_energy <= self.eps:
                    continue
                residual = x_edge - dictionary @ codes + np.outer(dictionary[:, atom], row)
                update = residual @ row
                norm = np.linalg.norm(update)
                if norm > self.eps:
                    dictionary[:, atom] = update / norm

            objective = np.linalg.norm(x_edge - dictionary @ codes, "fro") ** 2
            objective += 2.0 * self.alpha * np.abs(codes).sum()
            if np.isfinite(previous) and 0 <= previous - objective <= self.tol * max(previous, 1.0):
                break
            previous = objective
        return codes

    def transform(
        self,
        features: np.ndarray,
        incidence: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Return an attention-weighted copy of an n-by-e incidence matrix."""
        x = np.asarray(features, dtype=np.float64)
        h = np.asarray(incidence, dtype=np.float64)
        if x.ndim != 2 or h.ndim != 2 or h.shape[0] != x.shape[0]:
            raise ValueError("features must be n-by-d and incidence must be n-by-e")
        if len(centroids) != h.shape[1]:
            raise ValueError("one centroid index is required per hyperedge")

        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, self.eps)
        weighted = np.zeros_like(h)

        for edge in range(h.shape[1]):
            members = np.flatnonzero(h[:, edge] != 0)
            if members.size == 0:
                continue
            centroid = int(centroids[edge])
            where = np.flatnonzero(members == centroid)
            if where.size == 0:
                raise ValueError(f"centroid {centroid} is not in hyperedge {edge}")
            codes = self._fit_codes(x[members].T, self.random_state + 104729 * edge)
            centroid_code = codes[:, int(where[0])]

            if self.stabilize:
                code_norms = np.linalg.norm(codes, axis=0)
                denom = np.maximum(np.linalg.norm(centroid_code) * code_norms, self.eps)
                similarity = (centroid_code @ codes) / denom
                similarity = np.clip(similarity, self.eps, 1.0)
            else:
                similarity = centroid_code @ codes
            weighted[members, edge] = h[members, edge] * similarity
        return weighted
