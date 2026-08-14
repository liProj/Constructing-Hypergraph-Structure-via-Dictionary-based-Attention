import unittest

import numpy as np

from da_hypergraph.attention import DictionaryAttention
from da_hypergraph.hypergraph import build_hypergraph, propagation_matrix


class CoreTests(unittest.TestCase):
    def test_attention_is_nonnegative_and_preserves_support(self):
        rng = np.random.default_rng(1)
        x = rng.random((20, 8))
        h, centroids = build_hypergraph(x, "knn", k=3)
        weighted = DictionaryAttention(n_components=5, max_iter=2).transform(x, h, centroids)
        self.assertTrue(np.all(weighted >= 0))
        self.assertTrue(np.array_equal(weighted != 0, h != 0))

    def test_propagation_is_symmetric_and_finite(self):
        rng = np.random.default_rng(2)
        x = rng.random((16, 5))
        h, _ = build_hypergraph(x, "knn", k=4)
        theta = propagation_matrix(h)
        self.assertTrue(np.isfinite(theta).all())
        self.assertTrue(np.allclose(theta, theta.T))


if __name__ == "__main__":
    unittest.main()
