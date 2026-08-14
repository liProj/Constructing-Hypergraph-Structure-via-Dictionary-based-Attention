"""Independent reproduction of dictionary-based attention for hypergraphs."""

from .attention import DictionaryAttention
from .hypergraph import build_hypergraph, hypergraph_label_propagation

__all__ = ["DictionaryAttention", "build_hypergraph", "hypergraph_label_propagation"]
