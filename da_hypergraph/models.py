from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def scipy_to_torch(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


class TwoStepIncidenceNetwork(nn.Module):
    """Two vertex-edge propagation layers corresponding to Equation (17)."""

    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(in_features, hidden, bias=False)
        self.second = nn.Linear(hidden, classes, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(propagation, x)
        x = F.relu(self.first(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sparse.mm(propagation, x)
        return F.log_softmax(self.second(x), dim=1)


def train_network(
    features: sp.csr_matrix,
    labels: np.ndarray,
    propagation: sp.csr_matrix,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    hidden: int = 256,
    dropout: float = 0.5,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    epochs: int = 300,
    patience: int = 50,
    random_state: int = 0,
) -> tuple[float, int]:
    torch.manual_seed(random_state)
    x = torch.tensor(features.toarray(), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    theta = scipy_to_torch(propagation)
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    val_idx = torch.tensor(validation_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)
    model = TwoStepIncidenceNetwork(x.shape[1], hidden, int(y.max()) + 1, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state, best_val, stale, best_epoch = None, -1.0, 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, theta)
        loss = F.nll_loss(output[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(x, theta)
            val_accuracy = float((output[val_idx].argmax(1) == y[val_idx]).float().mean())
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(x, theta)
        accuracy = float((output[test_idx].argmax(1) == y[test_idx]).float().mean())
    return accuracy, best_epoch
