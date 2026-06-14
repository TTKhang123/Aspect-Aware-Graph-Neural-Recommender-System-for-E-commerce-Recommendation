from __future__ import annotations

import numpy as np
import torch


def build_normalized_adj(
    n_users: int,
    n_items: int,
    users: np.ndarray,
    items: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Build LightGCN normalized adjacency for a user-item bipartite graph.

    A = [[0, R], [R^T, 0]]
    norm_A = D^{-1/2} A D^{-1/2}
    """
    users = users.astype(np.int64)
    items = items.astype(np.int64)
    item_nodes = items + n_users

    row = np.concatenate([users, item_nodes])
    col = np.concatenate([item_nodes, users])
    data = np.ones(len(row), dtype=np.float32)

    n_nodes = n_users + n_items
    degree = np.bincount(row, weights=data, minlength=n_nodes).astype(np.float32)
    degree[degree == 0] = 1.0
    norm_data = data / np.sqrt(degree[row] * degree[col])

    indices = torch.from_numpy(np.vstack([row, col]).astype(np.int64))
    values = torch.from_numpy(norm_data.astype(np.float32))
    adj = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))
    adj = adj.coalesce().to(device)
    return adj
