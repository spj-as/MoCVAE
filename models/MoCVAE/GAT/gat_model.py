import torch
from torch import nn
from typing import List, Any
from .gat_layers import GATLayer


class GAT(nn.Module):
    """Dense version of GAT."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, alpha: float, n_heads: int) -> None:
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList(
            [GATLayer(in_dim, hidden_dim, alpha=alpha, concat=True) for _ in range(n_heads)]
        )
        self.out_att = GATLayer(hidden_dim * n_heads, out_dim, alpha=alpha, concat=False)
        self.bn1 = nn.BatchNorm1d(out_dim, eps=1e-03)

    def forward(self, x: torch.Tensor, adj: Any) -> torch.Tensor:
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        x = self.bn1(x)
        return torch.tanh(x)
