import torch
import torch.nn as nn
from .gcn_layers import GCNLayer


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
    ):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        return torch.tanh(x)
