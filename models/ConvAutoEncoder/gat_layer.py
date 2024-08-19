import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dgl import DGLGraph


class GATLayer(nn.Module):
    """
    Reference
    ---------
    https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html?highlight=gat
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def _edge_attention(self, edges) -> dict[str, Tensor]:
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a_input = self.attn_fc(z2)
        return {"e": F.leaky_relu(a_input)}

    def _message_func(self, edges) -> dict[str, Tensor]:
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def _reduce_func(self, nodes) -> dict[str, Tensor]:
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g: DGLGraph, h: Tensor) -> Tensor:
        z = self.fc(h)
        g.ndata["z"] = z
        g.apply_edges(self._edge_attention)
        g.update_all(self._message_func, self._reduce_func)
        return g.ndata.pop("h")
