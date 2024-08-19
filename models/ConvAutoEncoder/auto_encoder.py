import torch
from torch import nn, Tensor
import dgl
from .gat_layer import GATLayer
import logging


def create_fully_connected_graph(num_timesteps: int, num_players: int) -> dgl.DGLGraph:
    num_nodes = num_timesteps * num_players
    g = dgl.graph(
        (
            [i for i in range(num_nodes) for j in range(num_nodes) if i != j],
            [j for i in range(num_nodes) for j in range(num_nodes) if i != j],
        )
    )
    return g


class ConvAutoEncoder(nn.Module):
    """
    ConvAutoEncoder is a convolutional autoencoder for a single player
    """

    def __init__(self, timesteps: int, kernel_size: int = 3) -> None:
        super(ConvAutoEncoder, self).__init__()
        self.gat = GATLayer(2, 4)
        self.conv1d = nn.Conv1d(
            in_channels=timesteps,
            out_channels=timesteps,
            kernel_size=kernel_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(timesteps * 2, timesteps * 8),
            nn.ReLU(),
            nn.Linear(timesteps * 8, timesteps * 2),
        )

    @staticmethod
    def from_checkpoint(path: str, eval: bool = True) -> "ConvAutoEncoder":
        try:
            model: torch.nn.Module = ConvAutoEncoder(timesteps=12, kernel_size=3)
            weights = torch.load(path)
            model.load_state_dict(weights)
            return model.eval() if eval else model
        except:
            logging.warning(f"Could not load model from {path}, use default model")
            return ConvAutoEncoder(timesteps=12, kernel_size=3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, timesteps, coords = x.shape
        g = create_fully_connected_graph(num_timesteps=timesteps, num_players=1).to("cuda")
        out_batches: list[Tensor] = []
        encoder_out: list[Tensor] = []

        for b in range(batch_size):
            g.ndata["feat"] = x[b].reshape(-1, coords)
            out_gat = self.gat(g, g.ndata["feat"])
            out_gat = out_gat.reshape(timesteps, -1)
            out_conv = self.conv1d(out_gat.unsqueeze(0))
            encoder_out.append(out_conv)
            out_decoded = self.decoder(out_conv.flatten())
            out_decoded = out_decoded.reshape(timesteps, coords)
            out_batches.append(out_decoded)

        return torch.stack(encoder_out), torch.stack(out_batches)
