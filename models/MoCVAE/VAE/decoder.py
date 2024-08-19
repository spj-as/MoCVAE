from torch import nn, Tensor


class Decoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.dec_mean = nn.Linear(hidden_dim, out_dim)
        self.dec_logvar = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.dec(x)
        mean = self.dec_mean(x)
        logvar = self.dec_logvar(x)
        return mean, logvar
