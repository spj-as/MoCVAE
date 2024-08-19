from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.enc_mean = nn.Linear(hidden_dim, out_dim)
        self.enc_logvar = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.enc(x)
        mean = self.enc_mean(x)
        logvar = self.enc_logvar(x)
        return mean, logvar
