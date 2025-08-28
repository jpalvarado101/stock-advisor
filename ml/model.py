import torch, torch.nn as nn


class GRUGauss(nn.Module):
    def __init__(self, n_feat: int, hidden: int=64):
        super().__init__()
        self.gru = nn.GRU(input_size=n_feat, hidden_size=hidden, batch_first=True)
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logvar = nn.Linear(hidden, 1)
        self.softplus = nn.Softplus()


def forward(self, x):
    _, h = self.gru(x)
    h = h[-1]
    mu = self.head_mu(h).squeeze(-1)
    logvar = self.head_logvar(h).squeeze(-1)
    var = self.softplus(logvar) + 1e-6
    return mu, var