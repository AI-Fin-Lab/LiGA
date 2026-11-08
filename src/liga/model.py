"""Example covariate-aware forecast backbone; replace with TFT/TimesFM etc. for paper parity."""
import torch
from torch import nn


class MLPForecaster(nn.Module):
    def __init__(self, context: int, horizon: int, num_covariates: int, hidden=128, dropout=0.1):
        super().__init__()
        input_dim = context * (1 + num_covariates + horizon + 1)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon))

    def forward(self, sales, covariates, ca_features, availability):
        # Each tensor includes a batch dimension; sales [B,T], cov [B,T,M],
        # ca_features [B,T,F], availability [B,T,1].
        batch = sales.shape[0]
        combined = torch.cat([sales.unsqueeze(-1), covariates, ca_features, availability], -1)
        return self.net(combined.reshape(batch, -1))
