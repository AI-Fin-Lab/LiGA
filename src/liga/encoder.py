"""Causal multi-view encoder. Pluggable alternative to paper's pretrained TSFM.

The manuscript specifies a shared TSFM encoder but not its checkpoint or exact
pooling recipe. This self-contained reference uses a lightweight shared CNN.
"""
from typing import Sequence, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def causal_aggregate(values: np.ndarray, scale: int) -> np.ndarray:
    """Trailing rolling means with no future inputs; preserves sequence length."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    sums = np.cumsum(np.r_[0., x.astype(np.float64)])
    ix = np.arange(len(x)); starts = np.maximum(ix - scale + 1, 0)
    return ((sums[ix + 1] - sums[starts]) / (ix - starts + 1)).astype(np.float32)


class SharedMultiViewEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                     nn.Linear(64, embedding_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        # Scale-normalize each prefix; avoids training data statistics in query encoding.
        scale = sequence.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.normalize(self.network((sequence / scale).unsqueeze(1)), dim=-1)

    def encode_views(self, sales: np.ndarray, t: int, views: Sequence[Tuple[int, int]],
                     device='cpu'):
        """Returns [V,d] embeddings and [V] availability, for observed prefix sales[:t]."""
        embeddings = []; mask = []
        d = self.network[-1].out_features
        for length, scale in views:
            if t < length:
                embeddings.append(torch.zeros(d, device=device)); mask.append(False)
            else:
                seq = causal_aggregate(sales[t-length:t], scale)
                embeddings.append(self.forward(torch.as_tensor(seq.copy(), device=device)).squeeze(0))
                mask.append(True)
        return torch.stack(embeddings), torch.tensor(mask, device=device, dtype=torch.bool)
