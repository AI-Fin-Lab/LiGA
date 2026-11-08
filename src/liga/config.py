from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LiGAConfig:
    context: int = 30
    horizon: int = 7
    sufficient_history: int = 70
    views: Tuple[Tuple[int, int], ...] = ((7, 1), (14, 1), (30, 1), (14, 2))
    embedding_dim: int = 64
    projection_dim: int = 32
    graph_k: int = 5
    ca_k: int = 2
    sa_k: int = 3
    lifecycle_tolerance: int = 7
    query_tolerance: int = 7
    two_hop_weight: float = 0.5
    graph_temperature: float = 0.2
    mixup_alpha: float = 0.4
    seasonal_period: int = 7
    trend_noise_std: float = 0.02
    sa_ratio: float = 0.5
    random_seed: int = 42

    def __post_init__(self):
        assert self.context > 0 and self.horizon > 0
        assert 0 <= self.two_hop_weight <= 1
        assert self.graph_k > 0 and self.ca_k > 0 and self.sa_k > 0
        assert all(l > 0 and s > 0 for l, s in self.views)
