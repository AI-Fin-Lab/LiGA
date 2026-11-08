"""Lifecycle graph; graph edges use context only and cutoff-observed continuations."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .config import LiGAConfig
from .data import Product
from .encoder import SharedMultiViewEncoder


@dataclass(frozen=True)
class Anchor:
    product_id: str
    r: int  # one-based lifecycle position, also observed prefix length


@dataclass
class HistoricalGraph:
    anchors: List[Anchor]
    keys: torch.Tensor  # [A,V,d], detached
    valid: torch.Tensor  # [A,V]
    continuations: torch.Tensor  # [A,F]
    edges: Dict[str, Dict[int, List[Tuple[int, float]]]]  # flattened node -> (node, cosine)
    observed_cutoff: np.datetime64

    @property
    def views(self):
        return self.keys.shape[1]

    def decode(self, node: int):
        return divmod(node, self.views)


class LifecycleGraphBuilder(nn.Module):
    def __init__(self, config: LiGAConfig, encoder: SharedMultiViewEncoder):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.projections = nn.ModuleDict({
            'intra': nn.Linear(config.embedding_dim, config.projection_dim, bias=False),
            'inter': nn.Linear(config.embedding_dim, config.projection_dim, bias=False)})

    def project(self, key: torch.Tensor, relation: str):
        return F.normalize(self.projections[relation](key), dim=-1)

    @torch.no_grad()
    def build(self, products: Dict[str, Product], allowed_ids: List[str],
              cutoff: np.datetime64, excluded_id: str = '', device='cpu') -> HistoricalGraph:
        c = self.config
        anchors = []; keys = []; masks = []; conts = []
        for pid in allowed_ids:
            if pid == excluded_id:
                continue
            p = products[pid]; available = p.observed_length(cutoff)
            for r in range(c.context, available - c.horizon + 1):
                k, mask = self.encoder.encode_views(p.sales, r, c.views, device)
                anchors.append(Anchor(pid, r)); keys.append(k); masks.append(mask)
                conts.append(torch.tensor(p.sales[r:r+c.horizon].copy(), device=device))
        v = len(c.views); d = c.embedding_dim
        if not anchors:
            return HistoricalGraph([], torch.empty(0, v, d, device=device),
                                   torch.empty(0, v, dtype=torch.bool, device=device),
                                   torch.empty(0, c.horizon, device=device),
                                   {'intra': {}, 'inter': {}}, cutoff)
        K = torch.stack(keys).detach(); valid = torch.stack(masks)
        Z = torch.stack(conts)
        edges = {'intra': {}, 'inter': {}}
        for rel in ('intra', 'inter'):
            projected = self.project(K.reshape(-1, d), rel)
            # This reference implementation uses direct sparse candidate enumeration.
            for ai, a in enumerate(anchors):
                eligible = [bi for bi, b in enumerate(anchors)
                            if ai != bi and abs(a.r-b.r) <= c.lifecycle_tolerance]
                for vi in range(v):
                    if not valid[ai, vi]:
                        continue
                    u = ai * v + vi
                    candidates = [bi * v + wi for bi in eligible for wi in range(v)
                                  if bool(valid[bi, wi]) and ((wi == vi) if rel == 'intra' else (wi != vi))]
                    if not candidates:
                        continue
                    sims = projected[candidates] @ projected[u]
                    top = torch.topk(sims, min(c.graph_k, len(candidates)))
                    edges[rel][u] = [(candidates[int(idx)], float(score))
                                     for idx, score in zip(top.indices, top.values)]
        return HistoricalGraph(anchors, K, valid, Z, edges, cutoff)
