"""Graph-supported historical continuation covariate augmentation (Eqs. 8–11)."""
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .config import LiGAConfig
from .data import Product
from .graph import HistoricalGraph, LifecycleGraphBuilder


class CovariateAugmenter(nn.Module):
    def __init__(self, cfg: LiGAConfig, builder: LifecycleGraphBuilder):
        super().__init__()
        self.cfg = cfg
        self.builder = builder
        f = cfg.horizon; d = cfg.embedding_dim
        self.relation_fusion = nn.Sequential(nn.Linear(2*f+2, 64), nn.ReLU(), nn.Linear(64, f))
        self.att_key = nn.Linear(d, 64)
        self.att_history = nn.Linear(f, 64)
        self.att_vector = nn.Linear(64, 1)
        self.predictor = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Linear(64, f))

    def _entry_nodes(self, graph: HistoricalGraph, target_key: torch.Tensor,
                     target_r: int, view: int, relation: str) -> List[int]:
        """Top-k lifecycle aligned and anchor-deduplicated entry nodes."""
        if not graph.anchors:
            return []
        v = graph.views
        with torch.no_grad():
            q = self.builder.project(target_key, relation)
            choices = []
            for ai, a in enumerate(graph.anchors):
                if abs(a.r - target_r) > self.cfg.query_tolerance:
                    continue
                wi_choices = [view] if relation == 'intra' else [wi for wi in range(v) if wi != view]
                for wi in wi_choices:
                    if bool(graph.valid[ai, wi]):
                        score = float((q * self.builder.project(graph.keys[ai, wi], relation)).sum())
                        choices.append((score, ai, ai*v+wi))
            choices.sort(reverse=True)
            dedup = set(); result = []
            for _, ai, node in choices:
                if ai not in dedup:
                    dedup.add(ai); result.append(node)
                if len(result) >= self.cfg.ca_k:
                    break
            return result

    def _neighbor_evidence(self, graph: HistoricalGraph, node: int, rel: str):
        """Eq. (9), first-hop softmax and normalized two-hop path products."""
        zero = graph.continuations.new_zeros(self.cfg.horizon)
        first = graph.edges[rel].get(node, [])
        if not first:
            return zero, False
        weights = torch.softmax(torch.tensor([w for _, w in first],
                           device=zero.device) / self.cfg.graph_temperature, dim=0)
        first_total = sum((weights[i] * graph.continuations[graph.decode(m)[0]]
                           for i, (m, _) in enumerate(first)), zero.clone())
        first_anchors = {graph.decode(m)[0] for m, _ in first}
        origin_anchor = graph.decode(node)[0]
        paths = {}
        for i, (m, _) in enumerate(first):
            second_edges = graph.edges[rel].get(m, [])
            if not second_edges:
                continue
            second_weights = torch.softmax(torch.tensor([w for _, w in second_edges],
                        device=zero.device) / self.cfg.graph_temperature, dim=0)
            for j, (h, _) in enumerate(second_edges):
                aid = graph.decode(h)[0]
                if aid == origin_anchor or aid in first_anchors:
                    continue
                paths[aid] = paths.get(aid, 0.) + float(weights[i] * second_weights[j])
        if paths:
            z = sum((value * graph.continuations[aid] for aid, value in paths.items()), zero.clone())
            second_total = z / max(sum(paths.values()), 1e-12)
        else:
            second_total = first_total  # no two-hop evidence: do not dampen usable first-hop
        beta = self.cfg.two_hop_weight
        return (1-beta)*first_total + beta*second_total, True

    def at_position(self, product: Product, r: int, graph: HistoricalGraph):
        """Return augmented horizon features [F] and 1 if supported, else zeros and 0."""
        f = self.cfg.horizon
        if not graph.anchors:
            return graph.continuations.new_zeros(f), False
        device = graph.keys.device
        key, valid = self.builder.encoder.encode_views(product.sales, r, self.cfg.views, device)
        evidence = []; scores = []
        for vi in range(len(self.cfg.views)):
            if not bool(valid[vi]):
                continue
            values = []; flags = []
            for rel in ('intra', 'inter'):
                entries = self._entry_nodes(graph, key[vi], r, vi, rel)
                collected = [self._neighbor_evidence(graph, node, rel) for node in entries]
                usable = [value for value, available in collected if available]
                values.append(torch.stack(usable).mean(dim=0) if usable else key.new_zeros(f))
                flags.append(float(bool(usable)))
            if not any(flags):
                continue
            inp = torch.cat([values[0], values[1], key.new_tensor(flags)])
            h = self.relation_fusion(inp)
            evidence.append(h)
            scores.append(self.att_vector(torch.tanh(self.att_key(key[vi]) + self.att_history(h))).squeeze())
        if not evidence:
            return key.new_zeros(f), False
        weights = torch.softmax(torch.stack(scores), dim=0)
        fused = sum((weight*h for weight, h in zip(weights, evidence)), evidence[0].new_zeros(f))
        return self.predictor(fused), True

    def augment_window(self, product: Product, t: int, graph: HistoricalGraph):
        """Return [T,F] and [T,1], each position uses only its observed prefix.

        The historical repository itself is frozen at target calendar cutoff t.
        """
        features = []; masks = []
        for r in range(t-self.cfg.context+1, t+1):
            feat, present = self.at_position(product, r, graph)
            features.append(feat)
            masks.append(float(present))
        return torch.stack(features), torch.tensor(masks, device=features[0].device).unsqueeze(-1)

    def auxiliary_loss(self, product: Product, r: int, graph: HistoricalGraph):
        """Supervise c_aug(r) against observed x[r:r+F], training samples only."""
        if r + self.cfg.horizon > product.length:
            raise ValueError('Unobserved target continuation')
        features, available = self.at_position(product, r, graph)
        if not available:
            return None
        labels = features.new_tensor(product.sales[r:r+self.cfg.horizon].copy())
        return F.mse_loss(features, labels)
