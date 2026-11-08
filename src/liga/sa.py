"""Structure-preserving sample augmentation (Eqs. 12–14).

ResidualGenerator is a swappable component. Default uses a GRU adversarial
residual generator (TimeGAN-inspired), not a claim of exact original TimeGAN.
"""
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .config import LiGAConfig
from .data import Product, Origin
from .graph import HistoricalGraph, LifecycleGraphBuilder


def decompose(x: np.ndarray, period: int):
    x = np.asarray(x, dtype=np.float32)
    # Reflect-padded moving average is used for complete *training* trajectories.
    k = max(3, period | 1)
    left = k // 2
    smooth = np.convolve(np.pad(x, (left, left), mode='edge'),
                         np.ones(k, dtype=np.float32)/k, mode='valid')
    raw_season = x - smooth
    seasonal_template = np.array([raw_season[i::period].mean() for i in range(period)], dtype=np.float32)
    seasonal_template -= seasonal_template.mean()
    season = seasonal_template[np.arange(len(x)) % period]
    residual = x - smooth - season
    return smooth.astype(np.float32), season.astype(np.float32), residual.astype(np.float32)


def phase_shift(context_a: np.ndarray, seasonal_b: np.ndarray, context: int, period: int):
    shifts = [np.mean((context_a[:context] - np.roll(seasonal_b, d)[:context])**2)
              for d in range(period)]
    return int(np.argmin(shifts))


class ResidualGenerator(nn.Module):
    """Small GRU generator and discriminator trained adversarially on residuals.

    This is an engineering approximation, not the full TimeGAN architecture.
    Replace with a separately pretrained TimeGAN implementation for strict paper parity.
    """
    def __init__(self, hidden=32, noise_dim=8):
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = nn.GRU(noise_dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
        self.discriminator = nn.GRU(1, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)

    def sample(self, batch: int, length: int, device='cpu'):
        z = torch.randn(batch, length, self.noise_dim, device=device)
        return self.output(self.generator(z)[0]).squeeze(-1)

    def discriminate(self, series: torch.Tensor):
        seq = series.unsqueeze(-1)
        return self.classifier(self.discriminator(seq)[0][:, -1]).squeeze(-1)

    def fit(self, residuals: np.ndarray, epochs=30, batch_size=32, lr=1e-3, device='cpu'):
        data = torch.tensor(np.asarray(residuals, dtype=np.float32), device=device)
        if data.ndim != 2 or len(data) == 0:
            raise ValueError('Residuals must be a nonempty [N,T+F] array')
        self.to(device)
        opt_g = torch.optim.Adam(list(self.generator.parameters())+list(self.output.parameters()), lr=lr)
        opt_d = torch.optim.Adam(list(self.discriminator.parameters())+list(self.classifier.parameters()), lr=lr)
        for _ in range(epochs):
            order = torch.randperm(len(data), device=device)
            for ids in order.split(batch_size):
                real = data[ids]; fake = self.sample(len(ids), real.shape[1], device)
                opt_d.zero_grad()
                loss_d = F.softplus(-self.discriminate(real)).mean() + F.softplus(self.discriminate(fake.detach())).mean()
                loss_d.backward(); opt_d.step()
                opt_g.zero_grad()
                fake = self.sample(len(ids), real.shape[1], device)
                # Encourage matching temporal smoothness and residual spread.
                loss_g = F.softplus(-self.discriminate(fake)).mean() + \
                    0.1*(fake.std()-real.std()).abs() + \
                    0.05*(torch.diff(fake, dim=1).abs().mean()-torch.diff(real, dim=1).abs().mean()).abs()
                loss_g.backward(); opt_g.step()
        return self


class SampleAugmenter:
    def __init__(self, config: LiGAConfig, builder: LifecycleGraphBuilder,
                 generator: Optional[ResidualGenerator] = None):
        self.cfg = config; self.builder = builder; self.generator = generator
        self.rng = np.random.default_rng(config.random_seed)

    @torch.no_grad()
    def donors(self, target: Product, t: int, graph: HistoricalGraph) -> List[int]:
        if not graph.anchors:
            return []
        device = graph.keys.device; c = self.cfg
        query, mask = self.builder.encoder.encode_views(target.sales, t, c.views, device)
        similarities = []
        for ai, anchor in enumerate(graph.anchors):
            if anchor.product_id == target.product_id or abs(anchor.r-t) > c.lifecycle_tolerance:
                continue
            joint = mask & graph.valid[ai]
            if not joint.any():
                continue
            sims = []
            for vi in range(graph.views):
                if joint[vi]:
                    qa = self.builder.project(query[vi], 'intra')
                    qb = self.builder.project(graph.keys[ai, vi], 'intra')
                    sims.append(float(torch.dot(qa, qb)))
            similarities.append((sum(sims)/len(sims), ai))
        similarities.sort(reverse=True)
        return [idx for _, idx in similarities[:c.sa_k]]

    @torch.no_grad()
    def synthesize(self, target: Product, t: int, graph: HistoricalGraph,
                   products: Dict[str, Product]):
        choices = self.donors(target, t, graph)
        if not choices:
            return None
        c = self.cfg; ai = int(self.rng.choice(choices)); anchor = graph.anchors[ai]
        donor = products[anchor.product_id]
        length = c.context + c.horizon
        xa = target.sales[t-c.context:t+c.horizon]
        xb = donor.sales[anchor.r-c.context:anchor.r+c.horizon]
        if len(xa) != length or len(xb) != length:
            return None
        ta, sa, _ = decompose(xa, c.seasonal_period)
        tb, sb, _ = decompose(xb, c.seasonal_period)
        lt, ls = self.rng.beta(c.mixup_alpha, c.mixup_alpha, size=2)
        shift = phase_shift(sa, sb, c.context, c.seasonal_period)
        # Low-frequency trend noise, scaled to local series magnitude.
        z = self.rng.normal(0, c.trend_noise_std * max(float(np.mean(np.abs(xa))), 1e-4), length)
        z = np.convolve(np.pad(z, (3, 3), mode='edge'), np.ones(7)/7, mode='valid')
        trend = lt*ta + (1-lt)*tb + z
        seasonal = ls*sa + (1-ls)*np.roll(sb, shift)
        if self.generator is None:
            raise RuntimeError('Residual generator not trained: call prepare_residual_generator first')
        device = next(self.generator.parameters()).device
        self.generator.eval()
        residual = self.generator.sample(1, length, device).cpu().numpy()[0]
        # Scale residual generator's standardized output using target residual spread.
        _, _, ra = decompose(xa, c.seasonal_period)
        residual = residual * max(float(ra.std()), 1e-5)
        synthetic = np.maximum(trend + seasonal + residual, 0).astype(np.float32)
        return synthetic[:c.context], synthetic[c.context:]


def prepare_residual_generator(products: Dict[str, Product], train_ids: List[str],
                               config: LiGAConfig, epochs=30, device='cpu'):
    """Fit residual model using training products only; never fit on val/test."""
    chunks = []; length = config.context + config.horizon
    for pid in train_ids:
        x = products[pid].sales
        for start in range(0, len(x)-length+1, max(1, config.horizon)):
            _, _, r = decompose(x[start:start+length], config.seasonal_period)
            chunks.append(r / max(float(r.std()), 1e-5))
    if not chunks:
        raise ValueError('Not enough training windows for residual generator')
    model = ResidualGenerator()
    model.fit(np.stack(chunks), epochs=epochs, device=device)
    return model
