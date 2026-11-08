"""Reference training and inference pipeline aligned with manuscript Algorithm 1.

The implementation intentionally favors clarity over large-scale performance.
Graph recomputation per cutoff is expensive. Production systems should cache
cutoff-indexed eligible anchor banks and update their sparse neighborhoods offline.
"""
from dataclasses import asdict
from typing import Dict, List
import json
import numpy as np
import torch
from torch.nn import functional as F
from .config import LiGAConfig
from .data import Product, Origin, real_example, training_origins, evaluation_origins
from .encoder import SharedMultiViewEncoder
from .graph import LifecycleGraphBuilder
from .ca import CovariateAugmenter
from .sa import SampleAugmenter, prepare_residual_generator
from .model import MLPForecaster


class LiGAPipeline:
    def __init__(self, cfg: LiGAConfig, num_covariates: int, device='cpu'):
        self.cfg = cfg; self.device = device
        torch.manual_seed(cfg.random_seed); np.random.seed(cfg.random_seed)
        self.encoder = SharedMultiViewEncoder(cfg.embedding_dim).to(device)
        self.builder = LifecycleGraphBuilder(cfg, self.encoder).to(device)
        self.ca = CovariateAugmenter(cfg, self.builder).to(device)
        self.forecaster = MLPForecaster(cfg.context, cfg.horizon, num_covariates).to(device)
        self.sa = SampleAugmenter(cfg, self.builder)
        self.num_covariates = num_covariates

    def _graph(self, products: Dict[str, Product], source_ids: List[str], origin: Origin):
        target = products[origin.product_id]
        cutoff = target.dates[origin.t-1]
        return self.builder.build(products, source_ids, cutoff, origin.product_id, self.device)

    def fit_ca(self, products, train_ids, origins, epochs=2, lr=1e-3):
        """Pretrain CA auxiliary Eq. 11 before materializing CA features.

        Only train products supply repository anchors. Any target product is
        excluded from its own retrieval bank (leave-one-product-out).
        """
        optimizer = torch.optim.AdamW(self.ca.parameters(), lr=lr)
        for epoch in range(epochs):
            self.ca.train(); total = 0.; count = 0
            for origin in origins:
                graph = self._graph(products, train_ids, origin)
                target = products[origin.product_id]
                loss = self.ca.auxiliary_loss(target, origin.t, graph)
                if loss is None:
                    continue
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item(); count += 1
            print(f'CA epoch {epoch+1}/{epochs}: auxiliary MSE={total/max(count,1):.6f} (usable={count})')

    def _featurize(self, products, source_ids, origin, enable_ca=True):
        target = products[origin.product_id]
        graph = self._graph(products, source_ids, origin) if enable_ca else None
        ctx, cov, label = real_example(products, origin, self.cfg.context, self.cfg.horizon)
        if enable_ca:
            with torch.no_grad():
                ca, mask = self.ca.augment_window(target, origin.t, graph)
        else:
            ca = torch.zeros(self.cfg.context, self.cfg.horizon, device=self.device)
            mask = torch.zeros(self.cfg.context, 1, device=self.device)
        return (torch.tensor(ctx.copy(), device=self.device),
                torch.tensor(cov.copy(), device=self.device), ca.detach(), mask.detach(),
                torch.tensor(label.copy(), device=self.device), graph)

    def fit(self, products, train_ids, val_ids=None, ca_epochs=2, epochs=10, batch_size=32,
            lr=1e-3, residual_epochs=10, enable_ca=True, enable_sa=True,
            max_origins=128):
        """Train on a bounded origin subset for the self-contained reference demo.

        max_origins=None processes all origins, but graph construction is O(N^2).
        For full paper-scale experiments, use an ANN index and cached sparse graph.
        """
        origins = training_origins(products, train_ids, self.cfg.context,
                                   self.cfg.horizon, max_history=self.cfg.sufficient_history)
        if max_origins is not None and len(origins) > max_origins:
            rng = np.random.default_rng(self.cfg.random_seed)
            selected = sorted(rng.choice(len(origins), size=max_origins, replace=False))
            origins = [origins[int(i)] for i in selected]
        if not origins:
            raise ValueError('Training set has no complete input-target windows')
        if enable_ca:
            self.fit_ca(products, train_ids, origins, epochs=ca_epochs, lr=lr)
        if enable_sa:
            self.sa.generator = prepare_residual_generator(products, train_ids,
                                self.cfg, epochs=residual_epochs, device=self.device)
        self.ca.eval(); self.encoder.eval(); self.builder.eval()
        # Precompute all augmentation features once before downstream optimization.
        examples = []
        for origin in origins:
            ctx, cov, ca, mask, labels, graph = self._featurize(products, train_ids, origin, enable_ca)
            examples.append((ctx, cov, ca, mask, labels))
            if enable_sa and np.random.rand() < self.cfg.sa_ratio:
                if graph is None:
                    graph = self._graph(products, train_ids, origin)
                synthetic = self.sa.synthesize(products[origin.product_id], origin.t, graph, products)
                if synthetic is not None:
                    sx, sy = synthetic
                    # Paper: synthetic sales do not receive external covariates.
                    examples.append((torch.tensor(sx, device=self.device),
                        cov.new_zeros(cov.shape), ca.new_zeros(ca.shape), mask.new_zeros(mask.shape),
                        torch.tensor(sy, device=self.device)))
        optimizer = torch.optim.AdamW(self.forecaster.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs,1))
        best = float('inf'); best_weights = None; patience = 10; stale = 0
        for ep in range(epochs):
            self.forecaster.train()
            order = torch.randperm(len(examples))
            train_losses = []
            for indices in order.split(batch_size):
                batch = [torch.stack([examples[int(i)][k] for i in indices]) for k in range(5)]
                pred = self.forecaster(*batch[:4]); loss = F.mse_loss(pred, batch[4])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_losses.append(float(loss.detach()))
            scheduler.step()
            val = self.evaluate(products, val_ids, train_ids, max_origins=20, enable_ca=enable_ca) if val_ids else None
            metric = val['mse'] if val else float(np.mean(train_losses))
            print(f'Forecast epoch {ep+1}/{epochs}: train_MSE={np.mean(train_losses):.6f} '
                  + (f'val_MSE={metric:.6f}' if val else ''))
            if metric < best:
                best = metric; stale = 0
                best_weights = {k: v.detach().clone() for k,v in self.forecaster.state_dict().items()}
            else:
                stale += 1
                if stale >= patience: break
        if best_weights is not None:
            self.forecaster.load_state_dict(best_weights)
        return self

    @torch.no_grad()
    def predict(self, products, source_ids, origin: Origin, enable_ca=True):
        self.forecaster.eval(); self.ca.eval()
        ctx, cov, ca, mask, _, _ = self._featurize(products, source_ids, origin, enable_ca)
        return self.forecaster(ctx[None], cov[None], ca[None], mask[None])[0].cpu().numpy()

    @torch.no_grad()
    def evaluate(self, products, ids, source_ids, max_origins=None, enable_ca=True):
        origins = evaluation_origins(products, ids, self.cfg.context, self.cfg.horizon,
                                     self.cfg.sufficient_history)
        if max_origins is not None:
            origins = origins[:max_origins]
        preds = []; truths = []
        for origin in origins:
            preds.append(self.predict(products, source_ids, origin, enable_ca))
            truths.append(products[origin.product_id].sales[origin.t:origin.t+self.cfg.horizon])
        if not preds:
            return {'mse': float('nan'), 'mae': float('nan'), 'n': 0}
        a = np.stack(preds); b = np.stack(truths)
        return {'mse': float(np.mean((a-b)**2)), 'mae': float(np.mean(abs(a-b))), 'n': len(preds)}

    def save(self, path):
        torch.save({'config': asdict(self.cfg), 'num_covariates': self.num_covariates,
                    'encoder': self.encoder.state_dict(), 'builder': self.builder.state_dict(),
                    'ca': self.ca.state_dict(), 'forecaster': self.forecaster.state_dict(),
                    'residual_generator': self.sa.generator.state_dict() if self.sa.generator else None}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        obj = torch.load(path, map_location=device, weights_only=True)
        settings = obj['config']; settings['views'] = tuple(tuple(x) for x in settings['views'])
        pipeline = cls(LiGAConfig(**settings), obj['num_covariates'], device)
        pipeline.encoder.load_state_dict(obj['encoder'])
        pipeline.builder.load_state_dict(obj['builder'])
        pipeline.ca.load_state_dict(obj['ca'])
        pipeline.forecaster.load_state_dict(obj['forecaster'])
        if obj['residual_generator'] is not None:
            from .sa import ResidualGenerator
            pipeline.sa.generator = ResidualGenerator().to(device)
            pipeline.sa.generator.load_state_dict(obj['residual_generator'])
        return pipeline
