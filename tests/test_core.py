import numpy as np
import torch
from liga.config import LiGAConfig
from liga.data import Product, Origin, real_example, split_by_first_observation
from liga.encoder import SharedMultiViewEncoder, causal_aggregate
from liga.graph import LifecycleGraphBuilder
from liga.ca import CovariateAugmenter
from liga.sa import decompose, SampleAugmenter, ResidualGenerator
from liga.pipeline import LiGAPipeline


def products_fixture(n=3, days=43):
    return {f'p{i}': Product(f'p{i}', np.datetime64('2024-01-01') + np.arange(days).astype('timedelta64[D]')
                            + np.timedelta64(i*2, 'D'),
                 (10 + i + np.sin(np.arange(days))).astype(np.float32),
                 np.ones((days, 1), np.float32)) for i in range(n)}


def test_causal_aggregate():
    x = np.array([1.,2.,3.,4.])
    assert np.allclose(causal_aggregate(x, 2), [1.,1.5,2.5,3.5])


def test_split_exclusive():
    data = products_fixture(10)
    a,b,c = split_by_first_observation(data)
    assert (len(a), len(b), len(c)) == (8, 1, 1)
    assert not (set(a)&set(b) or set(a)&set(c) or set(b)&set(c))


def test_history_cutoff_and_no_target_leakage():
    data = products_fixture()
    cfg = LiGAConfig(context=30, horizon=7, embedding_dim=16, projection_dim=8,
                     views=((7,1),(30,1)), lifecycle_tolerance=7)
    enc = SharedMultiViewEncoder(cfg.embedding_dim)
    builder = LifecycleGraphBuilder(cfg, enc)
    cutoff = data['p2'].dates[34]
    graph = builder.build(data, list(data), cutoff, excluded_id='p2')
    assert graph.anchors
    assert all(a.product_id != 'p2' for a in graph.anchors)
    assert all(a.r+cfg.horizon <= data[a.product_id].observed_length(cutoff) for a in graph.anchors)
    assert all(graph.continuations.shape[1] == cfg.horizon for _ in graph.anchors)
    augmenter = CovariateAugmenter(cfg, builder)
    ca, mask = augmenter.augment_window(data['p2'], 35, graph)
    assert ca.shape == (30,7) and mask.shape == (30,1)
    assert torch.isfinite(ca).all()


def test_decomposition_reconstructs():
    x = np.linspace(1,10,40).astype(np.float32)
    t,s,r = decompose(x, 7)
    assert np.allclose(t+s+r, x)


def test_sa_nonnegative_and_complete():
    data = products_fixture(n=3, days=50)
    cfg = LiGAConfig(context=30, horizon=7, embedding_dim=16, projection_dim=8,
                     views=((7,1),(30,1)), lifecycle_tolerance=7)
    builder = LifecycleGraphBuilder(cfg, SharedMultiViewEncoder(cfg.embedding_dim))
    graph = builder.build(data, list(data), data['p2'].dates[41], excluded_id='p2')
    augmenter = SampleAugmenter(cfg, builder, ResidualGenerator())
    pair = augmenter.synthesize(data['p2'], 42, graph, data)
    assert pair is not None
    context, label = pair
    assert len(context) == 30 and len(label) == 7
    assert context.min() >= 0 and label.min() >= 0


def test_checkpoint_roundtrip(tmp_path):
    model = LiGAPipeline(LiGAConfig(), num_covariates=2)
    path = tmp_path / 'x.pt'; model.save(path)
    restored = LiGAPipeline.load(path)
    assert restored.cfg.horizon == 7 and restored.num_covariates == 2
