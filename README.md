# LiGA — Lifecycle-Aligned Multi-View Graph Augmentation

**Research reference implementation** for *Forecasting Before History Matures: Lifecycle-Aligned Multi-View Graph Augmentation for Financial Product Sales*.

LiGA augments a forecasting pipeline in two complementary ways:

- **Covariate augmentation (CA)**: lifecycle-compatible historical anchors, intra-/inter-view top-k graphs, one-/two-hop observed-continuation aggregation, multi-view fusion and a prediction auxiliary loss.
- **Structure-preserving sample augmentation (SA)**: lifecycle-compatible donor search, trend/seasonal/residual decomposition, trend interpolation, phase-aligned seasonal mixing, residual generation, and complete-window reconstruction.

CA is applied during training and inference; SA is applied **only during training**. The target product is excluded from its own retrieval bank. All historical continuations must have been observed by the **calendar cutoff of the target query**.

> [!IMPORTANT]
> This is an **independently written, executable reference implementation**, not the authors' original experimental code. It follows the algorithmic definitions in the manuscript but is **not an exact numerical reproduction**. In particular, the default causal CNN is a lightweight stand-in for the manuscript's pretrained TSFM encoder (reported `d=1280`), the sample residual GRU-GAN is **TimeGAN-inspired rather than the complete TimeGAN training procedure**, and the included MLP is a demonstration backbone, not the paper's TFT/TimesFM/etc. Official backbone integrations, proprietary MYBank data and exact training/checkpoint preprocessing are not included. Do not present demo results as paper results.

## Project structure

```text
src/liga/
  config.py       Hyperparameters / temporal views
  data.py         Product CSV loader, time splits, origin windows
  encoder.py      Shared causal multi-view representations
  graph.py        Lifecycle-safe historical anchors and sparse relations
  ca.py           CA retrieval, one-/two-hop aggregation and fusion
  sa.py           SA donor selection, decomposition and residual generator
  model.py        Lightweight example covariate-aware backbone
  pipeline.py     Preprocessing, training, evaluation and checkpointing
scripts/
  make_demo_data.py
  train.py
  evaluate.py
configs/default.json
tests/test_core.py
.github/workflows/tests.yml
```

## Quick start

Python 3.9+ (Python 3.11 recommended), PyTorch 2.x:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/make_demo_data.py --output data/examples/demo.csv --products 8 --days 50
python scripts/train.py --csv data/examples/demo.csv --epochs 2 --ca-epochs 1 --residual-epochs 1 --max-origins 8 --output runs/demo.pt
python scripts/evaluate.py --csv data/examples/demo.csv --checkpoint runs/demo.pt --max-origins 10
```

For a simple baseline without augmentation:

```bash
python scripts/train.py --csv data/examples/demo.csv --no-ca --no-sa --output runs/baseline.pt
```

> These commands demonstrate wiring. With only eight synthetic products, the holdout is tiny and metrics have no scientific meaning. In this reference the graph is rebuilt per forecasting cutoff and is intended only for small datasets. Full-scale evaluation requires offline graph/index caching and chunked/ANN nearest-neighbor retrieval.

## Data contract

CSV input schema:

| column | type | meaning |
|---|---|---|
| `product_id` | string | unique product ID |
| `date` | `YYYY-MM-DD` | calendar date of observed record |
| `sales` | numeric | daily sales target |
| `return_rate`, `uv`, `market_index`, ... | numeric | any number of observed covariates |

One row per `(product_id, date)`, sorted automatically on load. Values must be finite, dates strictly increasing per product and sales should be nonnegative. **Only covariates known at the query cutoff are allowed.** Fill missing data and fit scalers using training-only information in your own preprocessing code; the example deliberately does not invent business-specific missing-value policies. Keep production data out of Git (see `.gitignore`).

The example splits products chronologically by first available date (80%/10%/10%, product-exclusive). Training retrieval excludes the target product. Validation references training products, and test references train+validation products, subject to each query cutoff. `t` in code is the number of observed values, corresponding to lifecycle position `t` in the paper. A donor anchor `r` has complete horizon `r:r+F` and must satisfy `r+F <= available_length_at_cutoff`.

## Manuscript ↔ code map

| Manuscript | Implementation |
|---|---|
| Eq. (1) forecasting task | `data.real_example`, `model.MLPForecaster` |
| Eq. (2) query-time historical repository | `Product.observed_length`, `pipeline._graph` |
| Eq. (3) temporal view encoder | `encoder.SharedMultiViewEncoder.encode_views` |
| Eq. (4) eligible historical anchors | `graph.LifecycleGraphBuilder.build` |
| Eqs. (6)–(7) intra/inter-view top-k graph | `graph.LifecycleGraphBuilder.build` |
| Eq. (8) entry retrieval + anchor dedup | `ca.CovariateAugmenter._entry_nodes` |
| Eq. (9) first-/second-hop aggregation | `ca.CovariateAugmenter._neighbor_evidence` |
| Eqs. (10)–(11) fusion and CA loss | `ca.CovariateAugmenter.at_position`, `auxiliary_loss` |
| Eq. (12) donor retrieval | `sa.SampleAugmenter.donors` |
| Eqs. (13)–(14) augmentation and reconstruction | `sa.SampleAugmenter.synthesize` |
| Eqs. (15)–(17), Algorithm 1 | `pipeline.LiGAPipeline` |

## Reported paper settings vs. demo settings

| Setting | Manuscript | Default here |
|---|---|---|
| View configurations | `(7,1),(14,1),(30,1),(14,2)` | same |
| Input length | 30 | 30 |
| Forecast horizons | 7, 14, 28 | 7 (CLI configurable) |
| TSFM representation | 1280 | 64-d causal CNN |
| Graph projection | 640 | 32 |
| Graph degree | 5 | 5 |
| CA `k`, SA `k` | 2, 3 (Sec. 5.1.4) | 2, 3 |
| Lifecycle tolerance | 7 days | 7 days |
| Two-hop mixing β | 0.5 | 0.5 |
| Beta mixup α | 0.4 | 0.4 |
| Seasonal period | 7 | 7 |
| Residual generator | TimeGAN | lightweight GRU-GAN approximation |
| Forecast backbones | 12 configurations incl. TimesFM, TFT | MLP example |

**Manuscript discrepancy to verify:** Section 5.1.4 reports `k_CA=2`, `k_SA=3` while sensitivity discussion in Section 5.5 reports minima near `k_CA=3`, `k_SA=2`. We use the explicit implementation-settings values rather than infer a correction.

## Path to a faithful large-scale reproduction

1. Plug in the **exact pretrained TSFM encoder** (checkpoint, pooling, normalization and frozen/fine-tuned policy) instead of `SharedMultiViewEncoder`; change dimensions to 1280/640.
2. Replace `ResidualGenerator` with a trained **full TimeGAN** generator on *training-products-only* residuals. The provided adversarial GRU illustrates the interface only.
3. Integrate the named official forecasting backbones with their paper-consistent covariate protocols, external-regressor availability, normalization and optimization.
4. Replace per-query full graph enumeration with **calendar-cutoff-aware cached sparse indices** and scalable similarity search; preserve the exact same leakage restrictions.
5. Run matched seeds, history stages and F in `{7,14,28}`; log per-product metrics and complete configurations; compare only on the same test origins.
6. Confirm any unresolved research implementation choices (graph temperature, noise scale, base model hyperparameters, TSFM checkpoint, SA mixing ratio) against author training configs before claiming numerical reproduction.

## Privacy and reproducibility

The repository includes **no MYBank customer/product records**, no proprietary artifacts, no secrets and no paper result tables advertised as reproduced metrics. The generated demo dataset is synthetic. Raw data and model checkpoints are ignored by Git. Public M5 can be prepared separately in compliance with its original terms.

## Citation

Please cite the manuscript title shown above. Add its confirmed publication metadata/DOI once available; a placeholder citation is provided in `CITATION.cff`.

## License

MIT for this reference code; it does **not** grant rights to proprietary datasets or external pretrained model weights. See `LICENSE`.
