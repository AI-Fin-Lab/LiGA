import argparse
from pathlib import Path
import json
import torch
from liga.config import LiGAConfig
from liga.data import load_products, split_by_first_observation
from liga.pipeline import LiGAPipeline


def main():
    p = argparse.ArgumentParser(description='Train LiGA reference pipeline on CSV data')
    p.add_argument('--csv', required=True)
    p.add_argument('--output', default='runs/liga.pt')
    p.add_argument('--context', type=int, default=30)
    p.add_argument('--horizon', type=int, default=7)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--ca-epochs', type=int, default=1)
    p.add_argument('--residual-epochs', type=int, default=2)
    p.add_argument('--max-origins', type=int, default=16)
    p.add_argument('--no-ca', action='store_true')
    p.add_argument('--no-sa', action='store_true')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    products = load_products(args.csv)
    train, val, test = split_by_first_observation(products)
    if not train: raise ValueError('No training products')
    cfg = LiGAConfig(context=args.context, horizon=args.horizon)
    pipeline = LiGAPipeline(cfg, products[train[0]].covariates.shape[1], args.device)
    pipeline.fit(products, train, val, epochs=args.epochs, ca_epochs=args.ca_epochs,
                 residual_epochs=args.residual_epochs, max_origins=args.max_origins,
                 enable_ca=not args.no_ca, enable_sa=not args.no_sa)
    result = pipeline.evaluate(products, test, train+val, max_origins=20,
                               enable_ca=not args.no_ca)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(path))
    print('Test:', json.dumps(result, indent=2))
    path.with_suffix('.metrics.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__': main()
