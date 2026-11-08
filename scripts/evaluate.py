import argparse
import json
from liga.data import load_products, split_by_first_observation
from liga.pipeline import LiGAPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-origins', type=int, default=100)
    p.add_argument('--no-ca', action='store_true')
    args = p.parse_args()
    products = load_products(args.csv)
    train, val, test = split_by_first_observation(products)
    pipeline = LiGAPipeline.load(args.checkpoint, args.device)
    print(json.dumps(pipeline.evaluate(products, test, train+val,
          args.max_origins, enable_ca=not args.no_ca), indent=2))

if __name__ == '__main__': main()
