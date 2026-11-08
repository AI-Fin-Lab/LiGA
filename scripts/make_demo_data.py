"""Generate public synthetic example data. No MYBank data is bundled."""
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/examples/demo.csv')
    parser.add_argument('--products', type=int, default=8)
    parser.add_argument('--days', type=int, default=50)
    args = parser.parse_args()
    rng = np.random.default_rng(17); rows = []
    for i in range(args.products):
        start = np.datetime64('2023-01-01') + np.timedelta64(i*3, 'D')
        for t in range(args.days):
            rows.append(dict(product_id=f'P{i:03d}', date=str(start+np.timedelta64(t,'D')),
                sales=max(0., 20 + i*2 + .15*t + 3*np.sin(2*np.pi*t/7) + rng.normal(0,2)),
                return_rate=0.025 + 0.001*i, uv=150+2*t+rng.normal(0,3),
                market_index=3000+20*np.sin(t/11)))
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f'Wrote {args.output}: {len(rows)} rows, purely synthetic')

if __name__ == '__main__': main()
