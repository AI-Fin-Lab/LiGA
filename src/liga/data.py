"""Product-level data, leakage-safe calendar cutoffs, and training windows.

CSV schema: product_id,date,sales,<numeric covariates...>
Dates must be parseable as ISO YYYY-MM-DD. All observation indices are zero-based
in code; paper lifecycle position r is one-based, i.e. r = index + 1.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


@dataclass
class Product:
    product_id: str
    dates: np.ndarray  # dtype datetime64[ns]
    sales: np.ndarray  # float32
    covariates: np.ndarray  # shape [length, num_covariates]

    def __post_init__(self):
        self.dates = np.asarray(self.dates, dtype='datetime64[ns]')
        self.sales = np.asarray(self.sales, dtype=np.float32)
        self.covariates = np.asarray(self.covariates, dtype=np.float32)
        if self.covariates.ndim != 2 or len(self.dates) != len(self.sales) or len(self.sales) != len(self.covariates):
            raise ValueError('Expected aligned dates, sales and 2D covariates')
        if len(self.dates) and np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError(f'Dates must be strictly increasing for {self.product_id}')

    @property
    def length(self) -> int:
        return len(self.sales)

    def observed_length(self, cutoff: np.datetime64) -> int:
        """Count observations with calendar date <= target cutoff."""
        return int(np.searchsorted(self.dates, cutoff, side='right'))


def load_products(path: str, covariate_columns: Optional[Sequence[str]] = None) -> Dict[str, Product]:
    frame = pd.read_csv(path)
    required = {'product_id', 'date', 'sales'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if covariate_columns is None:
        covariate_columns = [c for c in frame.columns if c not in required]
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
    frame['sales'] = pd.to_numeric(frame['sales'], errors='raise')
    for c in covariate_columns:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    if frame[['sales', *covariate_columns]].isna().any().any():
        raise ValueError('Missing sales/covariates: impute using training-only statistics first')
    products = {}
    for pid, group in frame.groupby('product_id', sort=False):
        group = group.sort_values('date')
        products[str(pid)] = Product(str(pid), group['date'].to_numpy(),
                                    group['sales'].to_numpy(dtype=np.float32),
                                    group[list(covariate_columns)].to_numpy(dtype=np.float32))
    return products


def split_by_first_observation(products: Dict[str, Product], fractions=(0.8, 0.1, 0.1)):
    """Chronological product-exclusive 80/10/10 split; no product crosses partitions."""
    if not np.isclose(sum(fractions), 1):
        raise ValueError('Fractions must sum to 1')
    ids = sorted(products, key=lambda pid: (products[pid].dates[0], pid))
    n = len(ids)
    a = int(n * fractions[0]); b = int(n * (fractions[0] + fractions[1]))
    return ids[:a], ids[a:b], ids[b:]


@dataclass(frozen=True)
class Origin:
    product_id: str
    t: int  # number of available observations; context ends at t-1


def training_origins(products: Dict[str, Product], ids: Sequence[str], context: int,
                     horizon: int, max_history: Optional[int] = None) -> List[Origin]:
    origins = []
    for pid in ids:
        p = products[pid]
        upper = min(p.length - horizon, max_history or p.length)
        origins.extend(Origin(pid, t) for t in range(context, upper + 1))
    return origins


def evaluation_origins(products: Dict[str, Product], ids: Sequence[str], context: int,
                       horizon: int, sufficient: int = 70) -> List[Origin]:
    return [Origin(pid, t) for pid in ids
            for t in range(context, min(products[pid].length - horizon + 1, sufficient))]


def real_example(products: Dict[str, Product], origin: Origin, context: int, horizon: int):
    p = products[origin.product_id]; t = origin.t
    if t < context or t + horizon > p.length:
        raise ValueError('Invalid supervised example')
    return (p.sales[t-context:t], p.covariates[t-context:t], p.sales[t:t+horizon])
