"""Two small datasets where k-means struggles (Session 29: disadvantages).

Both are generated once with a fixed seed, and both come with their true
groups so the app can compare them with what k-means finds.

- "Two moons": two interleaved half-circles (non-spherical groups).
- "One big group, two small ones": a large, spread-out group of 300 points
  next to two small, tight groups of 40 points each (different sizes and
  densities).
"""

import numpy as np


def _moons(seed=1, n=150, noise=0.08):
    rng = np.random.default_rng(seed)
    t1, t2 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
    upper = np.column_stack([np.cos(t1), np.sin(t1)])
    lower = np.column_stack([1 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (2 * n, 2))
    return X, np.array([0] * n + [1] * n)


def _big_and_small(seed=3, n_big=300, n_small=40):
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal([0.0, 0.0], 1.5, (n_big, 2)),
        rng.normal([4.0, 2.0], 0.35, (n_small, 2)),
        rng.normal([4.0, -2.0], 0.35, (n_small, 2)),
    ])
    return X, np.array([0] * n_big + [1] * n_small + [2] * n_small)


DATASETS = {
    "Two moons (non-spherical groups)": _moons(),
    "One big group, two small ones (different sizes and densities)": _big_and_small(),
}
