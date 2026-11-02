"""The ShopSmart customer dataset (Session 29: k-means).

300 fictional customers of ShopSmart (the online store from Activities 8 and
14), described by two numbers: how many times per month they buy, and how
much they spend per visit (MXN). The data is generated once, with a fixed
seed, at import time.

The generator draws customers around four built-in segments. Those segment
labels are kept in TRUE_SEGMENT only for the instructor's answer key; the
app never shows them, because k-means (unsupervised learning) never sees
labels either.
"""

import numpy as np

FEATURE_LABELS = ("Visits per month", "Average spend per visit (MXN)")

# name, customers, visits mean, visits sd, spend mean, spend sd
_SEGMENTS = [
    ("Occasional bargain hunters", 95, 2.0, 0.8, 320, 90),
    ("Daily convenience shoppers", 80, 15.0, 2.0, 230, 60),
    ("Weekend big-basket families", 70, 4.0, 1.0, 1650, 230),
    ("Loyal premium customers", 55, 11.0, 1.8, 1250, 200),
]
SEGMENT_NAMES = [s[0] for s in _SEGMENTS]


def _generate(seed: int = 29):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for g, (_, n, v_mean, v_sd, s_mean, s_sd) in enumerate(_SEGMENTS):
        visits = np.clip(rng.normal(v_mean, v_sd, n), 0.5, 20.0)
        spend = np.clip(rng.normal(s_mean, s_sd, n), 80, 2500)
        X.append(np.column_stack([visits, spend]))
        y += [g] * n
    X, y = np.vstack(X), np.array(y)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X[:, 0] = np.round(X[:, 0], 1)
    X[:, 1] = np.round(X[:, 1])
    return X, y


X, TRUE_SEGMENT = _generate()
CUSTOMER_IDS = np.arange(1, len(X) + 1)
