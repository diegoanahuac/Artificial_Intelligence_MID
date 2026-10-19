"""Ridge (L2) and Lasso (L1) regularized linear regression, implemented from
scratch (Session 23). No scikit-learn.

Both fit functions minimize:
    (1/2n) * sum((y - Xw - b)^2)  +  penalty(w)

- Ridge's penalty is (lambda/2) * sum(w^2), solved exactly with the closed-form
  normal equations (a linear system — no iteration needed).
- Lasso's penalty is lambda * sum(|w|), solved with coordinate descent and
  soft-thresholding — the standard, exact algorithm for Lasso (not gradient
  descent, which cannot produce exact zeros).

Everything is a pure function of its inputs, so the same call always returns
the same result. The **canonical models** (fixed lambda values, computed once
at import time) are what every reflection question is graded against; the
app also lets students freely vary lambda for exploration.
"""

import numpy as np

from house_data import ALL_FEATURES, train_test_split

CANONICAL_RIDGE_LAMBDA = 0.03
CANONICAL_LASSO_LAMBDA = 30.0


def _normalize(X_train, X_test):
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std


def _r2(Xn, y, w, b):
    pred = Xn @ w + b
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _mse(Xn, y, w, b):
    pred = Xn @ w + b
    return float(np.mean((y - pred) ** 2))


def ridge_fit(feature_names: list[str], lam: float) -> dict:
    X_train, y_train, X_test, y_test = train_test_split(feature_names)
    Xn_train, Xn_test = _normalize(X_train, X_test)
    n, d = Xn_train.shape

    y_centered = y_train - y_train.mean()
    A = (Xn_train.T @ Xn_train) / n + lam * np.eye(d)
    rhs = (Xn_train.T @ y_centered) / n
    w = np.linalg.solve(A, rhs)
    b = float(y_train.mean())

    return {
        "feature_names": feature_names,
        "lam": lam,
        "weights": w,
        "bias": b,
        "train_r2": _r2(Xn_train, y_train, w, b),
        "test_r2": _r2(Xn_test, y_test, w, b),
        "train_mse": _mse(Xn_train, y_train, w, b),
        "test_mse": _mse(Xn_test, y_test, w, b),
    }


def lasso_fit(feature_names: list[str], lam: float, iterations: int = 1000) -> dict:
    X_train, y_train, X_test, y_test = train_test_split(feature_names)
    Xn_train, Xn_test = _normalize(X_train, X_test)
    n, d = Xn_train.shape

    w = np.zeros(d)
    b = float(y_train.mean())
    y_centered = y_train - b

    for _ in range(iterations):
        for j in range(d):
            residual = y_centered - Xn_train @ w + Xn_train[:, j] * w[j]
            rho = (Xn_train[:, j] @ residual) / n
            z = (Xn_train[:, j] @ Xn_train[:, j]) / n
            if rho < -lam:
                w[j] = (rho + lam) / z
            elif rho > lam:
                w[j] = (rho - lam) / z
            else:
                w[j] = 0.0

    return {
        "feature_names": feature_names,
        "lam": lam,
        "weights": w,
        "bias": b,
        "train_r2": _r2(Xn_train, y_train, w, b),
        "test_r2": _r2(Xn_test, y_test, w, b),
        "train_mse": _mse(Xn_train, y_train, w, b),
        "test_mse": _mse(Xn_test, y_test, w, b),
        "num_zeroed": int(np.sum(np.abs(w) < 1e-6)),
    }


CANONICAL_OLS = ridge_fit(ALL_FEATURES, 1e-8)  # lambda ~ 0 = ordinary least squares
CANONICAL_RIDGE = ridge_fit(ALL_FEATURES, CANONICAL_RIDGE_LAMBDA)
CANONICAL_LASSO = lasso_fit(ALL_FEATURES, CANONICAL_LASSO_LAMBDA)
