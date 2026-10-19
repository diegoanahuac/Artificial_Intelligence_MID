"""Regularized logistic regression via gradient descent, implemented from
scratch (Session 24). No scikit-learn.

- Plain and L2 (Ridge-style) use standard gradient descent on the
  cross-entropy cost, with L2 adding `lambda * w` to the gradient.
- L1 (Lasso-style) uses **proximal gradient descent (ISTA)**: an ordinary
  gradient step on the cross-entropy loss, followed by a soft-thresholding
  ("shrinkage") step. This is the standard way to get *exact* zero
  coefficients out of L1-regularized logistic regression — plain gradient
  descent on a subgradient cannot reliably do that.

Everything is a pure function of its inputs, so the same call always returns
the same result. The **canonical models** (fixed lambda values, computed
once at import time) are what every reflection question is graded against.
"""

import numpy as np

from churn_data import ALL_FEATURES, train_test_split

LEARNING_RATE = 0.5
ITERATIONS = 3000

CANONICAL_L2_LAMBDA = 0.05
CANONICAL_L1_LAMBDA = 0.05


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _normalize(X_train, X_test):
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std


def _accuracy(Xn, y, w, b):
    pred = _sigmoid(Xn @ w + b) > 0.5
    return float(np.mean(pred == y))


def _log_loss(Xn, y, w, b):
    p = np.clip(_sigmoid(Xn @ w + b), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(feature_names: list[str], lam: float, penalty: str) -> dict:
    """penalty: 'none', 'l2', or 'l1'."""
    assert penalty in ("none", "l2", "l1")
    X_train, y_train, X_test, y_test = train_test_split(feature_names)
    Xn_train, Xn_test = _normalize(X_train, X_test)
    n, d = Xn_train.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(ITERATIONS):
        z = Xn_train @ w + b
        pred = _sigmoid(z)
        error = pred - y_train
        grad_w = (Xn_train.T @ error) / n
        grad_b = np.sum(error) / n

        if penalty == "l2":
            grad_w = grad_w + lam * w
            w -= LEARNING_RATE * grad_w
        elif penalty == "l1":
            w -= LEARNING_RATE * grad_w
            threshold = LEARNING_RATE * lam
            w = np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)
        else:
            w -= LEARNING_RATE * grad_w
        b -= LEARNING_RATE * grad_b

    result = {
        "feature_names": feature_names,
        "lam": lam,
        "penalty": penalty,
        "weights": w,
        "bias": b,
        "train_accuracy": _accuracy(Xn_train, y_train, w, b),
        "test_accuracy": _accuracy(Xn_test, y_test, w, b),
        "train_loss": _log_loss(Xn_train, y_train, w, b),
        "test_loss": _log_loss(Xn_test, y_test, w, b),
    }
    if penalty == "l1":
        result["num_zeroed"] = int(np.sum(np.abs(w) < 1e-6))
    return result


CANONICAL_PLAIN = fit(ALL_FEATURES, 0.0, "none")
CANONICAL_L2 = fit(ALL_FEATURES, CANONICAL_L2_LAMBDA, "l2")
CANONICAL_L1 = fit(ALL_FEATURES, CANONICAL_L1_LAMBDA, "l1")
