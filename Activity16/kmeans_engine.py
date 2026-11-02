"""The k-means clustering algorithm (Lloyd's algorithm), implemented from
scratch in NumPy (Session 29). No scikit-learn.

One run = choose k starting centroids, then repeat two steps:
  1. Assignment: every point joins the cluster of its nearest centroid
     (ties go to the lowest-numbered centroid).
  2. Update: every centroid moves to the mean (average) of its points.
     A centroid that gets no points stays where it is.

The run stops at the first of these that happens, checked after every
iteration:
  - converged: no point changed cluster (so the centroids stopped moving), or
  - tolerance: the largest centroid move was no more than `tol` (only when
    tol > 0), or
  - max iterations: `max_iter` iterations were done.

If the run stopped early (tolerance / max iterations), every point is
reassigned once more to its nearest final centroid, so the final groups
always match the final centroids.

Starting centroids: "random" picks k distinct data points at random (the
lecture's method); "k-means++" is the original method of Arthur &
Vassilvitskii (2007): the first centroid is a random point, and each next one
is a point chosen with probability proportional to its squared distance to
the nearest centroid already chosen; "manual" uses points chosen by index.
Everything random comes from `np.random.default_rng(seed)`, so the same
settings always give exactly the same result.
"""

import itertools

import numpy as np

INITS = ("random", "k-means++", "manual")


def standardize(X):
    """z-scores: each column rescaled to mean 0 and standard deviation 1."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, mean, std


def _sq_dists(X, C):
    """Squared distance from every point (rows of X) to every centroid (rows of C)."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def assign(X, C):
    return _sq_dists(X, C).argmin(axis=1)


def wcss(X, labels, C):
    """Within-cluster sum of squares (inertia): total squared distance of every point to its centroid."""
    return float(((X - C[labels]) ** 2).sum())


def initial_indices(X, k, init, seed, manual_idx=None):
    if init == "manual":
        assert manual_idx is not None and len(manual_idx) == k
        return list(manual_idx)
    rng = np.random.default_rng(seed)
    if init == "random":
        return [int(i) for i in rng.choice(len(X), size=k, replace=False)]
    idx = [int(rng.integers(len(X)))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(rng.choice(len(X), p=d2 / d2.sum()))
        idx.append(nxt)
        d2 = np.minimum(d2, ((X - X[nxt]) ** 2).sum(axis=1))
    return idx


def run(X, k, init="k-means++", seed=0, max_iter=100, tol=0.0, manual_idx=None) -> dict:
    assert init in INITS
    start_idx = initial_indices(X, k, init, seed, manual_idx)
    C = X[start_idx].astype(float)
    steps = []
    labels_prev = None
    stop_reason = "max iterations"

    for it in range(1, max_iter + 1):
        labels = assign(X, C)
        changed = len(X) if labels_prev is None else int((labels != labels_prev).sum())
        counts = np.bincount(labels, minlength=k)
        C_new = C.copy()
        for j in range(k):
            if counts[j] > 0:
                C_new[j] = X[labels == j].mean(axis=0)
        max_move = float(np.sqrt(((C_new - C) ** 2).sum(axis=1)).max())
        C = C_new
        steps.append({
            "iteration": it,
            "labels": labels,
            "centroids": C.copy(),
            "changed": changed,
            "max_move": max_move,
            "wcss": wcss(X, labels, C),
            "empty": [j for j in range(k) if counts[j] == 0],
        })
        if labels_prev is not None and changed == 0:
            stop_reason = "converged"
            break
        if tol > 0 and max_move <= tol:
            stop_reason = "tolerance"
            break
        labels_prev = labels

    final_labels = steps[-1]["labels"] if stop_reason == "converged" else assign(X, C)
    return {
        "k": k,
        "init": init,
        "seed": seed,
        "max_iter": max_iter,
        "tol": tol,
        "start_idx": start_idx,
        "start_centroids": X[start_idx].astype(float),
        "steps": steps,
        "stop_reason": stop_reason,
        "iterations": len(steps),
        "labels": final_labels,
        "centroids": C,
        "wcss": wcss(X, final_labels, C),
        "sizes": np.bincount(final_labels, minlength=k),
    }


def silhouette(X, labels):
    """Mean silhouette score s = (b - a) / max(a, b) (Rousseeuw, 1987).

    a = mean distance to the other points of the same cluster,
    b = mean distance to the points of the nearest other cluster.
    Points alone in their cluster score 0. Needs at least 2 non-empty clusters.
    """
    groups = np.unique(labels)
    if len(groups) < 2:
        return None
    D = np.sqrt(_sq_dists(X, X))
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        n_own = own.sum()
        if n_own == 1:
            continue
        a = D[i, own].sum() / (n_own - 1)
        b = min(D[i, labels == g].mean() for g in groups if g != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def best_of(X, k, init="k-means++", n_runs=10, max_iter=300):
    """Lowest-WCSS run among seeds 0 .. n_runs-1."""
    return min((run(X, k, init, seed, max_iter) for seed in range(n_runs)), key=lambda r: r["wcss"])


def elbow_table(X, k_values=range(1, 11), n_runs=10):
    rows = []
    for k in k_values:
        best = best_of(X, k, n_runs=n_runs)
        rows.append({"k": k, "wcss": best["wcss"], "silhouette": silhouette(X, best["labels"]) if k > 1 else None})
    return rows


def restarts(X, k, init, n_runs, max_iter=300):
    return [run(X, k, init, seed, max_iter) for seed in range(n_runs)]


def match_accuracy(labels, true_labels):
    """Share of points whose cluster matches their true group, under the best
    pairing of clusters 0..k-1 with the k true groups (every pairing is tried)."""
    groups = np.unique(true_labels)
    return max(float(np.mean(np.array(perm)[labels] == true_labels)) for perm in itertools.permutations(groups))
