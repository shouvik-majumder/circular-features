"""Measuring whether a set of item vectors lies on a circle, and controls for saying so.

Three numbers are reported for every item set:

  pc12_variance   fraction of variance captured by the first two principal components.
                  A flat 2D structure of any kind needs this to be high.

  radial_cv       coefficient of variation of the distance from the centre, in the PC1-PC2
                  plane. A perfect circle gives 0. Low means "ring", not just "flat".

  order_score     how well the angular order around the ring matches the calendar order,
                  measured as the fraction of items that sit in exactly their calendar slot
                  going round (best over rotations and both directions). This is the part that separates a *meaningful* circle from a
                  coincidental ring: seven arbitrary points can look round, but they will not be
                  in the right order.

The controls matter more than the metrics. Random high-dimensional points projected onto their
own top two principal components tend to look ring-like, because the projection maximises spread.
So every number is reported alongside the same number computed for control item sets and for
shuffled labels.
"""
from __future__ import annotations

import numpy as np


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, then project onto the first two principal components.

    Returns (xy [n, 2], explained variance ratio of all components, the 2D basis [d, 2]).
    """
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / np.sum(S**2)
    return Xc @ Vt[:2].T, var, Vt[:2].T


def radial_cv(xy: np.ndarray) -> float:
    """Spread of the radius relative to its mean. 0 for a perfect circle."""
    c = xy.mean(0)
    r = np.linalg.norm(xy - c, axis=1)
    return float(np.std(r) / np.mean(r)) if np.mean(r) > 0 else np.nan


def angular_order_score(xy: np.ndarray) -> float:
    """Fraction of items that occupy exactly their calendar slot in angular order.

    Items are assumed to be given in their natural order (Monday...Sunday). We sort them by
    angle around the centroid, then compare each item's rank with its calendar index, allowing
    any rotation of the starting point and either direction, and keep the best match. 1.0 means
    the ring is traversed in calendar order. One adjacent swap of 7 items scores 5/7.
    """
    n = len(xy)
    c = xy.mean(0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    rank = np.argsort(np.argsort(ang))            # position of each item around the ring
    best = 0.0
    for direction in (1, -1):
        for shift in range(n):
            target = (direction * np.arange(n) + shift) % n
            best = max(best, float(np.mean(rank == target)))
    return best


def circularity(X: np.ndarray) -> dict:
    """All three measures for one item set."""
    xy, var, _ = pca_2d(X)
    return {
        "pc12_variance": float(var[:2].sum()),
        "radial_cv": radial_cv(xy),
        "order_score": angular_order_score(xy),
        "xy": xy,
        "explained_variance": var,
    }


def shuffle_null(X: np.ndarray, n_draws: int = 2000, seed: int = 0) -> dict:
    """Null distribution for the order score: same points, shuffled labels.

    The ring geometry is untouched; only the correspondence between items and points is broken.
    This isolates 'the ring is in the right order' from 'the points happen to form a ring'.
    """
    rng = np.random.default_rng(seed)
    xy, _, _ = pca_2d(X)
    scores = [angular_order_score(xy[rng.permutation(len(xy))]) for _ in range(n_draws)]
    return {"mean": float(np.mean(scores)), "p95": float(np.percentile(scores, 95)),
            "max": float(np.max(scores))}


def gaussian_null(n_items: int, d_model: int, n_draws: int = 200, seed: int = 0) -> dict:
    """Null for the ring shape itself: isotropic Gaussian points of the same size and dimension.

    A weak null: real activations are far from isotropic. Kept for continuity; the covariance-
    matched null below and the control item sets are the comparisons that carry weight.
    """
    rng = np.random.default_rng(seed)
    cvs, orders, pc12 = [], [], []
    for _ in range(n_draws):
        m = circularity(rng.standard_normal((n_items, d_model)))
        cvs.append(m["radial_cv"]); orders.append(m["order_score"]); pc12.append(m["pc12_variance"])
    return {"radial_cv_mean": float(np.mean(cvs)), "radial_cv_p05": float(np.percentile(cvs, 5)),
            "order_mean": float(np.mean(orders)), "order_p95": float(np.percentile(orders, 95)),
            "pc12_mean": float(np.mean(pc12))}


def matched_gaussian_null(X: np.ndarray, n_draws: int = 200, seed: int = 0) -> dict:
    """Null for the ring shape with the items' own second-order structure.

    Draws the same number of points from a Gaussian with the item set's covariance, so the
    spread along each principal direction matches. Any circularity the real set shows beyond
    this is not explained by 'a few dominant directions of variance'.
    """
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scale = S / np.sqrt(len(X))
    cvs, pc12 = [], []
    for _ in range(n_draws):
        G = (rng.standard_normal((len(X), len(S))) * scale) @ Vt
        m = circularity(G)
        cvs.append(m["radial_cv"]); pc12.append(m["pc12_variance"])
    return {"radial_cv_mean": float(np.mean(cvs)), "radial_cv_p05": float(np.percentile(cvs, 5)),
            "pc12_mean": float(np.mean(pc12))}


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through 2D points (algebraic fit). Returns (centre, radius)."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy**2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = sol[:2]
    radius = float(np.sqrt(sol[2] + centre @ centre))
    return centre, radius
