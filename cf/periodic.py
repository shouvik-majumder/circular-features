"""Characterising the geometry of one or more cyclic variables.

Three questions, three tools.

1. **Which periods are present?** A variable like "day of month" could carry a 31-cycle, a 7-cycle
   (weeks), a 10-cycle (base-10 digits), and a linear ramp, all at once. `fourier_spectrum`
   regresses the item vectors on sine/cosine pairs at each candidate period plus a linear term and
   reports how much variance each explains. A representation with both a linear and a circular
   component is a **helix**.

2. **Do two cyclic variables form a torus?** If the joint representation of (weekday, month) is
   additive, X(d, m) = c + f(d) + g(m), and f and g are each circular and lie in different
   subspaces, then the 7x12 points trace a torus. `two_way_decomposition` measures the additivity
   and `subspace_angles` measures whether the two circles are independent or share directions.

3. **What is the topology?** `betti_numbers` runs persistent homology. A circle gives one
   long-lived 1-cycle; a torus gives two, plus a 2-cycle. This does not assume any parametric
   shape, which is the point of using it.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ 1. periods
def design_matrix(values: np.ndarray, periods: list[float], linear: bool = True) -> tuple[np.ndarray, list[str]]:
    """Columns: constant, optional linear ramp, then cos/sin at each period."""
    cols, names = [np.ones(len(values))], ["const"]
    if linear:
        z = (values - values.mean()) / (values.std() + 1e-12)
        cols.append(z); names.append("linear")
    for T in periods:
        w = 2 * np.pi * values / T
        cols += [np.cos(w), np.sin(w)]
        names += [f"cos(T={T:g})", f"sin(T={T:g})"]
    return np.stack(cols, axis=1), names


def _r2(X: np.ndarray, D: np.ndarray) -> float:
    """Variance of X explained by least-squares regression on D."""
    Xc = X - X.mean(0, keepdims=True)
    W, *_ = np.linalg.lstsq(D, Xc, rcond=None)
    resid = Xc - D @ W
    return float(1.0 - (resid**2).sum() / max((Xc**2).sum(), 1e-12))


def fourier_spectrum(X: np.ndarray, values: np.ndarray, periods: list[float],
                     linear: bool = True) -> dict:
    """Unique variance explained by each period, and by the linear term.

    For each component we report the drop in R^2 when that component alone is removed from the
    full model. That is the variance it explains *that nothing else can*, which is what you want
    when periods are correlated (a 10-cycle and a 5-cycle share structure).
    """
    D_full, names = design_matrix(values, periods, linear)
    r2_full = _r2(X, D_full)
    unique = {}
    for T in periods:
        keep = [i for i, n in enumerate(names) if f"T={T:g}" not in n]
        unique[f"T={T:g}"] = r2_full - _r2(X, D_full[:, keep])
    if linear:
        keep = [i for i, n in enumerate(names) if n != "linear"]
        unique["linear"] = r2_full - _r2(X, D_full[:, keep])
    # each component on its own, for comparison
    alone = {}
    for T in periods:
        cols = [i for i, n in enumerate(names) if n == "const" or f"T={T:g}" in n]
        alone[f"T={T:g}"] = _r2(X, D_full[:, cols])
    if linear:
        cols = [i for i, n in enumerate(names) if n in ("const", "linear")]
        alone["linear"] = _r2(X, D_full[:, cols])
    return {"r2_full": r2_full, "unique": unique, "alone": alone}


def spectrum_null(X: np.ndarray, values: np.ndarray, periods: list[float], linear: bool = True,
                  n_draws: int = 200, seed: int = 0) -> dict:
    """How much unique variance each component explains when the labels are shuffled.

    Necessary because the design matrix has 2 parameters per period. With 31 items and 7 periods
    that is 17 parameters, so some apparent structure is simply fitting capacity. Any component
    that does not clear this null is not evidence of a period.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_draws):
        perm = rng.permutation(len(values))
        s = fourier_spectrum(X[perm], values, periods, linear)
        for k, v in s["unique"].items():
            draws.setdefault(k, []).append(v)
    return {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))}
            for k, v in draws.items()}


def harmonic_decomposition(X: np.ndarray, n_items: int | None = None, seed: int = 0,
                           n_draws: int = 200) -> dict:
    """Is the loop a *circle*, or a deformed one? Decompose it into harmonics.

    For n items in cyclic order, regress the item vectors on cos/sin at harmonic m, for
    m = 1 .. floor(n/2). A perfect circle in some plane is pure harmonic 1: going once round the
    item order sweeps exactly one revolution at constant radius. Any deformation - a squashed
    ellipse, uneven spacing, a kink, an out-of-plane wobble - shows up as power at m >= 2.

    Returns the fraction of explainable variance in each harmonic, so `fraction[1]` near 1 means
    "a genuine circle" and a long tail means "a loop that is only roughly circular".

    The null shuffles the item order, which *does* change this measure (unlike topology, where
    relabelling leaves the point cloud untouched), because harmonics are defined relative to the
    ordering.
    """
    n = n_items or len(X)
    Xc = X - X.mean(0, keepdims=True)
    k = np.arange(n)

    def power(order: np.ndarray) -> np.ndarray:
        Y = Xc[order]
        out = []
        for m in range(1, n // 2 + 1):
            D = np.stack([np.cos(2 * np.pi * m * k / n), np.sin(2 * np.pi * m * k / n)], axis=1)
            W, *_ = np.linalg.lstsq(D, Y, rcond=None)
            out.append(float(((D @ W) ** 2).sum()))
        return np.array(out)

    p = power(np.arange(n))
    total = float((Xc ** 2).sum())
    rng = np.random.default_rng(seed)
    null = np.stack([power(rng.permutation(n)) for _ in range(n_draws)])
    return {
        "harmonic_power": (p / total).tolist(),
        "fraction_in_fundamental": float(p[0] / max(p.sum(), 1e-12)),
        "null_p95": (np.percentile(null, 95, axis=0) / total).tolist(),
        "explained_by_all_harmonics": float(p.sum() / total),
    }


def pc_spread_ratio(X: np.ndarray) -> float:
    """Ratio of the spread along PC2 to the spread along PC1 of the whole item cloud.

    1.0 = equal spread in the best plane, 0 = a line. This is NOT an ellipse fit: for a deformed
    loop the top two PCs also absorb higher-harmonic and uneven-spacing variance.
    """
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(s[1] / max(s[0], 1e-12))


def angular_gaps(X: np.ndarray) -> dict:
    """Spacing of consecutive items around the loop. A regular polygon has equal gaps."""
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc @ Vt[:2].T
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi * np.sign(np.diff(ang).sum())))
    gaps = np.abs(gaps)
    ideal = 2 * np.pi / len(X)
    return {"gaps_deg": np.degrees(gaps).tolist(), "ideal_deg": float(np.degrees(ideal)),
            "cv_of_gaps": float(np.std(gaps) / max(np.mean(gaps), 1e-12))}


# -------------------------------------------------------------- 2. two factors
def two_way_decomposition(X: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> dict:
    """Split the variance of a 2-factor item set into main effects and interaction.

    X: [n_items, d]; a_idx, b_idx: the level of factor A and factor B for each item.
    Returns the fraction of variance in each part. Additivity (small interaction) is what makes a
    clean product geometry, i.e. a torus rather than a tangled surface.
    """
    Xc = X - X.mean(0, keepdims=True)
    A = np.zeros((len(X), a_idx.max() + 1)); A[np.arange(len(X)), a_idx] = 1
    B = np.zeros((len(X), b_idx.max() + 1)); B[np.arange(len(X)), b_idx] = 1
    fa = A @ np.linalg.lstsq(A, Xc, rcond=None)[0]
    fb = B @ np.linalg.lstsq(B, Xc, rcond=None)[0]
    D = np.concatenate([A, B], axis=1)
    additive = D @ np.linalg.lstsq(D, Xc, rcond=None)[0]
    total = (Xc**2).sum()
    return {
        "factor_a": float((fa**2).sum() / total),
        "factor_b": float((fb**2).sum() / total),
        "additive_model_r2": float(1 - ((Xc - additive)**2).sum() / total),
        "interaction": float(((Xc - additive)**2).sum() / total),
        "residual_after_additive": Xc - additive,
    }


def subspace_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between two subspaces, each given as columns.

    90 degrees on every angle means the two circles occupy completely independent directions,
    which is what a clean product (torus) requires. Small angles mean they share a subspace.
    """
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def circle_plane_of(X: np.ndarray) -> np.ndarray:
    """The 2D plane (as a [d, 2] matrix) best containing a set of item vectors."""
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:2].T


# ------------------------------------------------------------------ 3. topology
def betti_numbers(X: np.ndarray, maxdim: int = 2, n_report: int = 3) -> dict:
    """Persistent homology of a point cloud, summarised by the longest-lived features.

    Returns, for each dimension, the lifetimes of the `n_report` most persistent classes and a
    'gap' score: the ratio of the longest lifetime to the next one. A circle shows one dominant
    1-dimensional class; a torus shows two of similar length plus a 2-dimensional class.
    """
    try:
        from ripser import ripser
    except ImportError:
        return {"error": "ripser not installed"}
    Xn = (X - X.mean(0)) / (np.linalg.norm(X - X.mean(0), axis=1).mean() + 1e-12)
    dgms = ripser(Xn, maxdim=maxdim)["dgms"]
    out = {}
    for dim, d in enumerate(dgms):
        finite = d[np.isfinite(d[:, 1])]
        lives = np.sort(finite[:, 1] - finite[:, 0])[::-1] if len(finite) else np.array([])
        out[f"H{dim}"] = {
            "top_lifetimes": [float(x) for x in lives[:n_report]],
            "n_classes": int(len(lives)),
        }
    return out
