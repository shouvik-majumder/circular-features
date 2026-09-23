"""Step 7: is the weekday x month representation a torus?

  sampling    five templates per (weekday, month) pair, read at the weekday token, the month
              token and the final token. Only the final token has attended to both factors
              (attention is causal), so it is the valid joint measurement.

  real nulls  a label permutation cannot test topology: shuffling names leaves the point cloud,
              and therefore the barcode, untouched. Two nulls that do perturb the cloud:
                gaussian  84 points from a Gaussian with the same covariance. Keeps the
                          second-order structure (PCA looks similar), destroys the product.
                additive  the same additive model but with random vectors per level. Keeps
                          additivity, destroys the circularity of each factor.
              A torus needs to beat both.

  power       a null only means something if the test could have detected the real thing. A
              synthetic ideal torus with the SAME weekday/month variance split is pushed through
              the identical pipeline. At the last token the split is roughly 3% weekday to 96%
              month, and for a torus that lopsided the small weekday loop falls below the joint-
              cloud nulls even when it is perfect. So the joint barcode alone cannot rule a torus
              in or out for the minor factor. Two tests that do have power are added:
                per-factor  is each factor's MAIN EFFECT (7 or 12 points) a loop, against a
                            covariance-matched null of the same size?
                dominant    does the joint cloud show the dominant factor's loop at all?
                            A synthetic torus with this split does, strongly.

  depth       is the product structure present from the first layers or built later? Sweep
              every other layer.

  python scripts/07_torus_validation.py --model gemma-2-2b-it --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cf.model import MODELS, item_token_span, load  # noqa: E402
from cf.periodic import (betti_numbers, circle_plane_of, subspace_angles,  # noqa: E402
                         two_way_decomposition)
from cf.prompts import MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"

JOINT_TEMPLATES = [
    "The date is {a}, {b} the 3rd",
    "We met on {a} in {b}",
    "It happened on a {a} in {b}",
    "{a}, {b} 12, was a good day",
    "Scheduled for {a}, {b}",
]


@torch.no_grad()
def joint_activations(model, layers: list[int], hook: str, device: str):
    """Activations for every (weekday, month) pair at three read positions, averaged over
    templates, for ALL requested layers in a single forward pass each.

    Returns {layer: {position: [84, d]}} plus the factor index arrays. Caching every layer at
    once rather than re-running the prompts per layer is a 13x saving on the depth sweep.
    """
    names = [f"blocks.{L}.hook_{hook}" for L in layers]
    acc = {L: {k: {} for k in ("weekday_token", "month_token", "last_token")} for L in layers}
    for i, d in enumerate(WEEKDAYS.items):
        for j, mo in enumerate(MONTHS.items):
            for t in JOINT_TEMPLATES:
                p = t.format(a=d, b=mo)
                toks = model.to_tokens(p, prepend_bos=True)
                _, cache = model.run_with_cache(toks, names_filter=names)
                _, de = item_token_span(model, p, d)
                _, me = item_token_span(model, p, mo)
                for L, nm in zip(layers, names):
                    A = cache[nm][0].float().cpu().numpy()
                    for key, idx in (("weekday_token", de - 1), ("month_token", me - 1),
                                     ("last_token", A.shape[0] - 1)):
                        acc[L][key].setdefault((i, j), []).append(A[idx])
    keys = [(i, j) for i in range(len(WEEKDAYS.items)) for j in range(len(MONTHS.items))]
    out = {L: {k: np.stack([np.mean(acc[L][k][key], axis=0) for key in keys]) for k in acc[L]}
           for L in layers}
    a_idx = np.array([k[0] for k in keys])
    b_idx = np.array([k[1] for k in keys])
    return out, a_idx, b_idx


def h1_gap(X: np.ndarray) -> tuple[float, float]:
    """Lifetimes of the two longest 1-cycles. A torus needs two of comparable length."""
    t = betti_numbers(X, maxdim=1)
    lives = t.get("H1", {}).get("top_lifetimes", [])
    return (float(lives[0]) if lives else 0.0, float(lives[1]) if len(lives) > 1 else 0.0)


def nulls(X: np.ndarray, a_idx, b_idx, n_draws: int, seed: int) -> dict:
    """Two cloud-perturbing nulls for the two-loop signature."""
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0, keepdims=True)
    # keep only the leading subspace so the Gaussian draw is well conditioned
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.99) + 1)
    basis, scale = Vt[:k], S[:k] / np.sqrt(len(X))

    g1, g2, a1, a2 = [], [], [], []
    for _ in range(n_draws):
        G = (rng.standard_normal((len(X), k)) * scale) @ basis
        l1, l2 = h1_gap(G)
        g1.append(l1); g2.append(l2)
        # additive model, random vectors per level: additive but not circular
        fa = rng.standard_normal((a_idx.max() + 1, k)) @ basis
        fb = rng.standard_normal((b_idx.max() + 1, k)) @ basis
        A = fa[a_idx] + fb[b_idx]
        A = A / np.linalg.norm(A - A.mean(0), axis=1).mean() * np.linalg.norm(Xc, axis=1).mean()
        l1, l2 = h1_gap(A)
        a1.append(l1); a2.append(l2)
    q = lambda v: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))}
    return {"gaussian": {"first": q(g1), "second": q(g2)},
            "additive_random": {"first": q(a1), "second": q(a2)}}


def synthetic_torus(a_idx, b_idx, factor_a: float, factor_b: float, interaction: float,
                    d: int = 200, seed: int = 0) -> np.ndarray:
    """An ideal torus with a given variance split: two perfect, evenly spaced, orthogonal circles
    carrying `factor_a` and `factor_b` of the variance, plus isotropic noise carrying the
    interaction share. The positive control: what the pipeline reports for a real torus."""
    rng = np.random.default_rng(seed)
    ta = 2 * np.pi * a_idx / (a_idx.max() + 1)
    tb = 2 * np.pi * b_idx / (b_idx.max() + 1)
    X = np.zeros((len(a_idx), d))
    X[:, 0], X[:, 1] = np.sqrt(factor_a) * np.cos(ta), np.sqrt(factor_a) * np.sin(ta)
    X[:, 2], X[:, 3] = np.sqrt(factor_b) * np.cos(tb), np.sqrt(factor_b) * np.sin(tb)
    return X + rng.standard_normal(X.shape) * np.sqrt(interaction / d)


def loop_vs_null(W: np.ndarray, n_draws: int, seed: int) -> dict:
    """Longest 1-cycle of a small point set (one factor's main effect, 7 or 12 points) against
    a covariance-matched Gaussian null of the same size. This asks 'is this factor a loop?'
    without the other factor's much larger variance swamping the answer."""
    rng = np.random.default_rng(seed)
    Wc = W - W.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    scale = S / np.sqrt(len(W))
    real = h1_gap(W)[0]
    draws = [h1_gap((rng.standard_normal((len(W), len(S))) * scale) @ Vt)[0] for _ in range(n_draws)]
    return {"h1": real, "null_mean": float(np.mean(draws)), "null_p95": float(np.percentile(draws, 95)),
            "p_value": float((1 + sum(x >= real for x in draws)) / (1 + n_draws))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layers for the sweep")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)
    out = {"model": args.model, "layer": layer, "templates": JOINT_TEMPLATES}

    # ------------------------------------------- 1 + 2: read positions and nulls
    print(f"=== layer {layer}: {len(JOINT_TEMPLATES)} templates per pair, three read positions ===")
    layers = args.layers or list(range(0, model.cfg.n_layers, 2))
    all_acts, a_idx, b_idx = joint_activations(model, sorted(set(layers + [layer])), args.hook, device)
    acts = all_acts[layer]
    out["positions"] = {}
    for pos, X in acts.items():
        dec = two_way_decomposition(X, a_idx, b_idx)
        Wa = np.stack([X[a_idx == i].mean(0) for i in range(a_idx.max() + 1)])
        Wb = np.stack([X[b_idx == j].mean(0) for j in range(b_idx.max() + 1)])
        ang = subspace_angles(circle_plane_of(Wa), circle_plane_of(Wb))
        l1, l2 = h1_gap(X)
        nl = nulls(X, a_idx, b_idx, args.n_null, seed=0)
        out["positions"][pos] = {
            "additive_r2": dec["additive_model_r2"], "interaction": dec["interaction"],
            "factor_a": dec["factor_a"], "factor_b": dec["factor_b"],
            "principal_angles_deg": ang.tolist(), "h1": [l1, l2], "nulls": nl,
        }
        print(f"\n  read at {pos}")
        print(f"    additive R^2 {dec['additive_model_r2']:.3f}  interaction {dec['interaction']:.3f}"
              f"  angles {np.round(ang, 1)} deg")
        print(f"    two longest 1-cycles: {l1:.3f}, {l2:.3f}")
        print(f"      null, matched Gaussian      : {nl['gaussian']['first']['p95']:.3f}, "
              f"{nl['gaussian']['second']['p95']:.3f}  (95th pct)")
        print(f"      null, additive random levels: {nl['additive_random']['first']['p95']:.3f}, "
              f"{nl['additive_random']['second']['p95']:.3f}")
        # positive control: the same pipeline on an ideal torus with this variance split
        syn = synthetic_torus(a_idx, b_idx, dec["factor_a"], dec["factor_b"],
                              max(dec["interaction"], 1e-4))
        s1, s2 = h1_gap(syn)
        snl = nulls(syn, a_idx, b_idx, max(args.n_null // 2, 20), seed=1)
        syn_old_rule = (s2 > snl["gaussian"]["second"]["p95"]
                        and s2 > snl["additive_random"]["second"]["p95"])
        # per-factor loops, each factor's main effect on its own
        fa_loop = loop_vs_null(Wa, args.n_null, seed=2)
        fb_loop = loop_vs_null(Wb, args.n_null, seed=3)
        dominant_seen = l1 > max(nl["gaussian"]["first"]["p95"], nl["additive_random"]["first"]["p95"])
        out["positions"][pos].update({
            "positive_control": {"h1": [s1, s2], "nulls": snl,
                                 "passes_two_loop_rule": bool(syn_old_rule)},
            "weekday_main_effect_loop": fa_loop, "month_main_effect_loop": fb_loop,
            "dominant_loop_above_null": bool(dominant_seen),
        })
        print(f"    positive control (ideal torus, same split): 1-cycles {s1:.3f}, {s2:.3f}; "
              f"passes the two-loop rule: {syn_old_rule}")
        print(f"    weekday main effect alone: longest loop {fa_loop['h1']:.3f} "
              f"(null p95 {fa_loop['null_p95']:.3f}, p = {fa_loop['p_value']:.3f})")
        print(f"    month main effect alone:   longest loop {fb_loop['h1']:.3f} "
              f"(null p95 {fb_loop['null_p95']:.3f}, p = {fb_loop['p_value']:.3f})")
        # A torus needs both factors to be loops and the dominant loop to survive in the joint
        # cloud. The two-loop rule on the joint cloud is reported but not used: the positive
        # control shows it has no power for a split this lopsided.
        both_loops = fa_loop["p_value"] < 0.05 and fb_loop["p_value"] < 0.05
        verdict = ("consistent with a torus" if both_loops and dominant_seen else
                   "not a torus" if not dominant_seen and not both_loops else "partial")
        print(f"      verdict: {verdict}")
        out["positions"][pos]["verdict"] = verdict

    # ------------------------------------------------------- 3: depth profile
    # Read at the LAST token. Attention is causal, so at the weekday token the month has not been
    # seen yet: additivity there is 1.000 by construction, not by finding.
    layers = args.layers or list(range(0, model.cfg.n_layers, 2))
    print(f"\n=== depth sweep over layers {layers} (read at the last token) ===")
    prof = {"layer": [], "additive_r2": [], "interaction": [], "angle_min": [],
            "h1_first": [], "h1_second": []}
    for L in layers:
        X, ai, bi = all_acts[L]["last_token"], a_idx, b_idx
        dec = two_way_decomposition(X, ai, bi)
        Wa = np.stack([X[ai == i].mean(0) for i in range(ai.max() + 1)])
        Wb = np.stack([X[bi == j].mean(0) for j in range(bi.max() + 1)])
        ang = subspace_angles(circle_plane_of(Wa), circle_plane_of(Wb))
        l1, l2 = h1_gap(X)
        for k, v in (("layer", L), ("additive_r2", dec["additive_model_r2"]),
                     ("interaction", dec["interaction"]), ("angle_min", float(ang.min())),
                     ("h1_first", l1), ("h1_second", l2)):
            prof[k].append(v)
        print(f"  layer {L:2d}: additive R^2 {dec['additive_model_r2']:.3f}  "
              f"min angle {ang.min():5.1f} deg  1-cycles {l1:.3f}, {l2:.3f}")
    out["depth_profile"] = prof

    # save the numbers before any plotting, so a figure bug cannot lose a 25-minute run
    (ROOT / "data" / f"torus_validation_{args.model}.json").write_text(
        json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    pos_names = list(out["positions"])
    x = np.arange(len(pos_names))
    w = 0.35
    first = [out["positions"][p]["h1"][0] for p in pos_names]
    syn_first = [out["positions"][p]["positive_control"]["h1"][0] for p in pos_names]
    ax.plot(x - w / 2, syn_first, "o", mfc="none", mec=ACCENT, ms=8, mew=1.5,
            label="ideal torus, same split")
    second = [out["positions"][p]["h1"][1] for p in pos_names]
    gnull = [out["positions"][p]["nulls"]["gaussian"]["second"]["p95"] for p in pos_names]
    anull = [out["positions"][p]["nulls"]["additive_random"]["second"]["p95"] for p in pos_names]
    ax.bar(x - w / 2, first, w, color=ACCENT, edgecolor=DARK, label="longest 1-cycle")
    ax.bar(x + w / 2, second, w, color=GREY, edgecolor=DARK, label="second 1-cycle")
    ax.plot(x + w / 2, gnull, "_", color="k", ms=14, mew=2, label="null: matched Gaussian")
    ax.plot(x + w / 2, anull, "x", color="k", ms=7, mew=1.5, label="null: additive random")
    ax.set_xticks(x); ax.set_xticklabels([p.replace("_", "\n") for p in pos_names], fontsize=8)
    ax.set_ylabel("persistence")
    ax.set_title("Joint-cloud loops vs nulls and a positive control", fontsize=9.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(prof["layer"], prof["additive_r2"], "-o", ms=4, color=ACCENT, label="additive $R^2$")
    ax.plot(prof["layer"], prof["interaction"], "--s", ms=4, color=GREY, label="interaction")
    ax.set_xlabel("layer"); ax.set_ylabel("fraction of variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("Is the product structure built with depth?", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(prof["layer"], prof["h1_first"], "-o", ms=4, color=ACCENT, label="longest 1-cycle")
    ax.plot(prof["layer"], prof["h1_second"], "--s", ms=4, color=GREY, label="second 1-cycle")
    ax2 = ax.twinx()
    ax2.plot(prof["layer"], prof["angle_min"], ":^", ms=4, color=DARK, label="min plane angle")
    ax2.set_ylabel("smallest principal angle (deg)")
    ax2.set_ylim(0, 95)
    ax.set_xlabel("layer"); ax.set_ylabel("persistence")
    ax.set_title("Loops and independence across depth", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{args.model}, hook_{args.hook}: does the weekday x month torus hold up?",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = ROOT / "figures" / f"torus_validation_{args.model}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
