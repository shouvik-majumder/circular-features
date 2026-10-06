"""Step 1: do cyclic concepts lie on circles, and do controls not?

For each item set: collect one activation vector per item, project to its own first two
principal components, and measure how ring-like and how correctly ordered the result is.
Everything is reported against two nulls.

  python scripts/01_find_circles.py                        # gpt2, layer 7
  python scripts/01_find_circles.py --layer 5 --hook resid_pre
  python scripts/01_find_circles.py --model gemma-2-2b-it --dtype bfloat16
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

from cf.geometry import (circularity, fit_circle, gaussian_null, matched_gaussian_null,
                         shuffle_null)  # noqa: E402
from cf.model import MODELS, item_activations, load  # noqa: E402
from cf.prompts import DEFAULT_ORDER, SETS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None, help="default: the model's middle layer")
    ap.add_argument("--hook", default="resid_post", choices=["resid_pre", "resid_mid", "resid_post"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--sets", nargs="*", default=DEFAULT_ORDER)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)
    print(f"{args.model}: {model.cfg.n_layers} layers, d_model {model.cfg.d_model} | "
          f"looking at blocks.{layer}.hook_{args.hook}\n")

    results, rows = {}, []
    for name in args.sets:
        s = SETS[name]
        X, items = item_activations(model, s, layer, args.hook, device=device)
        m = circularity(X)
        sh = shuffle_null(X)
        gn = gaussian_null(len(items), X.shape[1])
        mg = matched_gaussian_null(X)
        results[name] = {"kind": s.kind, "items": items,
                         "pc12_variance": m["pc12_variance"], "radial_cv": m["radial_cv"],
                         "order_score": m["order_score"], "xy": m["xy"].tolist(),
                         "shuffle_null": sh, "gaussian_null": gn,
                         "matched_gaussian_null": mg}
        rows.append((name, s.kind, m, sh, gn))
        print(f"{name:<22} ({s.kind})")
        print(f"   variance in PC1+PC2 : {m['pc12_variance']:.2f}   (random points: {gn['pc12_mean']:.2f})")
        print(f"   radial CV (0=circle): {m['radial_cv']:.2f}   (isotropic points: {gn['radial_cv_mean']:.2f}; "
              f"covariance-matched: mean {mg['radial_cv_mean']:.2f}, 5th pct {mg['radial_cv_p05']:.2f})")
        print(f"   order score         : {m['order_score']:.2f}   "
              f"(shuffled labels: mean {sh['mean']:.2f}, 95th pct {sh['p95']:.2f})")

    # ------------------------------------------------------------------ figure
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.9))
    axes = np.atleast_1d(axes)
    for ax, (name, kind, m, sh, gn) in zip(axes, rows):
        xy = m["xy"]
        items = results[name]["items"]
        if kind != "control":
            c, r = fit_circle(xy)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), "-", color=GREY, lw=0.8)
        ax.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1], xy[0, 1]), "-",
                color=ACCENT if kind == "cyclic" else GREY, lw=1.0, alpha=0.7)
        ax.scatter(xy[:, 0], xy[:, 1], s=28, color=DARK, zorder=3)
        for (x, y), lab in zip(xy, items):
            ax.annotate(lab[:3], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nPC1+PC2 {m['pc12_variance']:.2f} | radial CV {m['radial_cv']:.2f}\n"
                     f"order {m['order_score']:.2f} (shuffled {sh['p95']:.2f})", fontsize=8.5)
    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: item vectors in their own top two "
                 f"principal components. Lines connect items in calendar order.", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "figures" / f"circles_{args.model}_L{layer}_{args.hook}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nfigure -> {out}")

    (ROOT / "data").mkdir(exist_ok=True)
    js = ROOT / "data" / f"circles_{args.model}_L{layer}_{args.hook}.json"
    js.write_text(json.dumps({"model": args.model, "layer": layer, "hook": args.hook,
                              "results": results}, indent=2))
    print(f"numbers -> {js}")


if __name__ == "__main__":
    main()
