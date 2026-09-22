"""Step 6: is it a circle, or a deformed loop?

Three independent ways of asking, all on the same item vectors:

  harmonics    decompose the loop into harmonics of the item order. A perfect circle is pure
               harmonic 1. Power at harmonic 2 and above is deformation.
  PC spread    PC2 spread / PC1 spread of the loop. 1.0 is round, smaller is squashed. Not an
               ellipse fit: higher harmonics also load on the top two PCs.
  gaps         angular spacing between consecutive items. A regular polygon has equal gaps.

  python scripts/06_deformation.py --model gemma-2-2b-it --dtype bfloat16
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

from cf.geometry import circularity  # noqa: E402
from cf.model import MODELS, item_activations, load  # noqa: E402
from cf.periodic import angular_gaps, harmonic_decomposition, pc_spread_ratio  # noqa: E402
from cf.prompts import MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)
    out = {"model": args.model, "layer": layer}

    sets = [("weekdays", WEEKDAYS), ("months", MONTHS)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for name, s in sets:
        X, items = item_activations(model, s, layer, args.hook, device=device)
        h = harmonic_decomposition(X)
        gaps = angular_gaps(X)
        m = circularity(X)
        rec = {"fraction_in_fundamental": h["fraction_in_fundamental"],
               "harmonic_power": h["harmonic_power"], "null_p95": h["null_p95"],
               "pc_spread_ratio": pc_spread_ratio(X), "pc12_variance": m["pc12_variance"],
               "gaps": gaps}
        out[name] = rec
        print(f"\n{name} ({len(items)} items)")
        print(f"  power in harmonic 1 (a true circle would be ~1.00): "
              f"{h['fraction_in_fundamental']:.2f}")
        print(f"  harmonic power / total variance: "
              f"{np.round(h['harmonic_power'], 3)}  (null 95th {np.round(h['null_p95'], 3)})")
        print(f"  PC2/PC1 spread (1 = round):     {rec['pc_spread_ratio']:.2f}")
        print(f"  variance in the best plane:     {m['pc12_variance']:.2f}  "
              f"(the rest is out-of-plane wobble)")
        print(f"  angular gaps (ideal {gaps['ideal_deg']:.0f} deg): "
              f"{np.round(gaps['gaps_deg'], 0)}, CV {gaps['cv_of_gaps']:.2f}")

    # ------------------------------------------------------------------ figure
    ax = axes[0]
    w = 0.38
    for i, (name, _) in enumerate(sets):
        p = out[name]["harmonic_power"]
        xs = np.arange(1, len(p) + 1) + (i - 0.5) * w
        ax.bar(xs, p, w, color=[ACCENT, GREY][i], edgecolor=DARK, label=name)
        ax.plot(xs, out[name]["null_p95"], "_", color=DARK, ms=9, mew=1.5)
    ax.set_xlabel("harmonic of the item order")
    ax.set_ylabel("fraction of variance")
    ax.set_title("A perfect circle is all harmonic 1\n(dashes = shuffled-order 95th pct)", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, (name, _) in enumerate(sets):
        g = out[name]["gaps"]
        ax.plot(np.arange(1, len(g["gaps_deg"]) + 1), g["gaps_deg"], "-o", ms=4,
                color=[ACCENT, GREY][i], label=f"{name} (CV {g['cv_of_gaps']:.2f})")
        ax.axhline(g["ideal_deg"], color=[ACCENT, GREY][i], ls=":", lw=1)
    ax.set_xlabel("step around the loop")
    ax.set_ylabel("angular gap (degrees)")
    ax.set_title("Even spacing? Dotted = regular polygon", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["power in\nharmonic 1", "PC2/PC1\nspread", "variance in\nbest plane"]
    for i, (name, _) in enumerate(sets):
        vals = [out[name]["fraction_in_fundamental"], out[name]["pc_spread_ratio"],
                out[name]["pc12_variance"]]
        ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, color=[ACCENT, GREY][i],
               edgecolor=DARK, label=name)
    ax.axhline(1.0, color=DARK, ls="--", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("value (1 = ideal circle)")
    ax.set_title("How circular is the circle?", fontsize=9.5)
    ax.legend(fontsize=8)

    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: circle or deformed loop?",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = ROOT / "figures" / f"deformation_{args.model}_L{layer}.png"
    fig.savefig(path, bbox_inches="tight")
    (ROOT / "data" / f"deformation_{args.model}_L{layer}.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
