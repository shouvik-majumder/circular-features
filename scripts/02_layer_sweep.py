"""Step 2: where in the network does the circle appear, and where does it go?

Runs the same three measures at every layer, for every item set. Two things to look for:

  * the depth profile of the cyclic sets versus the controls. If the controls track the cyclic
    sets, the measure is picking up something generic about the layer, not about cycles.
  * whether the ring is present from the embedding onward (so it comes from token identity)
    or is built up through the network (so the model computes it).

  python scripts/02_layer_sweep.py --model gpt2
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

from cf.geometry import circularity, gaussian_null, shuffle_null  # noqa: E402
from cf.model import MODELS, layer_sweep, load  # noqa: E402
from cf.prompts import DEFAULT_ORDER, SETS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
STYLE = {"cyclic": dict(lw=1.8, ls="-"), "ordered": dict(lw=1.3, ls="--"),
         "control": dict(lw=1.0, ls=":")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--hook", default="resid_post", choices=["resid_pre", "resid_mid", "resid_post"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--sets", nargs="*", default=DEFAULT_ORDER)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.model, device=device, dtype=args.dtype)
    n_layers = model.cfg.n_layers
    print(f"{args.model}: sweeping {n_layers} layers\n")

    out: dict[str, dict] = {}
    for name in args.sets:
        s = SETS[name]
        prof = {"kind": s.kind, "radial_cv": [], "order_score": [], "pc12_variance": []}
        sweep = layer_sweep(model, s, args.hook)          # every layer in one pass per prompt
        for L in range(n_layers):
            m = circularity(sweep[L])
            for k in ("radial_cv", "order_score", "pc12_variance"):
                prof[k].append(m[k])
        X, items = sweep[n_layers - 1], list(s.items)
        prof["shuffle_null_p95"] = shuffle_null(X)["p95"]
        prof["gaussian_null"] = gaussian_null(len(items), X.shape[1])
        out[name] = prof
        print(f"{name:<22} radial CV min {min(prof['radial_cv']):.2f} at layer "
              f"{int(np.argmin(prof['radial_cv']))}; order max {max(prof['order_score']):.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [("radial_cv", "radial CV  (0 = perfect ring)", True),
              ("order_score", "order score  (1 = calendar order)", False),
              ("pc12_variance", "variance in PC1 + PC2", False)]
    for ax, (key, label, invert) in zip(axes, panels):
        for name, prof in out.items():
            ax.plot(range(n_layers), prof[key], label=name, **STYLE[prof["kind"]])
        if key == "order_score":
            nulls = [p["shuffle_null_p95"] for p in out.values()]
            ax.axhspan(0, max(nulls), color="0.85", lw=0, zorder=0)
            ax.text(0.02, max(nulls) * 0.5, " shuffled-label range", fontsize=7.5, color="0.4",
                    transform=ax.get_yaxis_transform())
        ax.set_xlabel(f"layer (blocks.L.hook_{args.hook})")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"{args.model}: circularity of each item set at every layer", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = ROOT / "figures" / f"layer_sweep_{args.model}_{args.hook}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    (ROOT / "data" / f"layer_sweep_{args.model}_{args.hook}.json").write_text(json.dumps(out, indent=2))
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
