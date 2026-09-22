"""Step 5: the picture of the geometry itself.

Three views that make the numbers in step 4 visible:

  main effects  the fitted weekday and month main effects, each in its own best plane.
  torus 3D      the same points drawn on a torus, radius from the month angle, tube from the
                weekday angle. IMPORTANT: this is a rendering of two fitted angles onto a torus
                we drew ourselves. Any additive two-factor representation produces this picture,
                whatever its real topology. Step 7 shows the topology is NOT toroidal
                (persistence 0.039/0.035 against nulls at 0.23/0.30), so read the panel as "the
                two factors are independent coordinates", not as "the cloud is a torus".
  helix         day of month projected onto its 31-cycle plane and its linear direction.

  python scripts/05_summary_figures.py --model gemma-2-2b --dtype bfloat16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cf.intervene import circle_basis, item_angles  # noqa: E402
from cf.model import MODELS, item_activations, load  # noqa: E402
from cf.periodic import design_matrix  # noqa: E402
from cf.prompts import MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)

    # circles for each factor on its own
    Xw, _ = item_activations(model, WEEKDAYS, layer, args.hook, device=device)
    Xm, _ = item_activations(model, MONTHS, layer, args.hook, device=device)
    uw, vw, cw = circle_basis(Xw)
    um, vm, cm = circle_basis(Xm)

    # joint prompts, read at the last token
    name = f"blocks.{layer}.hook_{args.hook}"
    pts, wi, mi = [], [], []
    for i, d in enumerate(WEEKDAYS.items):
        for j, mo in enumerate(MONTHS.items):
            p = f"The date is {d}, {mo} the 3rd"
            toks = model.to_tokens(p, prepend_bos=True)
            _, cache = model.run_with_cache(toks, names_filter=[name])
            pts.append(cache[name][0, -1].float().cpu().numpy()); wi.append(i); mi.append(j)
    P = np.stack(pts); wi = np.array(wi); mi = np.array(mi)

    # The torus claim rests on the ADDITIVE decomposition of the joint cloud, so read the angles
    # off the fitted main effects rather than off the raw last-token vectors (whose projection
    # onto a single factor's plane spans only a narrow arc).
    Pc = P - P.mean(0, keepdims=True)
    A = np.zeros((len(P), 7)); A[np.arange(len(P)), wi] = 1
    B = np.zeros((len(P), 12)); B[np.arange(len(P)), mi] = 1
    Wa = np.linalg.lstsq(A, Pc, rcond=None)[0]     # [7, d]   weekday main effect per level
    Wb = np.linalg.lstsq(B, Pc, rcond=None)[0]     # [12, d]  month main effect per level
    ua, va, ca = circle_basis(Wa)
    ub, vb, cb = circle_basis(Wb)
    ang_a = np.arctan2((Wa - ca) @ va, (Wa - ca) @ ua)
    ang_b = np.arctan2((Wb - cb) @ vb, (Wb - cb) @ ub)
    th_w, th_m = ang_a[wi], ang_b[mi]

    fig = plt.figure(figsize=(13.5, 4.2))

    # ---- panel 1: each factor's main effect is itself a circle
    ax = fig.add_subplot(1, 3, 1)
    for W, u_, v_, c_, lab, col, marker in ((Wa, ua, va, ca, WEEKDAYS.items, ACCENT, "o"),
                                            (Wb, ub, vb, cb, MONTHS.items, DARK, "s")):
        xy = np.stack([(W - c_) @ u_, (W - c_) @ v_], axis=1)
        xy = xy / np.linalg.norm(xy, axis=1).mean()
        ax.plot(np.append(xy[:, 0], xy[0, 0]), np.append(xy[:, 1], xy[0, 1]), "-",
                color=col, lw=1, alpha=0.6)
        ax.scatter(xy[:, 0], xy[:, 1], s=24, color=col, marker=marker, zorder=3)
        for (x, y), t in zip(xy, lab):
            ax.annotate(t[:3], (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7,
                        color=col)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Additive main effects: weekday (blue), month (black).\n"
                 "Ordered loops, but less clean than in single-factor prompts.", fontsize=9.5)

    # ---- panel 2: the two factors as independent coordinates, drawn on a torus we supply
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    R, r = 3.0, 1.0
    x = (R + r * np.cos(th_w)) * np.cos(th_m)
    y = (R + r * np.cos(th_w)) * np.sin(th_m)
    z = r * np.sin(th_w)
    tu, tv = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 60))
    ax.plot_surface((R + r * np.cos(tu)) * np.cos(tv), (R + r * np.cos(tu)) * np.sin(tv),
                    r * np.sin(tu), color="0.9", alpha=0.25, linewidth=0, shade=False)
    ax.scatter(x, y, z, c=mi, cmap="twilight", s=22, depthshade=False)
    ax.set_box_aspect((1, 1, 0.45)); ax.set_axis_off()
    ax.set_title("Two independent coordinates, drawn on a torus.\n"
                 "The surface is imposed, not measured (see step 7).", fontsize=9.5)

    # ---- panel 3: the helix for day of month
    from importlib import import_module
    DAY = import_module("04_multicycle").DAY_NUMBERS
    Xd, _ = item_activations(model, DAY, layer, args.hook, device=device)
    vals = np.arange(1, 32, dtype=float)
    D, names = design_matrix(vals, [31.0], linear=True)
    Xc = Xd - Xd.mean(0, keepdims=True)
    W, *_ = np.linalg.lstsq(D, Xc, rcond=None)
    # W is [n_regressors, d_model]; rows 2 and 3 are the cos/sin directions of the 31-cycle,
    # row 1 is the linear direction. Orthonormalise them, then read off each item's coordinates.
    B, _ = np.linalg.qr(np.stack([W[2], W[3], W[1]], axis=1))   # [d_model, 3]
    coords = Xc @ B
    circ, lin = coords[:, :2], coords[:, 2]

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot(circ[:, 0], circ[:, 1], lin, "-", color=GREY, lw=1)
    sc = ax.scatter(circ[:, 0], circ[:, 1], lin, c=vals, cmap="viridis", s=26, depthshade=False)
    for i in (0, 9, 19, 30):
        ax.text(circ[i, 0], circ[i, 1], lin[i], f" {int(vals[i])}", fontsize=8)
    ax.set_xlabel("31-cycle", labelpad=-8); ax.set_ylabel("31-cycle", labelpad=-8)
    ax.set_zlabel("linear", labelpad=-8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("Day of month: a circle plus\na linear ramp, i.e. a helix", fontsize=9.5)

    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: the geometry of dates",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "figures" / f"geometry_{args.model}_L{layer}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
