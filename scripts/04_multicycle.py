"""Step 4: more than one cycle at a time. Helices, tori, and what the topology says.

Three experiments:

  helix    Day of month (1..31) carries a linear ramp, a 31-cycle, a 7-cycle (weeks) and
           possibly base-10 structure. Which periods are actually there? A linear component
           plus a circular one is a helix.

  torus    Weekday crossed with month (7 x 12 = 84 prompts). If the joint representation is
           additive and the two circles occupy independent directions, the point cloud is a
           torus. Measured as: interaction variance, principal angles between the two circle
           planes, and persistent homology of the joint cloud.

  control  The same analyses on an arbitrary pairing (animals x objects), where no product
           structure should appear.

  python scripts/04_multicycle.py --model gemma-2-2b-it --dtype bfloat16
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

from cf.model import MODELS, item_activations, load  # noqa: E402
from cf.periodic import (betti_numbers, circle_plane_of, fourier_spectrum,  # noqa: E402
                         spectrum_null, subspace_angles, two_way_decomposition)
from cf.prompts import ItemSet, MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"

# Neutral numeric contexts (no ordinal suffixes). Gemma tokenises numbers digit by digit, so for
# Gemma the vector is read at the units digit; 10_day_of_month.py repeats this on GPT-2, where
# 1-31 are single tokens.
DAY_NUMBERS = ItemSet(
    "day of month", [str(i) for i in range(1, 32)],
    ["The date is {item}", "We met on day {item}", "It happened on day {item}",
     "Room {item}", "Chapter {item}"], kind="ordered",
)


def pair_set(name: str, a: list[str], b: list[str], template: str) -> ItemSet:
    """A 2-factor item set: every (a, b) combination in one prompt."""
    items = [f"{x}|{y}" for x in a for y in b]
    s = ItemSet(name, items, ["{item}"], kind="ordered")
    s.templates = [template]
    s.prompts = lambda: [(it, template.format(a=it.split("|")[0], b=it.split("|")[1]))
                         for it in items]
    return s


def pair_activations(model, name, a, b, template, layer, hook, device):
    """Activations for every (a, b) pair, read at the last token of the prompt."""
    acts = []
    for x in a:
        for y in b:
            prompt = template.format(a=x, b=y)
            tokens = model.to_tokens(prompt, prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_{hook}"])
            acts.append(cache[f"blocks.{layer}.hook_{hook}"][0, -1].float().cpu().numpy())
    X = np.stack(acts)
    a_idx = np.repeat(np.arange(len(a)), len(b))
    b_idx = np.tile(np.arange(len(b)), len(a))
    return X, a_idx, b_idx


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
    out: dict = {"model": args.model, "layer": layer}

    # ------------------------------------------------------------- 1. helix
    print("=== periods present in 'day of month' (1-31) ===")
    Xd, _ = item_activations(model, DAY_NUMBERS, layer, args.hook, device=device)
    vals = np.arange(1, 32, dtype=float)
    periods = [2, 3, 5, 7, 10, 12, 31]
    spec = fourier_spectrum(Xd, vals, periods, linear=True)
    null = spectrum_null(Xd, vals, periods, linear=True)
    out["day_of_month_spectrum"] = spec
    out["day_of_month_null"] = null
    print(f"  full model R^2 = {spec['r2_full']:.3f}")
    print(f"    {'component':<12} {'unique':>8} {'null p95':>10}  verdict")
    for k, v in sorted(spec["unique"].items(), key=lambda kv: -kv[1]):
        ok = "REAL" if v > null[k]["p95"] else "fit noise"
        print(f"    {k:<12} {v:+8.3f} {null[k]['p95']:10.3f}  {ok}")

    # ------------------------------------------------------------- 2. torus
    print("\n=== weekday x month: is the joint representation a product? ===")
    Xp, a_idx, b_idx = pair_activations(
        model, "weekday x month", WEEKDAYS.items, MONTHS.items,
        "The date is {a}, {b} the 3rd", layer, args.hook, device)
    dec = two_way_decomposition(Xp, a_idx, b_idx)
    print(f"  weekday main effect   {dec['factor_a']:.3f} of variance")
    print(f"  month main effect     {dec['factor_b']:.3f}")
    print(f"  additive model R^2    {dec['additive_model_r2']:.3f}")
    print(f"  interaction left over {dec['interaction']:.3f}   (small = clean product)")

    Xw, _ = item_activations(model, WEEKDAYS, layer, args.hook, device=device)
    Xm, _ = item_activations(model, MONTHS, layer, args.hook, device=device)
    angles = subspace_angles(circle_plane_of(Xw), circle_plane_of(Xm))
    print(f"  principal angles between the two circle planes: {np.round(angles, 1)} degrees")
    print("    (near 90 = independent directions, which a torus needs)")

    topo = betti_numbers(Xp, maxdim=2)
    print(f"  persistent homology of the 84 joint points:")
    for dim, d in topo.items():
        if isinstance(d, dict):
            print(f"    {dim}: {d['n_classes']} classes, longest lifetimes "
                  f"{np.round(d['top_lifetimes'], 3)}")
    out["torus"] = {k: v for k, v in dec.items() if k != "residual_after_additive"}
    out["torus"]["principal_angles_deg"] = angles.tolist()
    out["torus"]["topology"] = topo

    # circle topology for reference: what does ONE circle look like?
    topo_w = betti_numbers(Xw, maxdim=1)
    print(f"  for reference, the 7 weekday points alone: "
          f"H1 lifetimes {np.round(topo_w['H1']['top_lifetimes'], 3)}")
    out["weekday_topology"] = topo_w

    # ----------------------------------------------------------- 3. control
    print("\n=== control: an arbitrary pairing ===")
    from cf.prompts import ANIMALS, OBJECTS
    Xc, ca, cb = pair_activations(model, "control pair", ANIMALS.items, OBJECTS.items,
                                  "The {a} and the {b}", layer, args.hook, device)
    dec_c = two_way_decomposition(Xc, ca, cb)
    print(f"  additive model R^2 {dec_c['additive_model_r2']:.3f}, "
          f"interaction {dec_c['interaction']:.3f}")
    topo_c = betti_numbers(Xc, maxdim=2)
    for dim, d in topo_c.items():
        if isinstance(d, dict):
            print(f"    {dim}: longest lifetimes {np.round(d['top_lifetimes'], 3)}")
    out["control_pair"] = {k: v for k, v in dec_c.items() if k != "residual_after_additive"}
    out["control_pair"]["topology"] = topo_c

    # ------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    keys = list(spec["unique"].keys())
    vals_u = [spec["unique"][k] for k in keys]
    order = np.argsort(vals_u)[::-1]
    ax.bar(range(len(keys)), [vals_u[i] for i in order], color=GREY, edgecolor=DARK)
    ax.plot(range(len(keys)), [null[keys[i]]["p95"] for i in order], "_", color=ACCENT,
            ms=14, mew=2, label="shuffled-label 95th pct")
    ax.legend(fontsize=7.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([keys[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("unique variance explained")
    ax.set_title(f"'Day of month' 1-31: which periods?\nfull model $R^2$ = {spec['r2_full']:.2f}",
                 fontsize=9.5)

    ax = axes[1]
    parts = ["weekday\nmain effect", "month\nmain effect", "interaction"]
    ax.bar(range(3), [dec["factor_a"], dec["factor_b"], dec["interaction"]],
           color=[ACCENT, ACCENT, GREY], edgecolor=DARK)
    ax.bar(range(3), [dec_c["factor_a"], dec_c["factor_b"], dec_c["interaction"]],
           width=0.35, color="none", edgecolor=DARK, ls="--", label="control pairing")
    ax.set_xticks(range(3)); ax.set_xticklabels(parts, fontsize=8)
    ax.set_ylabel("fraction of variance")
    ax.set_title("Weekday x month: additive?", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[2]
    h1 = topo["H1"]["top_lifetimes"][:4] if "H1" in topo else []
    h1c = topo_c["H1"]["top_lifetimes"][:4] if "H1" in topo_c else []
    w = 0.38
    ax.bar(np.arange(len(h1)) - w / 2, h1, w, color=ACCENT, edgecolor=DARK, label="weekday x month")
    ax.bar(np.arange(len(h1c)) + w / 2, h1c, w, color=GREY, edgecolor=DARK, label="control pairing")
    ax.set_xlabel("1-dimensional class (longest first)")
    ax.set_ylabel("persistence (lifetime)")
    ax.set_title("Topology: a torus needs TWO long-lived loops", fontsize=9.5)
    ax.legend(fontsize=8)

    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: multiple cycles at once",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = ROOT / "figures" / f"multicycle_{args.model}_L{layer}.png"
    fig.savefig(path, bbox_inches="tight")
    (ROOT / "data" / f"multicycle_{args.model}_L{layer}.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
