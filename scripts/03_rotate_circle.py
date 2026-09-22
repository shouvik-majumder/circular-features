"""Step 3: is the circle used, or merely present?

Take a prompt whose answer is a weekday ("Today is Monday. Tomorrow is"), rotate the
representation of the *input* day by k steps inside the fitted circle plane, and ask whether the
model's answer advances by k days. Everything outside that 2D plane is left untouched.

  present  = a probe can read the day off the circle (step 1 showed this)
  used     = rotating the circle changes the answer by the matching amount

Controls: the same rotation in the PC3-PC4 and PC5-PC6 planes of the same item vectors (these
displace the activation by a comparable amount, so they are the controls that count), and in
random 2D planes (which barely displace it, reported for completeness).

The shift is measured against the model's OWN unrotated answer for that prompt, not against the
correct answer. Otherwise baseline mistakes (one task is only 0.57 accurate) leak into the table
and can masquerade as, or mask, a rotation effect. The old correct-answer table is kept alongside.

  python scripts/03_rotate_circle.py --model gemma-2-2b-it --dtype bfloat16 --layer 16
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

from cf.intervene import (RotationHook, circle_basis, item_angles,  # noqa: E402
                          pc_plane, random_plane, rotation_matrix)
from cf.model import MODELS, item_activations, item_token_span, load  # noqa: E402
from cf.prompts import WEEKDAYS  # noqa: E402
from cf.stats import wilson  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"

TASKS = [
    ("Today is {item}. Tomorrow is", 1),
    ("Today is {item}. Yesterday was", -1),
    ("Today is {item}. The day after tomorrow is", 2),
]


def day_token_ids(model, days: list[str]) -> list[int]:
    """First token of ' Monday', ' Tuesday', ... so we can read a distribution over days."""
    ids = []
    for d in days:
        toks = model.to_tokens(" " + d, prepend_bos=False)[0]
        ids.append(int(toks[0]))
    assert len(set(ids)) == len(ids), "day names are not distinguished by their first token"
    return ids


@torch.no_grad()
def day_logits(model, prompt: str, ids: list[int], fwd_hooks=None) -> np.ndarray:
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if fwd_hooks:
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    else:
        logits = model(tokens)
    return logits[0, -1, ids].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random-planes", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)
    days = WEEKDAYS.items
    ids = day_token_ids(model, days)
    rng = np.random.default_rng(args.seed)

    # The circle, fitted exactly as in step 1.
    X, _ = item_activations(model, WEEKDAYS, layer, args.hook, device=device)
    u, v, centre = circle_basis(X)
    ang = item_angles(X, u, v, centre)
    step = 2 * np.pi / 7
    # Sign convention: does angle increase or decrease as the week advances?
    dirn = np.sign(np.median(np.diff(np.unwrap(ang))))
    print(f"{args.model} layer {layer}: circle fitted, angular step per day "
          f"{np.median(np.diff(np.unwrap(ang))) / np.pi:.3f}pi (ideal {2/7:.3f}pi)")

    # ---------------------------------------------------------------- baseline
    print("\nbaseline (no intervention):")
    baseline_ok = []
    base_pred = {}                                   # (task index, day index) -> unrotated answer
    for ti, (template, offset) in enumerate(TASKS):
        correct = 0
        for i, d in enumerate(days):
            p = template.format(item=d)
            pred = int(np.argmax(day_logits(model, p, ids)))
            base_pred[(ti, i)] = pred
            correct += (pred == (i + offset) % 7)
        acc = correct / len(days)
        baseline_ok.append(acc)
        print(f"  {template.format(item='X'):<45} offset {offset:+d}  accuracy {acc:.2f}")
    if max(baseline_ok) < 0.6:
        print("\n  model cannot do the task reliably; the intervention would be uninterpretable.")

    # ------------------------------------------------------------ intervention
    hook = RotationHook(model, layer, args.hook)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    results = {}
    shifts_k = list(range(-3, 4))

    conditions = ["circle", "pc34", "pc56"] + [f"random{j}" for j in range(args.n_random_planes)]
    for cond in conditions:
        if cond == "circle":
            pu, pv = u, v
        elif cond == "pc34":
            pu, pv = pc_plane(X, 2, 3)
        elif cond == "pc56":
            pu, pv = pc_plane(X, 4, 5)
        else:
            pu, pv = random_plane(len(u), rng)
        # rotation k -> distribution of observed shift, relative to the unrotated answer (primary)
        # and relative to the correct answer (kept for comparison with the first version)
        table = np.zeros((len(shifts_k), 7))
        table_vs_correct = np.zeros((len(shifts_k), 7))
        disp = []
        for ki, k in enumerate(shifts_k):
            R = rotation_matrix(pu, pv, dirn * k * step)
            disp.append(hook.displacement(X, R, centre))
            for ti, (template, offset) in enumerate(TASKS):
                for i, d in enumerate(days):
                    p = template.format(item=d)
                    _, end = item_token_span(model, p, d)
                    hook.set(R, [end - 1], centre, device, dtype)
                    pred = int(np.argmax(day_logits(model, p, ids, fwd_hooks=[(hook.name, hook)])))
                    table[ki, (pred - base_pred[(ti, i)]) % 7] += 1
                    table_vs_correct[ki, (pred - (i + offset)) % 7] += 1
        n_per_k = int(table[0].sum())
        table /= n_per_k
        table_vs_correct /= n_per_k
        nz = [ki for ki, k in enumerate(shifts_k) if k != 0]
        hits = sum(table[ki, shifts_k[ki] % 7] * n_per_k for ki in nz)
        n_nz = n_per_k * len(nz)
        lo, hi = wilson(hits, n_nz)
        results[cond] = {"table": table.tolist(), "table_vs_correct": table_vs_correct.tolist(),
                         "displacement": float(np.mean(disp)), "n_per_shift": n_per_k,
                         "on_target_nonzero": hits / n_nz, "on_target_nonzero_ci95": [lo, hi]}
        print(f"\n{cond}: answer moved by exactly k (k != 0) in {hits / n_nz:.3f} of {n_nz} trials "
              f"[95% CI {lo:.3f}-{hi:.3f}] (chance 1/7 = 0.143), "
              f"mean displacement {np.mean(disp):.1f}")

    hook.R = None

    # --------------------------------------------------------------- figure
    rand_tables = np.array([results[c]["table"] for c in results if c.startswith("random")])
    pc_table = (np.array(results["pc34"]["table"]) + np.array(results["pc56"]["table"])) / 2
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    for ax, (tab, title) in zip(axes[:2], [
        (np.array(results["circle"]["table"]),
         f"Rotate inside the fitted circle\n(mean displacement {results['circle']['displacement']:.0f})"),
        (pc_table,
         "Control: rotate in PC3-PC4 / PC5-PC6\n"
         f"(mean displacement {(results['pc34']['displacement'] + results['pc56']['displacement']) / 2:.0f})"),
    ]):
        im = ax.imshow(tab, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(7)); ax.set_xticklabels(range(7))
        ax.set_yticks(range(len(shifts_k))); ax.set_yticklabels(shifts_k)
        ax.set_xlabel("shift vs the unrotated answer (days)")
        ax.set_ylabel("rotation applied (days)")
        ax.set_title(title, fontsize=9.5)
        for ki, k in enumerate(shifts_k):
            ax.add_patch(plt.Rectangle((k % 7 - 0.5, ki - 0.5), 1, 1, fill=False, ec=ACCENT, lw=1.4))
        fig.colorbar(im, ax=ax, fraction=0.045, label="fraction of trials")

    ax = axes[2]
    circ = [np.array(results["circle"]["table"])[ki, k % 7] for ki, k in enumerate(shifts_k)]
    pcs = [pc_table[ki, k % 7] for ki, k in enumerate(shifts_k)]
    rnd = [rand_tables.mean(0)[ki, k % 7] for ki, k in enumerate(shifts_k)]
    ax.plot(shifts_k, circ, "-o", color=ACCENT, label="fitted circle")
    ax.plot(shifts_k, pcs, "--s", color=DARK, label="PC3-6 control (matched size)")
    ax.plot(shifts_k, rnd, ":^", color=GREY, label="random plane (tiny effect)")
    ax.axhline(1 / 7, color=DARK, ls=":", lw=1, label="chance")
    ax.set_xlabel("rotation applied (days)")
    ax.set_ylabel("fraction where the answer\nshifted by exactly that much")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Blue box = intended effect", fontsize=9.5)

    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: rotating the weekday circle",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "figures" / f"rotate_{args.model}_L{layer}.png"
    fig.savefig(out, bbox_inches="tight")
    (ROOT / "data" / f"rotate_{args.model}_L{layer}.json").write_text(json.dumps(
        {"model": args.model, "layer": layer, "shifts": shifts_k,
         "baseline_accuracy": baseline_ok, "results": results}, indent=2))
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
