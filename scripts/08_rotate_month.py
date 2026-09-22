"""Step 8: rotate the month circle inside a full date prompt.

Step 3 rotated a single-factor circle in a single-factor prompt. This asks the harder question:
in a prompt that mentions both a weekday and a month, does rotating the *month* subspace at the
month token move the model's month reasoning by the right number of months, while leaving the
weekday alone?

Two things are being tested at once:
  * does the joint geometry keep the two factors separable enough to intervene on one of them,
  * is the month circle used for month arithmetic in a multi-factor context.

Controls: rotate in a variance-matched plane (PC3-PC6 of the month main effect), and check the
weekday answer is undisturbed by a month rotation (it should be, if the factors are independent).

  python scripts/08_rotate_month.py --model gemma-2-2b --dtype bfloat16
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
                          pc_plane, rotation_matrix)
from cf.model import MODELS, item_activations, item_token_span, load  # noqa: E402
from cf.prompts import MONTHS, WEEKDAYS  # noqa: E402

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
DARK, GREY, ACCENT = "0.25", "0.65", "#1f77b4"

# Prompts that mention both factors but ask only about the month.
TASKS = [
    ("It was {a}, {b}. The next month is", 1),
    ("On {a} in {b}. One month later it was", 1),
    ("It was {a}, {b}. The previous month was", -1),
]


def first_token_ids(model, words: list[str]) -> list[int]:
    ids = [int(model.to_tokens(" " + w, prepend_bos=False)[0][0]) for w in words]
    assert len(set(ids)) == len(ids), "items not distinguished by their first token"
    return ids


@torch.no_grad()
def logits_over(model, prompt: str, ids: list[int], fwd_hooks=None) -> np.ndarray:
    toks = model.to_tokens(prompt, prepend_bos=True)
    out = model.run_with_hooks(toks, fwd_hooks=fwd_hooks) if fwd_hooks else model(toks)
    return out[0, -1, ids].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--weekdays", nargs="*", default=["Monday", "Thursday", "Saturday"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)
    months, days = MONTHS.items, args.weekdays
    m_ids, d_ids = first_token_ids(model, months), first_token_ids(model, WEEKDAYS.items)

    # month circle, fitted on single-factor prompts as before
    Xm, _ = item_activations(model, MONTHS, layer, args.hook, device=device)
    u, v, centre = circle_basis(Xm)
    ang = item_angles(Xm, u, v, centre)
    step = 2 * np.pi / 12
    dirn = np.sign(np.median(np.diff(np.unwrap(ang))))
    print(f"{args.model} layer {layer}: month circle, {np.median(np.diff(np.unwrap(ang)))/np.pi:.3f}pi "
          f"per month (ideal {2/12:.3f}pi)")

    print("\nbaseline:")
    base_acc = []
    for t, off in TASKS:
        ok = sum(int(np.argmax(logits_over(model, t.format(a=d, b=mo), m_ids))) == (j + off) % 12
                 for d in days for j, mo in enumerate(months))
        base_acc.append(ok / (len(days) * len(months)))
        print(f"  {t.format(a='D', b='M'):<44} offset {off:+d}  accuracy {base_acc[-1]:.2f}")

    hook = RotationHook(model, layer, args.hook)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    results = {}

    for cond, plane in (("month circle", (u, v)), ("PC3-PC4 control", pc_plane(Xm, 2, 3)),
                        ("PC5-PC6 control", pc_plane(Xm, 4, 5))):
        pu, pv = plane
        month_tab = np.zeros((len(shifts), 12))
        weekday_changed = np.zeros(len(shifts))
        n = 0
        for si, k in enumerate(shifts):
            R = rotation_matrix(pu, pv, dirn * k * step)
            for t, off in TASKS:
                for d in days:
                    for j, mo in enumerate(months):
                        p = t.format(a=d, b=mo)
                        _, me = item_token_span(model, p, mo)
                        hook.set(R, [me - 1], centre, device, dtype)
                        pred = int(np.argmax(logits_over(model, p, m_ids, [(hook.name, hook)])))
                        month_tab[si, (pred - (j + off)) % 12] += 1
                        # does a month rotation disturb the weekday reading?
                        pw = f"It was {d}, {mo}. The day of the week was"
                        _, me2 = item_token_span(model, pw, mo)
                        hook.set(R, [me2 - 1], centre, device, dtype)
                        wd = int(np.argmax(logits_over(model, pw, d_ids, [(hook.name, hook)])))
                        weekday_changed[si] += (WEEKDAYS.items[wd] != d)
                        n += 1
            month_tab[si] /= month_tab[si].sum()
        hook.R = None
        on_target = [month_tab[si, k % 12] for si, k in enumerate(shifts)]
        nz = [i for i, k in enumerate(shifts) if k != 0]
        results[cond] = {"table": month_tab.tolist(),
                         "on_target_nonzero": float(np.mean([on_target[i] for i in nz])),
                         "weekday_disturbed": (weekday_changed / (n / len(shifts))).tolist()}
        print(f"\n{cond}: month answer shifts by exactly k (k!=0) in "
              f"{results[cond]['on_target_nonzero']:.3f} of trials (chance 1/12 = 0.083)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(results)
    for ax, cond in zip(axes[:2], [names[0], names[1]]):
        tab = np.array(results[cond]["table"])
        im = ax.imshow(tab, cmap="Greys", vmin=0, vmax=max(0.5, tab.max()), aspect="auto")
        ax.set_xticks(range(0, 12, 2)); ax.set_yticks(range(len(shifts)))
        ax.set_yticklabels(shifts)
        ax.set_xlabel("observed shift (months)"); ax.set_ylabel("rotation applied (months)")
        for si, k in enumerate(shifts):
            ax.add_patch(plt.Rectangle((k % 12 - 0.5, si - 0.5), 1, 1, fill=False, ec=ACCENT, lw=1.4))
        ax.set_title(cond, fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axes[2]
    for cond, style, col in zip(names, ["-o", "--s", ":^"], [ACCENT, DARK, GREY]):
        tab = np.array(results[cond]["table"])
        ax.plot(shifts, [tab[si, k % 12] for si, k in enumerate(shifts)], style, color=col,
                ms=4, label=cond)
    ax.axhline(1 / 12, color=DARK, ls=":", lw=1, label="chance")
    ax.set_xlabel("rotation applied (months)")
    ax.set_ylabel("fraction shifted by exactly that much")
    ax.set_title("Rotating the month circle in a date prompt", fontsize=9.5)
    ax.legend(fontsize=7.5)

    fig.suptitle(f"{args.model}, blocks.{layer}.hook_{args.hook}: intervening on one factor "
                 f"of the joint geometry", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = ROOT / "figures" / f"rotate_month_{args.model}_L{layer}.png"
    fig.savefig(path, bbox_inches="tight")
    (ROOT / "data" / f"rotate_month_{args.model}_L{layer}.json").write_text(
        json.dumps({"baseline": base_acc, "shifts": shifts, "results": results}, indent=2,
                   default=float))
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
