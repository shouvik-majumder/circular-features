"""Step 9: did the month intervention actually land?

Step 8 found that rotating the month circle inside a date prompt does not shift the month
answer. Two very different things produce that result:

  (a) the circle is not used as a rotatable coordinate for month arithmetic in this context,
  (b) the rotation never really perturbed the activation, so nothing was tested.

(b) is a live worry because the circle is fitted on single-factor prompts ("It is March") and
applied in two-factor ones ("It was Monday, March. The next month is"). If the joint-prompt
month activation sits near the fitted centre in that plane, rotating about the centre moves it
hardly at all.

This measures, for both weekdays (step 3, which worked) and months (step 8, which did not):

  in-plane radius   how far the activation sits from the fitted centre inside the circle plane
  displacement      how far a k-step rotation actually moves the activation, in L2
  relative          that displacement as a fraction of the activation norm

  python scripts/09_intervention_diagnostic.py --model gemma-2-2b --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cf.intervene import circle_basis, pc_plane, rotation_matrix  # noqa: E402
from cf.model import MODELS, item_activations, item_token_span, load  # noqa: E402
from cf.prompts import MONTHS, WEEKDAYS  # noqa: E402

JOINT = "It was {a}, {b}. The next month is"
SINGLE_W = "Today is {a}. Tomorrow is"


@torch.no_grad()
def acts_at(model, prompts_and_items, layer, hook, device):
    """Residual stream at the last token of `item` inside `prompt`, for each pair."""
    name = f"blocks.{layer}.hook_{hook}"
    out = []
    for p, item in prompts_and_items:
        toks = model.to_tokens(p, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=[name])
        _, e = item_token_span(model, p, item)
        out.append(cache[name][0, e - 1].float().cpu().numpy())
    return np.stack(out)


def probe(label, X_fit, X_use, step_frac, rng):
    """Fit the circle on X_fit, report how much rotating moves the vectors in X_use."""
    u, v, c = circle_basis(X_fit)
    rec = {}
    for plane_name, (pu, pv) in (("circle", (u, v)),
                                 ("PC3-PC4", pc_plane(X_fit, 2, 3)),
                                 ("PC5-PC6", pc_plane(X_fit, 4, 5))):
        # radius inside the plane, measured about the fitted centre
        r_fit = np.linalg.norm(np.stack([(X_fit - c) @ pu, (X_fit - c) @ pv], 1), axis=1)
        r_use = np.linalg.norm(np.stack([(X_use - c) @ pu, (X_use - c) @ pv], 1), axis=1)
        disp = []
        for k in (1, 2, 3):
            R = rotation_matrix(pu, pv, 2 * np.pi * k * step_frac)
            Y = (X_use - c) @ R.T + c
            disp.append(float(np.linalg.norm(Y - X_use, axis=1).mean()))
        rec[plane_name] = {
            "radius_in_fitting_prompts": float(r_fit.mean()),
            "radius_in_use_prompts": float(r_use.mean()),
            "displacement_k1_k2_k3": disp,
            "relative_displacement_k1": float(disp[0] / np.linalg.norm(X_use, axis=1).mean()),
        }
        print(f"  {label:<10} {plane_name:<8} radius fit {r_fit.mean():7.2f}  "
              f"use {r_use.mean():7.2f}  displacement k=1 {disp[0]:7.2f}  "
              f"({rec[plane_name]['relative_displacement_k1']*100:.1f}% of |x|)")
    return rec


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
    rng = np.random.default_rng(0)
    out = {"model": args.model, "layer": layer}

    print(f"{args.model} layer {layer}\n")

    # months: fitted on single-factor prompts, used in joint date prompts (step 8)
    Xm_fit, _ = item_activations(model, MONTHS, layer, args.hook, device=device)
    Xm_use = acts_at(model, [(JOINT.format(a="Monday", b=mo), mo) for mo in MONTHS.items],
                     layer, args.hook, device)
    print("months, fitted on single-factor prompts, applied in date prompts (step 8):")
    out["months_single_fit"] = probe("month", Xm_fit, Xm_use, 1 / 12, rng)

    # the same, but fitted on the joint prompts themselves
    print("\nmonths, fitted on the date prompts themselves:")
    out["months_joint_fit"] = probe("month", Xm_use, Xm_use, 1 / 12, rng)

    # weekdays: the step 3 setting, which did produce an effect
    Xw_fit, _ = item_activations(model, WEEKDAYS, layer, args.hook, device=device)
    Xw_use = acts_at(model, [(SINGLE_W.format(a=d), d) for d in WEEKDAYS.items],
                     layer, args.hook, device)
    print("\nweekdays, the step 3 setting (which did move the answer):")
    out["weekdays"] = probe("weekday", Xw_fit, Xw_use, 1 / 7, rng)

    (ROOT / "data" / f"intervention_diagnostic_{args.model}_L{layer}.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\n-> data/intervention_diagnostic_{args.model}_L{layer}.json")


if __name__ == "__main__":
    main()
