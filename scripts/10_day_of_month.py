"""Step 10: day of month without the tokenisation confound.

Step 4 fitted periods to the 31 day-of-month vectors in Gemma. But Gemma's tokenizer splits every
number into single digits: "31" is " ", "3", "1". Reading at the last token of the item therefore
reads the UNITS DIGIT, with the tens digit visible only through attention, and 1-9 are one token
while 10-31 are two. Any period structure found that way is partly a statement about digit
tokens, and it cannot be compared with Kantamneni & Tegmark, whose models see each number as a
single token.

GPT-2 keeps " 1" ... " 31" as single tokens, so it is the clean test. This script runs the same
spectrum on any model, records how every item was tokenised, and refuses to call a model clean
unless all 31 items are single tokens.

  python scripts/10_day_of_month.py --model gpt2
  python scripts/10_day_of_month.py --model gemma-2-2b-it --dtype bfloat16
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
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module  # noqa: E402

from cf.model import MODELS, item_activations, load  # noqa: E402
from cf.periodic import fourier_spectrum, spectrum_null  # noqa: E402

DAY_NUMBERS = import_module("04_multicycle").DAY_NUMBERS
PERIODS = [2, 3, 5, 7, 10, 12, 31]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--hook", default="resid_post")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = args.layer if args.layer is not None else MODELS[args.model][1]
    model = load(args.model, device=device, dtype=args.dtype)

    # how is each number tokenised in context? (the templates put a space before the number)
    n_tok = {it: len(model.to_tokens(" " + it, prepend_bos=False)[0]) for it in DAY_NUMBERS.items}
    pieces = {it: [model.tokenizer.decode([t]) for t in model.to_tokens(" " + it, prepend_bos=False)[0].tolist()]
              for it in ("3", "12", "31")}
    single = all(n == 1 for n in n_tok.values())
    print(f"{args.model} layer {layer}: tokenisation examples {pieces}")
    print(f"  all 31 items single tokens: {single}"
          + ("" if single else f"  (token counts: {sorted(set(n_tok.values()))})"))

    X, _ = item_activations(model, DAY_NUMBERS, layer, args.hook, device=device)
    vals = np.arange(1, 32, dtype=float)
    spec = fourier_spectrum(X, vals, PERIODS, linear=True)
    null = spectrum_null(X, vals, PERIODS, linear=True)
    print(f"  full model R^2 = {spec['r2_full']:.3f}")
    print(f"    {'component':<12} {'unique':>8} {'null p95':>10}  verdict")
    rows = []
    for k, v in sorted(spec["unique"].items(), key=lambda kv: -kv[1]):
        ok = v > null[k]["p95"]
        rows.append({"component": k, "unique": v, "null_p95": null[k]["p95"], "real": bool(ok)})
        print(f"    {k:<12} {v:+8.3f} {null[k]['p95']:10.3f}  {'REAL' if ok else 'fit noise'}")
    if not single:
        print("  CAUTION: items are multi-token, read at the last digit; see the module docstring.")

    out = {"model": args.model, "layer": layer, "hook": args.hook, "single_token_items": single,
           "token_counts": n_tok, "examples": pieces, "r2_full": spec["r2_full"], "components": rows}
    path = ROOT / "data" / f"day_of_month_{args.model}_L{layer}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
