"""Loading models and collecting one activation vector per item.

We use TransformerLens for the small models because `run_with_cache` names every internal
activation, and the naming convention (`blocks.7.hook_resid_post`) is the same one used by
Gemma Scope and by the ESR project.

The only subtlety worth knowing: an item like "Wednesday" may be several tokens. We take the
activation at the **last token of the item**, which is where the model has finished reading the
word, and average over the templates the item appeared in.
"""
from __future__ import annotations

import numpy as np
import torch

MODELS = {
    # name -> (TransformerLens model id, a sensible middle layer to look at first)
    "gpt2": ("gpt2", 7),
    "gpt2-medium": ("gpt2-medium", 12),
    "gemma-2-2b": ("google/gemma-2-2b-it", 16),
}


def load(model_name: str, device: str = "cuda", dtype: str = "float32"):
    from transformer_lens import HookedTransformer

    hf_id, _ = MODELS[model_name]
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model = HookedTransformer.from_pretrained(hf_id, device=device, dtype=torch_dtype)
    model.eval()
    return model


def item_token_span(model, prompt: str, item: str) -> tuple[int, int]:
    """Index range of the item's tokens inside the tokenised prompt.

    Found by matching the tokenisation of the prefix that ends at the item, which avoids the
    usual leading-space and sub-word pitfalls of string matching on token strings.
    """
    start_char = prompt.index(item)
    n_before = len(model.to_tokens(prompt[:start_char], prepend_bos=True)[0])
    n_through = len(model.to_tokens(prompt[: start_char + len(item)], prepend_bos=True)[0])
    return n_before, n_through  # [start, end)


@torch.no_grad()
def item_activations(model, item_set, layer: int, hook: str = "resid_post",
                     device: str = "cuda") -> tuple[np.ndarray, list[str]]:
    """Mean activation per item, averaged over that item's templates.

    Returns (activations [n_items, d_model], item names in the set's own order).
    """
    name = f"blocks.{layer}.hook_{hook}"
    per_item: dict[str, list[np.ndarray]] = {it: [] for it in item_set.items}
    for item, prompt in item_set.prompts():
        tokens = model.to_tokens(prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, names_filter=[name])
        acts = cache[name][0].float().cpu().numpy()      # [seq, d_model]
        _, end = item_token_span(model, prompt, item)
        per_item[item].append(acts[end - 1])              # last token of the item
    items = list(item_set.items)
    return np.stack([np.mean(per_item[i], axis=0) for i in items]), items


@torch.no_grad()
def layer_sweep(model, item_set, hook: str = "resid_post") -> dict[int, np.ndarray]:
    """Activations at every layer, for asking where in the network the structure appears."""
    return {L: item_activations(model, item_set, L, hook)[0] for L in range(model.cfg.n_layers)}
