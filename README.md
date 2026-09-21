# Are cyclic concepts represented on circles?

Following Engels et al., *Not All Language Model Features Are One-Dimensionally Linear*, ICLR 2025
([arXiv:2405.14860](https://arxiv.org/abs/2405.14860), code
[JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures)).

Their claim: some concepts are not represented as single directions but as genuinely
multi-dimensional objects, and days of the week and months of the year sit on **circles** that the
model uses to do modular arithmetic.

This repository builds the measurement from scratch rather than running theirs, so that every
control is explicit, and extends it to Gemma-2-2B, which the original work did not test.
Free and local: GPT-2 is a 0.5 GB ungated download, Gemma-2-2B is already cached from an earlier
project.

## Setup

```powershell
conda activate circfeat
cd D:\dev\circular-features
python scripts/01_find_circles.py
```

Environment (`circfeat`, Python 3.12):

```powershell
conda create -n circfeat python=3.12 -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformer_lens sae-lens transformers numpy scipy matplotlib pandas tqdm scikit-learn einops python-dotenv
```

Gemma runs need the gated weights, so set `HF_HOME` to the shared cache and export `HF_TOKEN`
from `D:\dev\ESR\.env`.

## Scripts

| Script | What it does | Cost |
|---|---|---|
| `01_find_circles.py` | Item vectors, PCA, circularity measures against two nulls, figure | ~1 min GPT-2, ~3 min Gemma |
| `02_layer_sweep.py` | The same three measures at every layer | ~5 min |

```powershell
python scripts/01_find_circles.py                                  # gpt2, layer 7
python scripts/01_find_circles.py --model gemma-2-2b --dtype bfloat16
python scripts/02_layer_sweep.py --hook resid_pre                  # includes the raw embedding
```

## How it is measured

Each item ("Monday") is placed in five short templates, the residual stream is read at the last
token of the item, and the five are averaged. The seven item vectors are then projected onto
their own first two principal components and scored on:

- **pc12_variance** how much of the variance those two components hold, so the structure is flat
- **radial_cv** spread of the radius about the centre; 0 is a perfect ring
- **order_score** how much of the calendar order survives as angular order around the ring

None of those mean anything alone. Seven arbitrary vectors projected onto their own top two
components often look ring-shaped, because the projection is chosen to maximise spread. So each
number is reported against two nulls, and against item sets where the answer should be no:

- **shuffled labels** same points, permuted item names; isolates order from shape
- **isotropic Gaussian points** of the same count and dimension; isolates shape from chance
- **an ordered but non-cyclic set** (number words) where a line, not a ring, is predicted
- **two arbitrary control sets** (animals, objects) where neither is predicted

## What we found

**The dissociation is clean, in both models.**

GPT-2 small, `blocks.7.hook_resid_post`:

| Item set | PC1+PC2 | radial CV | order score | shuffled 95th |
|---|---|---|---|---|
| weekdays (cyclic) | 0.69 | **0.15** | **1.00** | 0.71 |
| months (cyclic) | 0.44 | **0.14** | **1.00** | 0.42 |
| number words (ordered) | 0.59 | 0.63 | 1.00 | 0.50 |
| animals (control) | 0.44 | 0.62 | 0.43 | 0.71 |
| objects (control) | 0.44 | 0.43 | 0.57 | 0.71 |

Gemma-2-2B, `blocks.16.hook_resid_post`: weekdays 0.19 / 1.00, months 0.20 / 1.00, controls at or
below chance. **The circles replicate in a model the original paper did not test.**

The number-words row is the one that makes the result meaningful. It is perfectly *ordered* but
not at all *circular*, which is exactly what an ordinal, non-cyclic concept should look like. The
measures are therefore distinguishing ring from line, not just structure from noise.

**But most of the circle is in the token embeddings.** Reading the residual stream *before* any
attention or MLP has run (`blocks.0.hook_resid_pre`), weekdays already score radial CV 0.23 with
order 1.00. Through the network it sharpens to 0.14 by layer 3 and holds. So the network refines
a circle it largely inherits from the embedding rather than constructing one.

That is the same lesson as the previous two projects: the interesting quantity is not the
headline number but the gap between it and a baseline with the same access to the data.

## Next

1. **Does the model use the circle?** Add a vector along the circle's tangent and test whether
   "two days after Monday is" moves the prediction by two positions around the ring. That is the
   step that turns geometry into mechanism, and it reuses the steering hook from the ESR project.
2. **Find it unsupervised.** The paper's actual contribution was discovering these circles from
   SAE feature clustering rather than being told which words to look at. `sae-lens` is installed
   and Gemma Scope is cached.
3. **Is the circle an attractor?** Push the state off the ring and see whether later tokens fall
   back onto it.
4. **Other cyclic sets** with no lexical ordering cue: compass directions, musical notes,
   seasons, clock hours.
