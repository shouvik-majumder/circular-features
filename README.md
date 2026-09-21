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

---

## Step 3: is the circle used, or only present?

Rotate a day's representation inside the fitted circle plane by k steps, leave everything outside
that plane untouched, and ask whether the model's answer to "Today is X. Tomorrow is" advances by
k days. Baseline accuracy on the tasks is 1.00, 1.00 and 0.57.

| Condition | answer shifts by exactly k (k != 0) | answer unchanged | mean displacement |
|---|---|---|---|
| rotate in the fitted circle | **0.222** | 0.365 | 62 |
| rotate in PC3-PC4 / PC5-PC6 (variance-matched control) | 0.060 | 0.575 | 33 |
| rotate in a random plane | ~0.14 (chance) | - | 1.5 |

Chance is 1/7 = 0.143. The circle plane moves the answer in the intended direction almost four
times as often as a control plane of comparable size, and the control mostly leaves the answer
alone. A random plane in the full space barely perturbs the activation at all, which is why it is
not the control that counts.

So the circle is **used**, not merely present, though the effect is partial rather than a clean
one-to-one rotation of the answer.

## Step 4: more than one cycle at once

**Weekday x month is a torus.** 84 prompts, one per (weekday, month) pair:

| Measure | Weekday x month | Arbitrary control pairing |
|---|---|---|
| additive model R^2 | **0.982** | 0.888 |
| interaction left over | **0.018** | 0.112 |
| principal angles between the two circle planes | 83, 89 degrees | - |
| persistent homology, two longest 1-cycles | **0.292, 0.291** | 0.063, 0.043 |

Three independent signatures agree. The joint representation is almost perfectly additive, the
two circles occupy near-orthogonal directions, and the point cloud carries **two** long-lived
loops of nearly equal persistence, which is the topological signature of a torus. A single circle
(the 7 weekday points alone) gives one loop at 0.136. The control pairing gives loops five times
shorter.

**Day of month is a helix.** Regressing the 31 item vectors on a linear ramp plus sine/cosine
pairs at several candidate periods, and testing each against a shuffled-label null:

| Component | unique variance | null 95th pct | verdict |
|---|---|---|---|
| period 3 | 0.206 | 0.178 | real |
| period 31 | 0.193 | 0.096 | real |
| linear | 0.106 | 0.051 | real |
| periods 10, 5, 12, 7, 2 | 0.033 to 0.088 | ~0.10 | fit noise |

A linear component plus a circular one is a helix, which matches
[Kantamneni & Tegmark 2025](https://arxiv.org/abs/2502.00873) for integers. The base-10 periods
they report (2, 5, 10) do **not** clear the null here, plausibly because 31 items and 17 fitted
parameters leave little power. The period-3 component is unexplained and survived a fix to the
prompt templates.

### Prior work worth knowing

- Numbers are represented as a generalised helix with periods 2, 5, 10, 100 plus a linear term,
  and the model uses it causally ("Clock" algorithm):
  [arXiv:2502.00873](https://arxiv.org/abs/2502.00873), code
  [subhashk01/LLM-addition](https://github.com/subhashk01/LLM-addition).
- Cyclic-concept arithmetic in Llama-3.1-8B runs through generic base-10 addition rather than
  concept-specific modular arithmetic: [arXiv:2605.01148](https://arxiv.org/abs/2605.01148).
- Neither studies **joint** cyclic variables or toroidal structure. That part appears open.
