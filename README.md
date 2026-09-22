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
| `03_rotate_circle.py` | Rotate inside the fitted circle plane; does the answer advance by k? | ~10 min |
| `04_multicycle.py` | Weekday x month joint geometry, day-of-month Fourier spectrum | ~10 min |
| `05_summary_figures.py` | Cross-project summary panels | ~5 min |
| `06_deformation.py` | Harmonics, ellipse axis ratio, angular gaps: circle or deformed loop? | ~3 min |
| `07_torus_validation.py` | Denser sampling, three read positions, cloud-perturbing nulls, depth sweep | ~25 min |
| `08_rotate_month.py` | Rotate the month subspace inside a full date prompt | ~30 min |

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

84 prompts, one per (weekday, month) pair. The first pass looked like a torus. It is not one; see
step 7. What is real:

| Measure | Weekday x month | Arbitrary control pairing |
|---|---|---|
| additive model R^2 | **0.995** | 0.888 |
| interaction left over | **0.005** | 0.112 |
| principal angles between the two circle planes | 88.5, 89.7 degrees | - |

The joint representation is almost perfectly additive and the two factors live in near-orthogonal
subspaces. That is a clean **product structure**: the model writes weekday and month into the
residual stream as independent, non-interfering summands. The claim it does *not* support is the
topological one.

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

## Step 6: is it a circle, or a deformed loop?

Three independent measures, all on the same item vectors, at `blocks.16.hook_resid_post`:

| | weekdays | months | perfect circle |
|---|---|---|---|
| power in harmonic 1 | 0.57 (null 0.45) | 0.38 (null 0.23) | 1.00 |
| variance in the best plane | 0.60 | 0.39 | 1.00 |
| ellipse axis ratio | 0.81 | 0.94 | 1.00 |
| angular gaps | 33-75 deg (ideal 51) | 12-74 deg (ideal 30) | equal |

**They are deformed loops, not circles.** Roughly half the structure is at harmonic 1; the rest is
higher harmonics, out-of-plane wobble and uneven spacing. Months are the more deformed of the two,
which is not obvious from a PCA scatter plot - the projection is chosen to make the loop look as
round as possible. The order is right, the shape is only approximately circular.

## Step 7: the torus does not survive proper nulls

The step-4 topology result was compared against one arbitrary control pairing that happened to
have unusually low persistence. Redone with denser sampling (5 templates per pair, averaged),
three read positions, and two nulls that actually perturb the point cloud:

- **matched Gaussian**: 84 points from a Gaussian with the same covariance. PCA looks the same,
  the product structure is destroyed.
- **additive random levels**: the same additive model with random vectors per level. Additive, but
  each factor is no longer circular.

At layer 16, with the two longest 1-cycle lifetimes (a torus needs two long ones):

| read position | real | Gaussian null 95th | additive-random null 95th |
|---|---|---|---|
| weekday token (see caveat) | 0.143, 0.000 | 0.278, 0.236 | 0.281, 0.249 |
| month token | 0.024, 0.015 | 0.240, 0.201 | 0.288, 0.268 |
| last token | 0.039, 0.035 | 0.226, 0.191 | 0.297, 0.263 |

**The real cloud has an order of magnitude *less* persistence than chance**, at every read
position and at every layer sampled (0 to 24: longest 1-cycle 0.034 to 0.128, never approaching
the nulls). There is no torus.

Two methodological notes, both mistakes made and then caught here:

- A **label permutation cannot test topology**. Shuffling the item names leaves the point cloud,
  and therefore the barcode, exactly as it was. Any null for a topological claim has to move the
  points.
- **Reading at the weekday token is not a valid joint measurement.** Attention is causal, so in
  "The date is Monday, March the 3rd" the month has not been seen when the weekday token is
  computed; additivity there is 1.000 by construction, and the barcode shows exactly one loop
  (0.143) with nothing second (0.000) - the weekday circle alone, as it must. Only the final token
  sees both factors.

So the honest statement is: two deformed loops, written additively into near-orthogonal subspaces,
with no toroidal topology. Additivity plus per-factor circularity is *necessary* for a torus but
not, at this sampling density and this degree of deformation, sufficient to produce one.

### Prior work worth knowing

- Numbers are represented as a generalised helix with periods 2, 5, 10, 100 plus a linear term,
  and the model uses it causally ("Clock" algorithm):
  [arXiv:2502.00873](https://arxiv.org/abs/2502.00873), code
  [subhashk01/LLM-addition](https://github.com/subhashk01/LLM-addition).
- Cyclic-concept arithmetic in Llama-3.1-8B runs through generic base-10 addition rather than
  concept-specific modular arithmetic: [arXiv:2605.01148](https://arxiv.org/abs/2605.01148).
- Neither studies **joint** cyclic variables or toroidal structure. That part appears open.
