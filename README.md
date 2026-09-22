# Are cyclic concepts represented on circles?

Following Engels et al., *Not All Language Model Features Are One-Dimensionally Linear*, ICLR 2025
([arXiv:2405.14860](https://arxiv.org/abs/2405.14860), code
[JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures)).

Their claim: some concepts are not represented as single directions but as genuinely
multi-dimensional objects, and days of the week and months of the year sit on **circles** that the
model uses to do modular arithmetic.

This repository builds the measurement from scratch rather than running theirs, so that every
control is explicit, and extends it to Gemma-2-2B-IT, which the original work did not test.
**Model note:** every Gemma result here uses the instruction-tuned checkpoint
`google/gemma-2-2b-it` (prompts are plain text, no chat template), not the base model. Earlier
versions of this README called it Gemma-2-2B; the key is now `gemma-2-2b-it` everywhere.
Free and local: GPT-2 is a 0.5 GB ungated download, Gemma-2-2B-IT is already cached from an earlier
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
| `06_deformation.py` | Harmonics, PC2/PC1 spread, angular gaps: circle or deformed loop? | ~3 min |
| `07_torus_validation.py` | Denser sampling, three read positions, cloud-perturbing nulls, depth sweep | ~25 min |
| `08_rotate_month.py` | Rotate the month subspace inside a full date prompt | ~30 min |
| `09_intervention_diagnostic.py` | Did the rotation actually perturb the activation? | ~2 min |
| `10_day_of_month.py` | Day-of-month periods with tokenisation recorded; clean on GPT-2 | ~2 min |

```powershell
python scripts/01_find_circles.py                                  # gpt2, layer 7
python scripts/01_find_circles.py --model gemma-2-2b-it --dtype bfloat16
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

Gemma-2-2B-IT, `blocks.16.hook_resid_post`: weekdays 0.19 / 1.00, months 0.20 / 1.00, controls at or
below chance. **The circles replicate in a model the original paper did not test.**

The number-words row is the one that makes the result meaningful. It is perfectly *ordered* but
not at all *circular*, which is exactly what an ordinal, non-cyclic concept should look like. The
measures are therefore distinguishing ring from line, not just structure from noise.

**A stronger shape null.** Isotropic Gaussian points are a weak comparison, because real
activations have a few dominant directions. `01_find_circles.py` now also draws points from a
Gaussian with each item set's *own* covariance. Weekdays and months still sit below that null's
5th percentile of radial CV in both models (GPT-2: 0.15 vs 0.26 and 0.14 vs 0.31; Gemma: 0.19 vs
0.24 and 0.20 vs 0.32); number words and the controls do not.

**How much of the circle is in the token embeddings depends on the concept.** Reading the residual
stream *before* any attention or MLP has run (`blocks.0.hook_resid_pre`), radial CV / order:

| | embedding | early layers | resid_post, layer 16 (Gemma) / 7 (GPT-2) |
|---|---|---|---|
| GPT-2 weekdays | 0.23 / 1.00 | 0.14 by layer 2 | 0.15 |
| GPT-2 months | 0.31 / 1.00 | 0.18 by layer 4 | 0.14 |
| Gemma weekdays | 0.23 / 1.00 | 0.19 by layer 5 | 0.19 |
| Gemma months | 0.33 / 1.00 | **0.10** by layer 5 | 0.20 |

The **weekday** ring is largely inherited from the embedding in both models: already ordered and
nearly as round before any layer runs. The **month** ring starts looser and the first few layers
tighten it substantially, most clearly in Gemma (0.33 -> 0.10). An earlier version of this README
stated the embedding result without saying it came from GPT-2 only; the Gemma sweep
(`02_layer_sweep.py --model gemma-2-2b-it --hook resid_pre`) has now been run.

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
that plane untouched, and ask whether the model's answer advances by k days. Three tasks ("Today is
X. Tomorrow is" / "Yesterday was" / "The day after tomorrow is"); baseline accuracy 1.00, 0.57 and
1.00. Each shift is scored against the model's **own unrotated answer** for that prompt, so the
0.57 task's baseline mistakes cannot leak in. 21 trials per shift, 126 for all k != 0; 95% Wilson
intervals.

| Condition | answer moves by exactly k (k != 0) | answer unchanged | mean displacement |
|---|---|---|---|
| rotate in the fitted circle | **0.230** [0.165-0.311] | 0.35 | 61.5 |
| rotate in PC3-PC4 (variance-matched control) | 0.056 [0.027-0.110] | 0.53 | 38.8 |
| rotate in PC5-PC6 (variance-matched control) | 0.032 [0.012-0.079] | 0.69 | 28.2 |
| rotate in a random plane (5 planes) | 0.000-0.008 | 0.96-1.00 | 1.4-2.0 |

The circle plane moves the answer by the intended amount four to seven times as often as a
control plane of comparable size, and the intervals do not overlap. A random plane barely perturbs
the activation at all, so the answer simply does not change; that is why it is not the control
that counts. (The first version scored against the correct answer, which gave 0.222 for the circle
and let random planes land on "target" by chance about 0.14 of the time via baseline errors.)

Per-shift rates rest on 21 trials each and are noisy (for the circle they range from 0.05 at k = -2
to 0.38 at k = +3), so read the pooled number, not the curve's shape.

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
subspaces: the model writes weekday and month into the residual stream as independent,
non-interfering summands. Two qualifications from step 7: at the final token the month carries 96%
of that variance and the weekday only 3%, and the claim it does *not* support is the topological
one.

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

**Tokenisation check.** Gemma splits every number into single digits (" 31" is " ", "3", "1"), so
the Gemma vectors above are read at the units digit, and 1-9 are shorter than 10-31. That could
have manufactured period structure. GPT-2 keeps " 1" ... " 31" as single tokens, so
`10_day_of_month.py` repeats the analysis there (`blocks.7.hook_resid_post`):

| Component | GPT-2 unique variance | null 95th pct | verdict |
|---|---|---|---|
| period 3 | 0.187 | 0.175 | real |
| period 31 | 0.167 | 0.093 | real |
| linear | 0.102 | 0.054 | real |
| period 2 | 0.049 | 0.049 | at the threshold |
| periods 5, 10, 12, 7 | 0.053 to 0.089 | ~0.10 | fit noise |

The same three components come out on top in a model with no digit splitting, so the helix and
the odd period-3 component are **not** tokenisation artefacts. One caveat on reading "period 31" as
a circle: with 31 items it is a single cycle across the whole range, which a smooth curved (non-
circular) trajectory can also fit. Whether day 31 actually sits next to day 1 has not been tested.

## Step 6: is it a circle, or a deformed loop?

Three independent measures, all on the same item vectors, at `blocks.16.hook_resid_post`:

| | weekdays | months | perfect circle |
|---|---|---|---|
| power in harmonic 1 | 0.57 (null 0.45) | 0.38 (null 0.23) | 1.00 |
| variance in the best plane | 0.60 | 0.39 | 1.00 |
| PC2/PC1 spread (not an ellipse fit) | 0.81 | 0.94 | 1.00 |
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

**But the joint barcode alone could not have detected a torus here.** A positive control fixes
this: an ideal torus (two perfect, evenly spaced, orthogonal circles) built with the *same*
weekday/month variance split and pushed through the identical pipeline. At the final token the
split is lopsided - weekday 3.3% of the variance, month 96.2% - and for a torus that lopsided the
small weekday loop scores 0.175, *below* the Gaussian null (0.191). The original "both loops must
beat the null" rule would have called a perfect torus "not clear". That rule is therefore reported
but no longer used.

Two tests that do have power:

| read position | ideal torus, longest loop | real, longest loop | weekday main effect a loop? | month main effect a loop? |
|---|---|---|---|---|
| month token | 1.19 | 0.024 | 0.000 (null 95th 0.130) | 0.007 (null 95th 0.199, p = 0.89) |
| final token | 1.18 | 0.039 | 0.000 (null 95th 0.115) | 0.021 (null 95th 0.213, p = 0.78) |

- **Dominant loop.** An ideal torus with this split shows the month loop at ~1.2. The real cloud's
  longest loop is 0.02-0.04. The dominant factor does not form a loop in the joint cloud at all.
- **Per-factor loops.** Each factor's main effect (7 or 12 points), tested on its own against a
  covariance-matched null of the same size: neither is a loop in the full space, at either read
  position.

So the conclusion stands, for a stronger reason than first given: **no torus, because neither
factor's joint-prompt representation is itself a persistent loop.** The single-factor circles are
ordered rings *in their top two principal components* (step 1), but with only 38-57% of their
variance at harmonic 1 (step 6), the rest spreads over other directions and the full-dimensional
point set does not close into a loop that a Rips filtration can see. The step-1 weekday set at the
weekday token is borderline on the same test (0.143 against a null 95th of 0.141, p = 0.06).

"Additive product" also needs qualifying: at the final token the structure is **mostly month plus a
small weekday offset** (3.3% of variance), in near-orthogonal directions, at every layer.

Two methodological notes, both mistakes made and then caught here:

- A **label permutation cannot test topology**. Shuffling the item names leaves the point cloud,
  and therefore the barcode, exactly as it was. Any null for a topological claim has to move the
  points. And a null is only half a test: without a positive control there is no way to know the
  test could have said yes.
- **Reading at the weekday token is not a valid joint measurement.** Attention is causal, so in
  "The date is Monday, March the 3rd" the month has not been seen when the weekday token is
  computed; additivity there is 1.000 by construction. Only the final token sees both factors.

## Step 8: the month circle does not steer month arithmetic

Step 3 rotated a single-factor circle in a single-factor prompt and the answer moved. This asks the
harder version: in a prompt naming both a weekday and a month, does rotating the *month* subspace
at the month token move the model's month answer by the right number of months?

Three tasks, 3 weekdays x 12 months each; baseline accuracy 1.00, 0.69 and 1.00. Shifts are scored
against the model's own unrotated answer, per task. 648 trials for all k != 0; 95% Wilson intervals.

| | answer moves by exactly k (k != 0) |
|---|---|
| rotate in the fitted month circle | 0.023 [0.014-0.038] |
| rotate in PC3-PC4 (variance-matched control) | 0.000 [0.000-0.006] |
| rotate in PC5-PC6 (variance-matched control) | 0.002 [0.000-0.009] |
| chance | 0.083 |

**The month answer does not rotate.** What does change is how often the model falls back to the
input month, and broken down by task that effect lives entirely in one place:

| answer = input month, at rotation -3 ... +3 | month circle | PC3-PC4 control | PC5-PC6 control |
|---|---|---|---|
| "It was D, M. The next month is" (baseline 1.00) | 0.00 everywhere | 0.00 everywhere | 0.00 everywhere |
| "On D in M. One month later it was" (baseline 0.69) | 0.81 0.64 0.50 **0.31** 0.14 0.14 0.00 | 0.08 0.08 0.14 **0.31** 0.39 0.50 0.47 | 0.25 0.28 0.31 **0.31** 0.36 0.36 0.36 |
| "It was D, M. The previous month was" (baseline 1.00) | 0.00, except 0.11 at +3 | 0.00 everywhere | 0.00 everywhere |

- On the two tasks the model does reliably, rotating the month circle does **nothing**: no shift,
  no fallback.
- On the one task where the model is already unsure (it echoes the input month 31% of the time
  unperturbed), rotation pushes it monotonically towards or away from echoing. But the control
  planes do the same, just less (swing 0.81 for the circle, 0.39 and 0.11 for the controls, in
  line with how far each rotation moves the activation: 64, 50 and 37 at k = 3). The circle's
  small on-target excess (0.023) is this same effect: at k = -1, falling back to the input month
  counts as "shifted by -1".
- Rotating the month subspace never disturbed the weekday answer in the same prompt (0.00 in every
  condition), consistent with the near-orthogonality measured in step 7.

An earlier version of this section pooled the three tasks and read the fallback curve as "the
month circle gates the +1 operation". Per task, that reading does not hold: the effect is confined
to a fragile prompt, and it is only partly specific to the circle plane.

### Ruling out "the intervention never landed"

The circle is fitted on single-factor prompts and applied in date prompts, so it could have missed.
`09_intervention_diagnostic.py` measures where the activations actually sit and how far a rotation
moves them:

| | in-plane radius, fitting prompts | in-plane radius, use prompts | displacement at k=1 | at k=3 |
|---|---|---|---|---|
| months, circle | 44.8 | 45.4 | 23.5 (9.8% of \|x\|) | 64.2 |
| weekdays, circle (step 3, which worked) | 49.2 | 61.2 | 53.1 (17.6% of \|x\|) | 119.3 |

The fitted centre transfers cleanly to the date prompts (radius 44.8 -> 45.4), so the intervention
is landing on the right part of the space. One month step is a smaller perturbation than one
weekday step, because 1/12 of a circle is a smaller angle than 1/7 - but **a 3-month rotation
displaces the activation by 64, more than the 53 that moved the weekday answer, and still produces
zero on-target shifts.** Perturbation size does not explain the null.

So the honest reading is that the weekday result of step 3 does not generalise to months in a
two-factor prompt. The month circle is present and ordered, but rotating it neither moves the
answer nor specifically controls whether the model applies the offset. That distinction is easy to
lose when the only evidence is a picture of a ring.

### Prior work worth knowing

- Numbers are represented as a generalised helix with periods 2, 5, 10, 100 plus a linear term,
  and the model uses it causally ("Clock" algorithm):
  [arXiv:2502.00873](https://arxiv.org/abs/2502.00873), code
  [subhashk01/LLM-addition](https://github.com/subhashk01/LLM-addition).
- Cyclic-concept arithmetic in Llama-3.1-8B runs through generic base-10 addition rather than
  concept-specific modular arithmetic: [arXiv:2605.01148](https://arxiv.org/abs/2605.01148).
- Neither studies **joint** cyclic variables or toroidal structure. That part appears open.
