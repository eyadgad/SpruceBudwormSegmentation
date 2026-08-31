# Leakage-free night split and controlled ablation

Status: implementation complete and verified; training in progress. Results
sections are marked **PENDING** and will be filled from the runs, not predicted.

---

## 1. Why

Every experiment in `outputs/experiments/` was ranked on a **scan-level**
stratified split (`src/data_prep.py:stratified_split_labels`, stratified by year
only). The radar scans the same moth exodus roughly every 30 minutes, so
consecutive scans of one plume are strongly correlated. Measured on the current
manifest:

> **172 of 248 operational nights (69%) span more than one split.**

`docs/evaluation_dashboard.md` §7 already lists *"True generalisation estimate →
a night-disjoint split and a retrain"* as the outstanding gap, and
`progress.md:114-125` records the leakage as a knowingly accepted trade-off
("acceptable here because the goal is a CONSISTENT split to RANK architectures").
This work closes it and then tests three improvements under a controlled
ablation.

The model is held fixed throughout: `sweep_attunet_dbz0_e012345678_focaltv` —
Attention U-Net (base_filters 32), channels `th_e0..th_e8` + `valid_mask`,
dbZ ≥ 0 target, Focal-Tversky (α=0.3, β=0.7, γ=1.333), AdamW 3e-4 / wd 1e-5,
256px patches, threshold calibrated on validation positives, full-scene
sliding-window inference, TTA off.

---

## 2. Measured properties of the corpus

All derived from `artifacts/manifest.csv` (2052 scans, 2013-2019) before any
change was made. These numbers drove the design.

| Property | Value | Why it matters |
|---|---|---|
| Operational nights (UTC noon-noon) | 248 | the split unit |
| Migration / quiet nights | 117 / 131 | stratification variable |
| **Nights spanning >=2 splits (old split)** | **172 / 248 (69%)** | the leakage being removed |
| Nights mixing positive + negative scans | **0** | night truth is homogeneous |
| Scans on migration / quiet nights | 1579 / 473 | all positives lie on migration nights |
| Scans per night, migration / quiet | mean 13.5 / 3.6 | 70/20/10 must target *scans*, not nights |
| Scan cadence | **median 30 min**, 78% of gaps <= 40 min | sets the temporal window tolerance |
| Scans with both +/-1 in-night neighbours | 54% | |
| Scans with exactly one / none | 29% / **17%** | missing neighbours are the common case |
| `night` column populated for negatives | **no - empty for all 473** | night IDs must come from timestamps |

Two consequences worth stating plainly:

* Because no night mixes truth classes, **scan presence and night identity are
  perfectly confounded** on this corpus. A model can score well on presence by
  recognising "this looks like a quiet night" rather than "there is no plume
  here". This limits what the classification stage can be said to demonstrate.
* Because 17% of scans have no in-night neighbour, any temporal design must
  handle absent frames as a first-class case, not an edge case.

---

## 3. Literature review

### 3.1 Leakage-free splitting for temporally correlated data

The consensus is unambiguous. **Kattenborn et al. (2022)**, *Spatially
autocorrelated training and validation samples inflate performance assessment of
convolutional neural networks* (ISPRS Open Journal of Photogrammetry and Remote
Sensing) found that **more than 90% of reviewed remote-sensing CNN studies did
not ensure independence between training and validation data**, and that the
resulting inflation is systematic. **Ploton et al. (2020)** (Nature
Communications) show the same failure in ecological modelling, and the
leave-profile-out work in *Geoderma* (2025) shows it in 3D soil mapping.

The standard remedy is **blocked / grouped splitting**: choose the natural
correlation unit and assign whole groups, never individual samples. scikit-learn
encodes this as `GroupKFold`, with `StratifiedGroupKFold` additionally preserving
class ratios across folds while keeping each group intact.

**Applied here.** The correlation unit is the operational night — one exodus
event sampled every 30 minutes. `src/presence.py:operational_night_id` already
implements a tested UTC noon-to-noon night ID (with month/year/leap-day
rollovers) and, critically, derives it from the *timestamp*, so it works for the
473 negatives whose `night` column is empty. The design is stratified-group
assignment: group = night, strata = year x (migration | quiet).

### 3.2 Joint segmentation + scan-level classification

Multi-task hard parameter sharing — one encoder, task-specific heads — is well
established in medical imaging: **Wu et al. (2021)** (*Medical Image Analysis*)
and **Amyar et al. (2020)** (*Computers in Biology and Medicine*) both report
joint classification + segmentation on CT improving both tasks, and several works
add a presence classifier specifically to **suppress false positives on
lesion-free inputs**.

The design question for a *patch-based* trainer is label assignment. The
**Multiple Instance Learning** literature (**Ilse et al. 2018**, ICML;
**Campanella et al. 2019**, Nature Medicine; **DSMIL**, Li et al. CVPR 2021; and
the max-pooling MIL analysis in arXiv:2408.09449) frames an image as a **bag** of
patches whose label is the max over instance labels. Propagating the bag label
down to every instance is precisely the *instance-based MIL with noisy labels*
baseline the literature identifies as weak — a plume-free crop of a positive scan
would be told it is positive.

**Applied here.** This repository does not need noisy propagation at all:
`RadarPatchDataset.__getitem__` already holds the patch's own ground-truth crop,
so the exact instance label is `yp.sum() > 0` — free and noise-free, and derived
from the full ground truth. Scan presence is recovered at inference by
**max-pooling the window scores** (the standard bag aggregation).

### 3.3 Spatiotemporal modelling of successive scans

The decisive reference is **Garnot & Landrieu, *Panoptic Segmentation of
Satellite Image Time Series with Convolutional Temporal Attention Networks*,
ICCV 2021** (U-TAE, arXiv:2107.07933) — peer-reviewed, remote sensing, and
*segmentation of an image time series* rather than nowcasting. A 2D spatial
encoder runs on every frame with shared weights, a **Lightweight Temporal
Attention Encoder (L-TAE) is applied only at the lowest-resolution bottleneck**,
and its attention masks are bilinearly upsampled to temporally collapse the skip
connections. Reported: **63.1 mIoU on PASTIS, ~+5 mIoU over UNet-3d,
UNet+ConvLSTM and FPN+ConvLSTM**. The companion **L-TAE paper** (Garnot &
Landrieu, ICPR-W 2020, arXiv:2007.00586) reports L-TAE matching temporal
attention encoders with ~10x the parameters and recurrent units with **>300x**
the parameters.

Counter-evidence was weighed rather than ignored. **Vu et al. (2020)**,
*Evaluation of multi-slice inputs to convolutional neural networks for medical
image segmentation* (*Medical Physics*, arXiv:1912.09287), tested adjacent-slice
channel stacking across five datasets and two backbones and found **a significant
improvement over plain 2D in only one of five**, with no relation between the
number of input slices and performance. **Zhang et al. (2022)** (*Computerized
Medical Imaging and Graphics*) reach the same "2.5D is cheap but not reliably
better" conclusion. At the other extreme, 3D CNNs carry a documented parameter
explosion and overfitting risk on small datasets.

**Why U-TAE-style wins for this specific dataset:**

1. **It preserves the required architecture.** Encoder, attention gates and
   decoder are reused verbatim; only a bottleneck module and a skip-collapse step
   are added. A ConvLSTM decoder or 3D U-Net would replace the Attention U-Net
   the task requires preserving. Measured cost here: **+35,457 parameters**
   (7,892,190 vs 7,856,733), i.e. +0.45%.
2. **It handles missing neighbours natively.** 17% of scans have none and 29%
   have one; U-TAE's `pad_mask` removes padded slots from the attention softmax.
   Channel stacking has no such mechanism and must impute — injecting an artefact
   into exactly the population (isolated scans) most likely to be quiet nights.
3. **Variable-length sequences need no architectural change**, and length here is
   data-dependent (1-3 frames), not fixed.
4. **30-minute cadence is coarse** relative to nowcasting (5-10 min). Attention
   *selects* informative neighbours instead of assuming smooth motion — the safer
   inductive bias when a neighbour may be 30 or 480 minutes away.

**Decision:** L-TAE temporal attention at the bottleneck with attention-collapsed
skips (S3b), with naive channel stacking retained as an explicit **control**
(S3a). Vu et al. predict stacking should not reliably help; if it matches S3b,
the added complexity is not earned and this document will say so.

---

## 4. The night split

`src/nights.py` is the single source of truth.

* `night_id()` delegates to `presence.operational_night_id`, so the training
  split and the dashboard's night analysis can never disagree.
* `assign_night_split()` groups scans by night, labels each `migration` (any
  positive scan) or `quiet`, forms year x kind strata, and within each stratum
  assigns whole nights **largest-first to the split furthest below its target
  scan count** — the greedy rule `build_year_split` already used for whole years,
  lifted to nights and driven by scans.
* `night_split_report()` / `verify_no_leakage()` report and enforce zero overlap.

**Deficits are per-stratum, not global.** A single global running total sends
every early year to train (whose 70% target is satisfied last) and leaves val and
test drawn only from the final years — the first implementation did exactly that,
producing a val split covering only 2017-2019 and a test split covering only
2018-2019. Per-stratum targets split each year x kind ~70/20/10 independently.

### Resulting split (`artifacts_night/split_summary.json`)

| Split | Nights | Migration / quiet | Scans | Fraction | Positives | Negatives | Years |
|---|---|---|---|---|---|---|---|
| train | 148 | 69 / 79 | 1439 | 0.701 | 1108 | 331 | all 7 |
| val | 58 | 27 / 31 | 403 | 0.196 | 310 | 93 | all 7 |
| test | 42 | 21 / 21 | 210 | 0.102 | 161 | 49 | all 7 |

**`overlap_nights: 0`.** Verified three ways: by `verify_no_leakage` during data
prep, by `scripts/test_night_split.py`, and independently by
`presence.analyze_presence`, whose cohort block reports
`night_overlap.validation_test == 0` and
`training_exposure.test.nights_seen_in_train == 0`. For comparison, the old split
produced 110 of 113 test nights seen in training and 93 shared val/test nights.

Artifacts live in **`artifacts_night/`** with its own `norm_stats.json`
recomputed from the new train split — reusing the old statistics would leak
normalization. `artifacts/` is untouched, so the published dashboard model
remains reproducible. The split-independent target cache is shared via
`data.targets_dir`.

---

## 5. Implementation

| Component | Where | Notes |
|---|---|---|
| Night ID + split | `src/nights.py` | delegates to `presence.operational_night_id` |
| `split.mode: night` | `src/data_prep.py` | `night` derived for all rows; the mask-file variable name moves to `mask_night` (`cache_targets` indexes the netCDF by it) |
| Global seeding | `src/experiment.py:seed_everything` | nothing in `src/` seeded torch/numpy before this |
| Deferred test eval | `src/experiment.py`, `src/finalize.py` | `eval.defer_test` makes training report `val_full_scene` only |
| Classification head | `src/models/attention_unet.py` | GAP -> Linear on the bottleneck; `forward` returns `(seg, cls)` |
| Multi-task loss | `src/losses.py:MultiTaskLoss` | `L_seg + lambda*BCE`; lambda=0 reproduces pure segmentation exactly |
| L-TAE / aggregator | `src/models/temporal.py` | learned master query per head; `att_group` skip collapse |
| Temporal sequences | `src/dataset.py` | same night **and** same split; <=35 min gap |
| Night-balanced sampler | `src/dataset.py:NightBalancedSceneSampler` | train only |
| Variant comparison | `scripts/export_variant_comparison.py` | reuses `analyze_presence` unchanged |

### Two design details worth calling out

**Padding content vs padding mask.** A missing temporal slot is filled with a
**copy of the centre frame**, not zeros, *and* flagged in `pad_mask`. The copy
keeps the shared encoder's BatchNorm from ingesting all-zero images (~20% of
frames would otherwise be blank, shifting the running statistics); the mask
removes the slot from the attention softmax so it contributes nothing. A scan
with no neighbours therefore degrades **exactly** to the single-frame model —
asserted in `test_isolated_scan_degrades_to_the_single_frame_model`.

**Backward compatibility.** Plain runs still yield `(x, y)` and a bare logits
tensor; extras appear only when enabled. Verified: the old config still builds a
7,856,733-parameter model with a 2-tuple dataset item and a non-multitask loss.

---

## 6. Experiment protocol

`configs/base_config_night.yaml` + `configs/experiments_night.yaml`. Cumulative,
one component per stage; optimizer, seed (42), patch geometry, threshold grid and
calibration procedure held fixed. 100 epochs, **early-stopping patience 50**
(the old base config had `patience == epochs`, making early stopping unreachable).

| Stage | Name | Adds |
|---|---|---|
| S0 | `night_base_attunet9` | night split only — **the baseline** |
| S1 | `night_bal_attunet9` | night-balanced sampling |
| S2 | `night_mtl_attunet9` | + joint presence classification |
| S3a | `night_tstack_attunet9` | + temporal channel stacking (control) |
| S3b | `night_ltae_attunet9` | + L-TAE temporal attention (primary) |

lambda is selected by validation-only probes
(`configs/experiments_night_lambda.yaml`, lambda in {0.1, 0.3, 1.0}, 25 epochs)
ranked on validation patch Dice. **PENDING.**

The test set is untouched until every decision is frozen; `src/finalize.py` then
scores it once at the threshold already calibrated on validation. Confirmed in
practice: finished probe results contain `val_full_scene` and no
`test_full_scene` key at all.

### Compute characteristics (measured, not assumed)

Training is **I/O-bound, not GPU-bound**. Benchmarking the data loader alone
(no forward or backward pass) gives ~290 ms per batch of 8, i.e. **~420 s/epoch
of pure scene loading** — matching the historical run's 418 s/epoch median almost
exactly. `nvidia-smi` shows the GPU averaging ~20% during training.

* Neither the night-balanced sampler nor `scene_cache` is a cost driver. The
  balanced sampler is marginally *faster* (~350 s vs ~416 s train-only) because
  resampled scenes hit the LRU cache.
* `num_workers` does not help: 0/2/4/6 workers measured at 437/411/436/537
  ms/batch. The limit is disk bandwidth, so extra workers only contend for it.
  Left at 0.
* The first epoch is much slower than the rest (cold OS file cache), and
  concurrent disk activity inflates epoch times substantially. Judge throughput
  on a quiet machine from epoch 2 onward.
* The T=3 temporal stages load three scenes per sample and cost roughly 3x per
  epoch. That is inherent to the design, not overhead.

The machine currently runs at ~900-1000 s/epoch, about 2.4x slower than when the
historical model was trained. This is environmental, not a regression:
re-benchmarking the **historical** config today gives 429 s of data loading
alone, more than that run's entire 417 s epoch. Operational guidance is in
`docs/RUNBOOK_night_split.md`.

---

## 7. Verification

`scripts/test_night_split.py` — 41 tests, CPU-only, no data required.

Night derivation (noon boundary, year/leap rollovers, negatives get IDs,
agreement with `presence.operational_night_id`, `mask_night` preserved) · split
integrity (zero overlap, whole nights, a deliberately broken split *is* detected,
scan fractions, every year and both night kinds in every split, both truth
classes in val, reproducible and seed-sensitive, fractions must sum to 1) ·
leakage via `analyze_presence` (all overlap counts and training exposure zero, no
partial nights) · patch labels (**a plume-free crop of a positive scan is
labelled negative** — the regression this design exists to prevent) · multi-task
(shapes, lambda=0 equivalence, gradients reach both heads, plain model unchanged)
· temporal (never crosses night or split, centre never padded, chronological, gap
tolerance, night boundaries produce padding, `pad_mask` correctness, padding is a
centre copy not zeros, isolated scan == single-frame, stack mode has no mask) ·
night-balanced sampling (epoch length preserved, contiguity preserved, classes
balanced, differs from the natural distribution, per-night spread reduced,
deterministic per seed).

Existing `scripts/test_presence.py` still passes unchanged. Its old-split
constants (615 scans, 110/113 nights, `validation_test == 93`) describe the
**historical** artifacts and were deliberately left alone.

---

## 8. Results

**PENDING** — to be filled from the completed runs. No improvement will be
claimed that the controlled comparison against S0 does not demonstrate.

---

## 9. Limitations

* **The historical dashboard model is not comparable.** It was trained on the
  scan-level split; it appears only as context, never as a baseline.
* **Absolute scores should fall** relative to the historical 0.6347 test Dice.
  Removing leakage removes inflation; a drop is the expected, correct outcome.
* **Night-level statistics are noisy.** Test holds 42 nights (21 migration). ROC
  AUC and Mann-Whitney on that many points have wide intervals.
* **Presence is confounded with night identity** — every negative scan lies on an
  all-negative night, so the classifier may be learning "quiet night" rather than
  "no plume here".
* **30-minute cadence is coarse** for insect flight; the temporal stages may show
  no gain. Both temporal variants are reported either way.
* **Negative masks are synthesized** as all-zero arrays by dataset construction,
  not independently annotated.
* **One seed per stage.** Differences smaller than seed-to-seed variance are not
  evidence; no seed-variance estimate is available within this compute budget.
