# Phase 3 — Boundary metrics, preprocessing & architecture techniques

Motivation: round-1 diagnosis showed the models are **boundary-limited** — at region
Dice ~0.62 the measured **Boundary IoU is 0.32** and **NSD@2px is 0.19** (30 test
scenes, DeepLabV3+). The plume *contour* is drawn too wide, not the region missed.
This document reviews (1) evaluation metrics for fuzzy/gradient boundaries,
(2) preprocessing techniques, and (3) architecture techniques, with what we
implemented vs recommend. All sources were checked online.

---

## 1. Evaluation metrics for fuzzy / gradient boundaries

Region Dice/IoU treat every pixel equally, so a contour drawn a few pixels too wide
on a diffuse plume is penalized as if it were a gross error. Boundary-aware metrics
quantify contour quality directly:

- **Boundary IoU** — Cheng, Girshick, Dollár, Berg, Kirillov, *CVPR 2021*
  (arXiv:2103.16562). IoU computed only within a band around each contour; sensitive
  to boundary errors, scale-balanced. https://arxiv.org/abs/2103.16562
- **Normalized Surface Dice (NSD) at tolerance τ** — Nikolov et al., 2018/2021.
  Fraction of each surface lying within τ pixels of the other. Tolerant of
  sub-τ boundary jitter — the right metric when the true boundary itself is fuzzy
  (τ ≈ inter-annotator/estimation uncertainty). https://arxiv.org/abs/1809.04430
- **HD95 / ASSD** — 95th-percentile Hausdorff distance and average symmetric surface
  distance: standard distance-based boundary metrics; HD95 trims outliers.
  (See distance-metric pitfalls: https://arxiv.org/abs/2302.03868 and the
  segmentation-metrics package https://github.com/Jingnan-Jia/segmentation_metrics.)

**Implemented** (`src/metrics.py`, reported by `evaluate_full_scene` and the re-eval
table): `boundary_iou`, `nsd` (τ configurable via `eval.nsd_tolerance`), `hd95`,
`assd`. Measured baseline (DeepLabV3+, 30 test scenes): Dice 0.62, Boundary IoU 0.32,
NSD@2px 0.19, HD95 95 px, ASSD 30 px. **Recommendation:** report NSD at τ = 2 and a
looser τ (e.g. 5–10 px ≈ 2.5–5 km) — 2 px is strict for a 500 m-resolution diffuse
target; the looser τ better reflects the physically-ambiguous plume edge.

---

## 2. Data-preprocessing techniques

Current pipeline: per-channel z-score from train scenes (clip ±5), NaN→0 fill +
valid-pixel mask, positive-biased patch sampling (0.5), geometric+photometric aug,
year-balanced split, negatives at 0.3. Reviewed additions:

- **Soft / boundary label smoothing (SVLS)** — Islam & Glocker, *MICCAI 2021*
  (arXiv:2104.05788): Gaussian-blur the one-hot label so ambiguous boundaries become
  soft targets → better calibration and boundary prediction than hard labels.
  https://arxiv.org/abs/2104.05788
  **Implemented:** `target.soft_sigma` (train-only Gaussian blur of the mask;
  experiment `smp_unetpp_focal_tversky_soft`). Directly targets the fuzzy edge.
- **Gaussian-weighted sliding-window inference** — nnU-Net (Isensee et al., 2021):
  weight patch contributions by a centred Gaussian (1 at centre → ~0 at edge) to
  suppress tiling seams. **Implemented** in `sliding_window_predict`
  (`eval.gaussian_window`, default on). Effect here was marginal (patches large vs
  plumes) but it is the correct default.
- **Cleaner target** — the label is `isfinite(dispersal dBZ)` incl. sub-zero noise;
  the reference notebook preferred dBZ ≥ 0. **Implemented** as `target.mode=threshold`
  (experiment `smp_deeplabv3p_focal_dbz0`) — expected precision gain.
- **Copy-Paste augmentation** — Ghiasi et al., *CVPR 2021* (arXiv:2012.07177): paste
  rare positive regions into other scenes; strong gains on rare/imbalanced classes.
  **Recommended** (not yet implemented): paste plume crops from positive scenes onto
  negative/other scenes to enrich the ~5%-positive distribution.
  https://arxiv.org/abs/2012.07177
- **Learning with noisy labels** — the label is an algorithmically-cleaned, noisy
  product (Phase-1/feasibility finding). Survey: Med. Image Analysis 2024
  (https://www.sciencedirect.com/science/article/abs/pii/S1361841524000914);
  Confident Learning for segmentation, MICCAI 2020
  (https://doi.org/10.1007/978-3-030-59710-8_70); Mean-Teacher-assisted Confident
  Learning (https://pubmed.ncbi.nlm.nih.gov/35604969/). **Recommended** for a later
  round — these set the realistic performance ceiling more than architecture does.

---

## 3. Architecture / loss techniques

Current: U-Net, Attention U-Net, nnU-Net(2D), U-Net++, DeepLabV3+, SegFormer, behind a
common interface; Dice/BCE/Focal/Tversky/Focal-Tversky losses; TTA; ensemble.

- **Boundary loss** — Kervadec et al., *MIDL 2019* (arXiv:1812.07032; extended MedIA
  2021): integral of the prediction over the signed distance transform of the GT — a
  contour-space distance that is robust to class imbalance and sharpens boundaries
  (reported up to +8% Dice, +10% Hausdorff vs Dice alone).
  https://arxiv.org/abs/1812.07032
  **Implemented:** `loss.name=dice_boundary` (Dice+BCE + `boundary_weight`·boundary;
  experiment `smp_deeplabv3p_dice_boundary`). Highest-leverage boundary lever we can add.
- **Gated-SCNN / shape stream** — Takikawa, Acuna, Jampani, Fidler, *ICCV 2019*
  (arXiv:1907.05740): a parallel shape branch supervised on boundaries, fused with the
  main stream; sharper contours, big gains on thin/small structures.
  https://arxiv.org/abs/1907.05740 **Recommended** (heavier; a second boundary head is
  the lighter first step, i.e. add an auxiliary boundary-prediction output).
- **HRNet (+OCR)** — Sun/Wang et al., 2019/2020 (arXiv:1908.07919): maintains
  high-resolution representations throughout, strong for precise localization; usable
  as an SMP/timm encoder backbone. https://arxiv.org/abs/1908.07919 **Recommended** to
  add as an encoder option for the SMP models.
- **Deep supervision** — auxiliary losses at decoder scales (standard in nnU-Net,
  U-Net++); stabilizes training and can sharpen output. **Recommended** for `nnunet`.
- **Ensembling** — averaging the 6 members already gives **+~0.04 macro Dice**
  (validated); **implemented** (`python -m src.evaluate --ensemble`).

---

## Recommended order of experiments (expected impact, all still bounded by §feasibility)

1. `dice_boundary` + `soft_sigma` (boundary sharpening) — best boundary-metric gain.
2. `dbz0` cleaner target (precision) and richer channels (input information).
3. Ensemble of the best members (+~0.04 macro, no training).
4. Later: copy-paste augmentation; noisy-label refinement (Confident Learning /
   Mean-Teacher) — these, plus richer radar moments, are what would move the ceiling.

Honest expectation (see progress.md feasibility): these lift macro Dice toward
~0.60–0.66 and materially improve the boundary metrics (Boundary IoU / NSD), but do
not reach 90% — that needs a cleaner label and/or richer inputs (dual-pol, velocity,
full elevation volume, temporal).

## References (verified)
- Cheng et al. (2021) Boundary IoU. CVPR. arXiv:2103.16562. https://arxiv.org/abs/2103.16562
- Nikolov et al. (2018/2021) Deep learning to achieve clinically applicable segmentation (Surface Dice / NSD). arXiv:1809.04430. https://arxiv.org/abs/1809.04430
- Kervadec et al. (2019) Boundary loss for highly unbalanced segmentation. MIDL. arXiv:1812.07032. https://arxiv.org/abs/1812.07032
- Islam & Glocker (2021) Spatially Varying Label Smoothing. MICCAI. arXiv:2104.05788. https://arxiv.org/abs/2104.05788
- Takikawa et al. (2019) Gated-SCNN. ICCV. arXiv:1907.05740. https://arxiv.org/abs/1907.05740
- Sun/Wang et al. (2019/2020) HRNet. arXiv:1908.07919. https://arxiv.org/abs/1908.07919
- Ghiasi et al. (2021) Simple Copy-Paste. CVPR. arXiv:2012.07177. https://arxiv.org/abs/2012.07177
- Isensee et al. (2021) nnU-Net (Gaussian-weighted inference). Nature Methods. https://www.nature.com/articles/s41592-020-01008-z
- Label-noise survey (2024), Medical Image Analysis. https://www.sciencedirect.com/science/article/abs/pii/S1361841524000914
