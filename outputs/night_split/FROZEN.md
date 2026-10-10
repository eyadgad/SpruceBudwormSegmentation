# Frozen publication protocol

Written before `src.finalize` opens the test set. Thresholds below were calibrated on validation during training and are not refit on test.

## Split

- Manifest SHA-256: `170c442d1dd7b878cdb4157139f7b4dc178f6d0e078534b6dc291d3c7091222f`
- Artifacts: `artifacts_night/` (`manifest.csv`, `norm_stats.json`, `split_summary.json`, `manifest.sha256`)
- Unit: operational night, UTC noon-to-noon. Fractions 0.70 / 0.20 / 0.10, `split.seed` 42.
- Negatives: `balanced: true`, `ratio: 1.0`.
- Counts: train 1108 pos / 1050 neg, val 310 / 302, test 161 / 147. 3078 scans, 267 nights, 0 nights shared across splits.

## Models

- Final spatial model: Attention U-Net, `base_filters` 32, channels `th_e0`–`th_e8` + `valid_mask`, Focal-Tversky α=0.3 β=0.7 γ=1.333, dBZ ≥ 0 target.
  Seeds: `night_base_attunet9_s42`, `s43`, `s44`, `s45`, `s46` (100 epochs, patience 50).
- Spatial baseline: U-Net, same recipe.
  Seeds: `unet_night_s42`, `s43`, `s44`, `s45`, `s46`.
- Paired confirmatory comparison: `night_base_attunet9_s42` versus `unet_night_s42`.
- Presence model: Swin-Tiny, image size 384.
  Seeds: `cls_swin_tiny_bal` (42), `cls_swin_tiny_bal_s43`, `s44`, `s45`, `s46`.
  `p_cls` attachment uses `cls_swin_tiny_bal`.

### Extra runs included in the paper (single seed 42 unless noted)

Useful as controlled negatives / architecture checks on the same frozen split. Not a second 5-seed family except where listed. Thresholds are validation-calibrated and not refit on test.

- Cumulative ablation (vs S0 `night_base_attunet9_s42`): `night_bal_attunet9_cur` (S1, t=0.15), `night_mtl_attunet9_cur` (S2, t=0.15), `night_tstack_attunet9_cur` (S3a, t=0.20), `night_ltae_attunet9_cur` (S3b, t=0.15).
- One-model cascade: `gated_attn_unet_cur` (t=0.15, hard gate on its own presence head at the validation-calibrated `calibrated_cls_threshold` = 1e-6).
  Its first test pass (2026-10-09 17:52) gated at 0.5 because `src.finalize` did not pass the locked gate threshold. That pass is kept in `outputs/night_cascade/experiments/superseded_gate05/` and is not reported. It was rescored once at the locked 1e-6 after the fix; no setting was chosen from test.
- Other decoders (n=1): `transunet_night` (t=0.15), `gated_swin_attn_night` (t=0.20), `swin_unet_night` (t=0.15). Swin-UNet is exploratory n=1 (best patch epoch 41); it is not a 5-seed family.

## Locked decisions

- Segmentation threshold: 0.15 for every main-paper AttnUNet and U-Net seed (validation-calibrated). Extra-run thresholds as listed above (S3a and gated Swin-Attn use 0.20).
- Classifier thresholds (validation Youden J): seed 42 = 0.05, 43 = 0.10, 44 = 0.05, 45 = 0.55, 46 = 0.15.
- `eval.gate`: `"off"`. `eval.tta`: false. Main 15-run test is scored. Extra-run test is scored at the locked validation thresholds listed above.

- `cls_r_min`: 0.98 (highest-specificity hard-gate cutoff with scan sensitivity at least this value).
- Night rules: both are reported. A night score is the max, or the mean, of its in-split scan scores. The default aggregation is **max**. Night truth is presence in any full-manifest scan of that operational night. Scan score for a segmenter is `pred_area` at the locked pixel threshold; for the classifier it is `p_cls`. Area cutoffs maximize validation Youden J; ties prefer higher specificity, then a higher cutoff.
- FAR area `A`: 25.0 km² (`eval.far_min_area_km2`).
- Fuzzy width σ: 2.0 px (`eval.bf1_sigma_px`).
- NSD tolerance: 2.0 px. Curve taus: 1, 2, 3, 4, 6, 8, 10, 14, 20 px.
- Bootstrap: night-clustered percentile interval, `n_boot` 2000, `alpha` 0.05, `seed` 0. The paired test's lead interval is this bootstrap on the difference. Wilcoxon over scans is secondary.
