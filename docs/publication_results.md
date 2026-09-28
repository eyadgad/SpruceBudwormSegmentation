# Results

Held-out test performance of the frozen night-split retrain. Five Attention U-Net seeds are compared with five U-Net seeds, scored once at a validation-locked threshold of 0.15. The manifest SHA-256 is `170c442d1dd7b878cdb4157139f7b4dc178f6d0e078534b6dc291d3c7091222f`. The test split contains 161 positive and 147 negative scans, from 21 migration nights and 19 quiet nights. No night is shared with training or validation.

The manuscript page is `publication.html` on the evaluation site. It contains the data description, the five-seed comparison, the attention-gate ablation, robustness by year and boundary tolerance, test-set case plates, Swin-Tiny test detection for seeds 43–46, and the cascade at the Youden and high-sensitivity operating points. Figures are written by `scripts/plot_publication.py` and `scripts/publication_finish.py`.

## Segmentation

Both families use the same ten channels (nine reflectivity elevations and a validity mask), the same focal-Tversky loss, the same patch sampling, and the same 100-epoch schedule. The designed difference is the decoder: attention gates versus a plain U-Net. Seeds 42–46 change the initialisation and the random crops. The test set was not read until those choices had been recorded in `outputs/night_split/FROZEN.md`.

Macro Dice, the mean of per-scene Dice on the 161 positive test scans, is 0.526 ± 0.005 for Attention U-Net and 0.517 ± 0.009 for U-Net (mean ± sample standard deviation, five seeds). Pixel-pooled Dice on those positive scans is 0.654 ± 0.006 and 0.651 ± 0.003. Pooling quiet scans in as well gives 0.595 ± 0.012 and 0.598 ± 0.006. The architecture gap is about one macro-Dice point. It is smaller than the seed range inside the U-Net family (0.504 to 0.528) and much smaller than the drop from validation to test.

| Family | Dice | Dice, pooled | Precision | Recall | NSD | Fuzzy BF1 |
|---|---|---|---|---|---|---|
| Attention U-Net | 0.526 ± 0.005 | 0.654 ± 0.006 | 0.538 ± 0.021 | 0.594 ± 0.033 | 0.323 ± 0.008 | 0.334 ± 0.006 |
| U-Net | 0.517 ± 0.009 | 0.651 ± 0.003 | 0.553 ± 0.020 | 0.569 ± 0.030 | 0.305 ± 0.019 | 0.318 ± 0.017 |

NSD and fuzzy boundary F1 use a 2-pixel tolerance. Precision is higher for the plain U-Net. Recall and the boundary scores are higher for Attention U-Net.

## Pre-specified paired comparison

The confirmatory contrast named before the test set was opened is seed 42 against seed 42, on the same 161 positive scans, with nights as the resampling unit (2,000 night-clustered bootstrap draws, 21 nights). Attention U-Net minus U-Net is +0.0037 Dice (95% interval −0.0084 to +0.0170). The interval includes zero. The same contrast is +0.0206 for normalised surface distance (0.0082 to 0.0319) and +0.0179 for fuzzy boundary F1 (0.0062 to 0.0290). Those two intervals exclude zero. On Dice, seed 42 does not establish a difference. On the boundary metrics, it does.

## Robustness

The robustness check repeats the paired contrast at every matched seed, then asks whether the ranking survives the year and the move from validation to test. The threshold is not re-tuned: every run had already selected 0.15 on validation.

| Seed | Dice difference | 95% interval | NSD difference | 95% interval |
|---|---|---|---|---|
| 42 | +0.0037 | −0.0084 to +0.0170 | +0.0206 | +0.0082 to +0.0319 |
| 43 | +0.0009 | −0.0064 to +0.0083 | −0.0079 | −0.0133 to −0.0020 |
| 44 | +0.0097 | +0.0022 to +0.0184 | +0.0212 | +0.0119 to +0.0306 |
| 45 | +0.0227 | +0.0037 to +0.0478 | +0.0512 | +0.0371 to +0.0679 |
| 46 | +0.0118 | +0.0015 to +0.0229 | +0.0062 | −0.0020 to +0.0168 |

Dice is positive at all five seeds, and the interval excludes zero at three of them. NSD excludes zero in favour of Attention U-Net at seeds 42, 44 and 45, and in favour of U-Net at seed 43. The Dice advantage is same-signed and small. Two of the five intervals, including the pre-specified seed, include zero. The boundary advantage does not survive the seed check.

The year profile is shared. Both models are weak in 2013 (14 positive test scans; mean Dice 0.37 and 0.33) and stronger in 2014 and 2019 (0.66 and 0.63; 0.66 and 0.65). Attention U-Net is ahead in six of the seven years. In 2018 (16 scans) U-Net is ahead by 0.002. The absolute level moves by almost 0.3 Dice across years, which is an order of magnitude larger than the architecture gap.

Validation macro Dice sits between 0.553 and 0.571. Test macro Dice sits between 0.504 and 0.532. The mean drop is 0.041 for Attention U-Net and 0.042 for U-Net. Either family’s validation number overstates its test number by about four points. That shift, not the choice of decoder, is the larger fact in the experiment.

The false-alarm rate at 25 km² is 0.561 ± 0.055 and 0.550 ± 0.048. A little over half of quiet test scans still contain a false region of that size, in both families. Sensitivity retained at the same area is 0.94 and 0.93.

## Scan and night presence from the segmenter

Calling a scan positive when its predicted area exceeds a validation-chosen cutoff, the test ROC area is 0.830 for Attention U-Net and 0.833 for U-Net. Aggregating scans to the operational night by the maximum area gives 0.804 and 0.810; aggregating by the mean gives 0.851 and 0.851. These areas do not separate the two architectures. A night-level score built from segmented area ranks migration nights above quiet nights well above chance, and the mean is the stronger of the two reductions on this split.

## What this section does not estimate

The Swin-Tiny presence model was trained for five seeds. Its validation ROC areas are 0.941, 0.948, 0.927, 0.919 and 0.932. The seed-42 checkpoint used to attach scan probabilities is not on disk and not on the checkpoint mirror, so those probabilities were not scored on the test set. The cascade and the oracle-presence row were not run. Development-screening runs on the earlier, unbalanced split are omitted.

The historical dashboard model is omitted. It was trained on a scan-level split in which most nights appeared in more than one of train, validation and test. A lower Dice here than that model’s published figure is the expected consequence of removing that leakage.
