# Architecture comparison — held-out full-scene evaluation

Test Dice/IoU are averaged over positive test scenes (comparable to the
reference notebook). `bg_fp_rate` is the mean false-positive pixel fraction on
negative (all-background) test scenes. `REF:` rows are the reference-notebook
baselines this comparison is measured against.

| experiment                                   | model          | loss          |   n_channels |      n_params |   val_dice_patch |   threshold |   test_dice_macro |   test_dice_micro |   test_precision |   test_recall |   boundary_iou |      nsd |   bg_fp_rate | source                                 |
|:---------------------------------------------|:---------------|:--------------|-------------:|--------------:|-----------------:|------------:|------------------:|------------------:|-----------------:|--------------:|---------------:|---------:|-------------:|:---------------------------------------|
| REF: RandomForest (notebook best ML)         | random_forest  | -             |          nan | nan           |         nan      |      nan    |            0.6832 |          nan      |         nan      |      nan      |       nan      | nan      |    nan       | reference_notebook (macro, ~10 scenes) |
| REF: AttentionU-Net+Focal (notebook best DL) | attention_unet | focal         |          nan | nan           |         nan      |      nan    |            0.6412 |          nan      |         nan      |      nan      |       nan      | nan      |    nan       | reference_notebook (macro, ~10 scenes) |
| stat_maxmed_std                              | attention_unet | focal_tversky |            2 |   7.85443e+06 |           0.6854 |        0.15 |            0.6038 |            0.6982 |           0.5591 |        0.7184 |         0.3797 |   0.3465 |      0.00626 | this_framework                         |
| stat_max_raw                                 | attention_unet | focal_tversky |            1 |   7.85414e+06 |           0.6702 |        0.15 |            0.6034 |            0.697  |           0.559  |        0.7169 |         0.3682 |   0.3064 |      0.00683 | this_framework                         |
| stat_maxmedmean_raw                          | attention_unet | focal_tversky |            3 |   7.85472e+06 |           0.6896 |        0.15 |            0.6011 |            0.7048 |           0.5764 |        0.6922 |         0.3787 |   0.3296 |      0.00546 | this_framework                         |
| stat_max_std                                 | attention_unet | focal_tversky |            1 |   7.85414e+06 |           0.6657 |        0.15 |            0.6006 |            0.6898 |           0.5497 |        0.7359 |         0.3687 |   0.3181 |      0.00756 | this_framework                         |
| stat_maxmed_raw                              | attention_unet | focal_tversky |            2 |   7.85443e+06 |           0.6819 |        0.15 |            0.597  |            0.6977 |           0.5801 |        0.6858 |         0.3704 |   0.3167 |      0.00523 | this_framework                         |
| stat_maxmedmean_std                          | attention_unet | focal_tversky |            3 |   7.85472e+06 |           0.6874 |        0.15 |            0.5962 |            0.6968 |           0.5601 |        0.7027 |         0.3713 |   0.3236 |      0.00606 | this_framework                         |
