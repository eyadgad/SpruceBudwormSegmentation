# Architecture comparison — held-out full-scene evaluation

Test Dice/IoU are averaged over positive test scenes (comparable to the
reference notebook). `bg_fp_rate` is the mean false-positive pixel fraction on
negative (all-background) test scenes. `REF:` rows are the reference-notebook
baselines this comparison is measured against.

| experiment                                   | model          | loss          |   n_channels |      n_params |   val_dice_patch |   threshold |   test_dice_macro |   test_dice_micro |   test_precision |   test_recall | boundary_iou   | nsd   |   bg_fp_rate | source                                 |
|:---------------------------------------------|:---------------|:--------------|-------------:|--------------:|-----------------:|------------:|------------------:|------------------:|-----------------:|--------------:|:---------------|:------|-------------:|:---------------------------------------|
| REF: RandomForest (notebook best ML)         | random_forest  | -             |          nan | nan           |         nan      |      nan    |            0.6832 |          nan      |         nan      |       nan     |                |       |    nan       | reference_notebook (macro, ~10 scenes) |
| REF: AttentionU-Net+Focal (notebook best DL) | attention_unet | focal         |          nan | nan           |         nan      |      nan    |            0.6412 |          nan      |         nan      |       nan     |                |       |    nan       | reference_notebook (macro, ~10 scenes) |
| gated_attn_unet                              | attention_unet | focal_tversky |           10 |   7.85725e+06 |           0.5966 |        0.15 |            0.033  |            0.1356 |           0.0279 |         0.046 |                |       |      0.00014 | this_framework (val; test deferred)    |
