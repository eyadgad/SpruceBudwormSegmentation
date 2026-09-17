# Architecture comparison — held-out full-scene evaluation

Test Dice/IoU are averaged over positive test scenes (comparable to the
reference notebook). `bg_fp_rate` is the mean false-positive pixel fraction on
negative (all-background) test scenes. `REF:` rows are the reference-notebook
baselines this comparison is measured against.

| experiment                                   | model          | loss   | n_channels   | n_params   | val_dice_patch   | threshold   |   test_dice_macro | test_dice_micro   | test_precision   | test_recall   | boundary_iou   | nsd   | bg_fp_rate   | source                                 |
|:---------------------------------------------|:---------------|:-------|:-------------|:-----------|:-----------------|:------------|------------------:|:------------------|:-----------------|:--------------|:---------------|:------|:-------------|:---------------------------------------|
| REF: RandomForest (notebook best ML)         | random_forest  | -      |              |            |                  |             |            0.6832 |                   |                  |               |                |       |              | reference_notebook (macro, ~10 scenes) |
| REF: AttentionU-Net+Focal (notebook best DL) | attention_unet | focal  |              |            |                  |             |            0.6412 |                   |                  |               |                |       |              | reference_notebook (macro, ~10 scenes) |
