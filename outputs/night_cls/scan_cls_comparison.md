# Scan-level swarm vs swarm-free (night split, val)

Classifiers are trained on the scan label. Segmenters are converted with `any pixel > locked threshold => swarm`. AUROC for segmenters uses the per-scan max probability so the ranking is threshold-free.

| name                  | family                 | model            |    auroc |   accuracy |   precision |   recall |       f1 |   threshold |
|:----------------------|:-----------------------|:-----------------|---------:|-----------:|------------:|---------:|---------:|------------:|
| cls_swin_tiny         | classifier             | swin_tiny        | 0.958914 |   0.920596 |    0.942675 | 0.954839 | 0.948718 |        0.6  |
| cls_convnext_tiny     | classifier             | convnext_tiny    | 0.952549 |   0.885856 |    0.968085 | 0.880645 | 0.922297 |        0.85 |
| cls_resnext50         | classifier             | resnext50        | 0.934842 |   0.91067  |    0.936306 | 0.948387 | 0.942308 |        0.4  |
| cls_efficientnetv2_s  | classifier             | efficientnetv2_s | 0.925078 |   0.908189 |    0.947541 | 0.932258 | 0.939837 |        0.5  |
| night_ltae_attunet9   | segmentation_any_pixel | attention_unet   | 0.893722 |   0.794045 |    0.7979   | 0.980645 | 0.879884 |        0.15 |
| night_mtl_attunet9    | segmentation_any_pixel | attention_unet   | 0.852532 |   0.791563 |    0.792746 | 0.987097 | 0.87931  |        0.15 |
| night_bal_attunet9    | segmentation_any_pixel | attention_unet   | 0.816996 |   0.813896 |    0.820163 | 0.970968 | 0.889217 |        0.15 |
| night_tstack_attunet9 | segmentation_any_pixel | attention_unet   | 0.792438 |   0.799007 |    0.797403 | 0.990323 | 0.883453 |        0.15 |
| night_base_attunet9   | segmentation_any_pixel | attention_unet   | 0.787374 |   0.808933 |    0.807388 | 0.987097 | 0.888244 |        0.15 |
