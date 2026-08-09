# Phase 2 — Literature Review

**Scope:** computer-vision methods for detecting/segmenting biological scatterers
(insects, birds) in weather radar, the use of terrain/elevation as auxiliary input,
and modern segmentation architectures suited to this imagery and its severe class
imbalance. Every source below was located and checked online during this review
(URLs given); none is cited from memory alone.

---

## 1. Prior work: classifying biological signal (birds/insects) in weather radar

**The problem is well-established as a segmentation/pixel-classification task, not
generic image segmentation.** The dominant challenge in radar aeroecology is
separating biological echoes (birds, bats, insects) from meteorological echoes
(rain) and ground clutter within a single scan.

- **MistNet** (Lin et al., 2019, *Methods in Ecology and Evolution* 10(11):1908–1922,
  DOI 10.1111/2041-210X.13280) is the most directly relevant prior work. It is a deep
  **convolutional neural network that makes fine-scale, per-pixel predictions** to
  discriminate precipitation from biology in weather-radar scans. Two design ideas
  transfer directly to our project: (a) it is trained from **abundant but noisy labels**
  derived automatically (from dual-polarization data) instead of hand annotation, and
  (b) it performs **dense per-pixel segmentation** over gridded multi-elevation radar
  imagery. Reported performance on WSR-88D data: ≥95.9% of biomass identified at a 1.3%
  false-discovery rate, retaining ~15% more biomass than whole-scan screening.
  This validates our setup, where the ground-truth mask is itself a noisy label
  (finite dBZ = "dispersal present") and the model is a fully-convolutional
  encoder–decoder. Code: https://github.com/adokter/MistNet ;
  paper: https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13280

- **Boulanger et al. (2017)**, *Agricultural and Forest Meteorology* 234–235:127–135 —
  "The use of weather surveillance radar and high-resolution three dimensional weather
  data to monitor a spruce budworm mass exodus flight." This is the **foundational
  domain paper for our exact dataset**: it analyses the **Val d'Irène (XAM) radar** in
  eastern Québec for the **spruce budworm mass exodus of 15–16 July 2013** — the same
  radar and one of the same dates present in `Data/`. It establishes that budworm moths
  disperse downwind in a shallow layer (~400–800 m), which is why the low elevation
  sweeps carry the biological signal.
  https://www.sciencedirect.com/science/article/abs/pii/S0168192316307456
  (open PDF: https://www.fs.usda.gov/nrs/pubs/jrnl/2017/nrs_2017_boulanger_001.pdf)

- **A Machine Learning Approach for Classifying Bird and Insect Radar Echoes with S-Band
  Polarimetric Weather Radar** (*Journal of Atmospheric and Oceanic Technology* 38(10),
  2021, JTECH-D-20-0180.1). Uses dual-pol variables with classical ML (ridge / decision
  tree) to separate bird vs insect echoes; establishes which radar variables are most
  discriminative and that dual-pol products materially improve bird/precip/clutter
  separation.
  https://journals.ametsoc.org/view/journals/atot/38/10/JTECH-D-20-0180.1.xml

- **Schekler et al. (2023)**, *Methods in Ecology and Evolution*,
  "Automatic detection of migrating soaring bird flocks using weather radars by deep
  learning" — a more recent CNN application to radar aeroecology, confirming deep
  learning as the current direction for this task.
  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14161

- **Learning with noisy labels for classifying biological echoes in polarimetric weather
  radar observations using artificial neural networks** (*Neurocomputing*, 2025).
  Reinforces MistNet's noisy-label theme — the labels available for this task are
  inherently imperfect, and networks must be trained accordingly.
  https://www.sciencedirect.com/science/article/pii/S0925231225005648

*Note on novelty:* a targeted search found **no published deep-learning segmentation
work on this specific budworm/Val d'Irène dataset.** The reference notebook is
unpublished internal work, so the Phase 3 architecture comparison is genuinely new.

---

## 2. Terrain / elevation as auxiliary input

Two independent lines of evidence justify feeding the DEM and beam-height grids as
extra input channels:

- **Radar-specific (why the dataset ships these grids):** terrain drives **beam
  blockage** and **ground clutter**, the two dominant non-biological error sources near
  the surface. DEM-based geometry is the standard tool for mapping both. See the
  GIS-based beam-blockage methodology for the US NEXRAD network
  (*Computers & Geosciences*,
  https://www.sciencedirect.com/science/article/abs/pii/S0098300405001548) and
  "Mapping of Weather Radar Ground Clutter Using the Digital Elevation Model (SRTM)"
  (https://www.researchgate.net/publication/268006251). The **beam height above sea
  level** grid encodes range-dependent sampling altitude, which determines whether the
  shallow (~400–800 m) budworm layer is even intercepted at a given range. So DEM +
  beam height give the model exactly the geometric context needed to down-weight clutter
  and reason about detectability.

- **General CV evidence that elevation helps segmentation:** adding elevation/nDSM as an
  input channel consistently improves semantic segmentation of overhead imagery, e.g.
  multimodal fusion of elevation with imagery (MFMINet,
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12887190/) and DEM-derived terrain
  channels (DEM/slope/TPI) for terrain segmentation (DAM-CGNet,
  https://www.sciencedirect.com/science/article/abs/pii/S0098300425001621).

**Implication for Phase 3:** keep DEM and beam-height (elevation-0) as optional input
channels, and make the channel set configurable so their contribution can be measured
by ablation (channels-with-terrain vs channels-without).

---

## 3. Modern SOTA segmentation architectures and class-imbalance handling

The task is binary, single-class, spatially sparse (positive pixels average ~5% of a
scene, down to 0.005% — see Phase 1). The literature points to two levers: (a)
architectures that preserve fine detail and multi-scale context, and (b) loss functions
built for imbalance.

**Architectures**
- **U-Net** (Ronneberger et al., 2015, MICCAI, arXiv:1505.04597) — the encoder–decoder
  baseline; still the reference point for biomedical/scientific segmentation.
  https://arxiv.org/abs/1505.04597
- **Attention U-Net** (Oktay et al., 2018, arXiv:1804.03999) — adds attention gates that
  **suppress irrelevant background and highlight sparse targets**, with gains that are
  largest on small/variable structures and small datasets — a good match for sparse
  biological plumes. It was also the reference notebook's best deep model.
  https://arxiv.org/abs/1804.03999
- **U-Net++** (Zhou et al., 2018, DLMIA, arXiv:1807.10165) — nested, dense skip pathways
  reduce the encoder–decoder semantic gap and improve segmentation of **fine/small
  structures**; ~3–4 IoU points over U-Net. Available with pretrained encoders.
  https://arxiv.org/abs/1807.10165
- **DeepLabV3+** (Chen et al., 2018, ECCV, arXiv:1802.02611) — atrous spatial pyramid
  pooling captures **multi-scale context** (useful for plumes that vary from compact to
  basin-filling) with a decoder that recovers boundaries.
  https://arxiv.org/abs/1802.02611
- **SegFormer** (Xie et al., 2021, NeurIPS, arXiv:2105.15203) — a hierarchical
  **transformer** encoder with a lightweight MLP decoder; strong accuracy/efficiency
  trade-off, robust across resolutions, and small variants (B0–B2) fit a single 16 GB
  GPU. Represents the current transformer SOTA slot.
  https://arxiv.org/abs/2105.15203
- **nnU-Net** (Isensee et al., 2021, *Nature Methods* 18:203–211) — cited as method
  context, not a shortlist model: it shows a **well-configured U-Net still matches or
  beats fancier architectures** when preprocessing, loss, and training are tuned. Its
  lesson (get the pipeline right before the architecture) is baked into the Phase 3
  framework rather than adopted as a separate model.
  https://www.nature.com/articles/s41592-020-01008-z

**Loss functions for imbalance** (varied via config in Phase 3)
- **Focal loss** (Lin et al., 2017, ICCV, arXiv:1708.02002) — down-weights easy
  negatives so the vast background does not swamp training; the notebook found it gave
  the best precision. https://arxiv.org/abs/1708.02002
- **Tversky loss** (Salehi et al., 2017, MLMI, arXiv:1706.05721) — asymmetric FP/FN
  penalty (β>α) to trade precision for recall on rare positives.
  https://arxiv.org/abs/1706.05721
- **Focal Tversky loss** (Abraham & Khan, 2019, ISBI, arXiv:1810.07842) — combines both;
  designed explicitly for **class imbalance and small structures**, reporting large
  gains over plain Dice on small lesions. https://arxiv.org/abs/1810.07842
- Dice+BCE combo (baseline, as in the reference notebook) is retained for comparison.

---

## Shortlist carried into Phase 3

Baseline plus four SOTA architectures, all exposed behind one interface (single-channel
logits at input resolution). Four of the five are available in
`segmentation_models_pytorch` (SMP) with pretrained encoders; Attention U-Net is a small
custom module reused from the reference notebook.

| # | Architecture | Source | Why it's included |
|---|--------------|--------|-------------------|
| 0 | **U-Net** (baseline) | Ronneberger 2015 | Required baseline; mirrors the notebook's `SimpleUNet` so results are directly comparable to its reported ~0.64 full-scene Dice. |
| 1 | **Attention U-Net** | Oktay 2018 | Attention gates focus on sparse positives and suppress clutter; was the notebook's strongest deep model — the one to beat. |
| 2 | **U-Net++** | Zhou 2018 | Dense nested skips help recover the small/fine biological structures that dominate this data; pretrained encoders offset the small dataset. |
| 3 | **DeepLabV3+** | Chen 2018 | ASPP multi-scale context suits plumes spanning a wide range of spatial extents; different inductive bias from the U-Net family. |
| 4 | **SegFormer (MiT-B0–B2)** | Xie 2021 | Modern transformer SOTA; global context and resolution-robustness, with small variants that fit one 16 GB GPU. Represents the non-CNN slot. |

**Losses to sweep (config-driven):** Dice+BCE (baseline), Focal, Tversky (recall-biased),
Focal Tversky. **Auxiliary-channel ablation:** with vs without DEM/beam-height, to
quantify the terrain contribution motivated in §2.

**Feasibility note:** all five run on a single RTX 4080 SUPER (16 GB) at 256×256 patches
with AMP; SegFormer-B2 and heavier SMP encoders may need gradient accumulation, which the
Phase 3 framework will support. The `ETA_raw` channel from the notebook is dropped
(absent locally — Phase 1), so the input-channel set is reduced and made configurable.

---

## References (all verified online during this review)

1. Lin, T.-Y., et al. (2019). MistNet: Measuring historical bird migration in the US using archived weather radar data and convolutional neural networks. *Methods in Ecology and Evolution* 10(11):1908–1922. https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13280
2. Boulanger, Y., Fabry, F., Kilambi, A., Pureswaran, D.S., Sturtevant, B.R., Saint-Amant, R. (2017). The use of weather surveillance radar and high-resolution three dimensional weather data to monitor a spruce budworm mass exodus flight. *Agricultural and Forest Meteorology* 234–235:127–135. https://www.sciencedirect.com/science/article/abs/pii/S0168192316307456
3. A Machine Learning Approach for Classifying Bird and Insect Radar Echoes with S-Band Polarimetric Weather Radar (2021). *J. Atmospheric and Oceanic Technology* 38(10). https://journals.ametsoc.org/view/journals/atot/38/10/JTECH-D-20-0180.1.xml
4. Schekler, I., et al. (2023). Automatic detection of migrating soaring bird flocks using weather radars by deep learning. *Methods in Ecology and Evolution*. https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14161
5. Learning with noisy labels for classifying biological echoes in polarimetric weather radar observations using artificial neural networks (2025). *Neurocomputing*. https://www.sciencedirect.com/science/article/pii/S0925231225005648
6. GIS-based methodology for the assessment of weather radar beam blockage in mountainous regions (US NEXRAD). *Computers & Geosciences*. https://www.sciencedirect.com/science/article/abs/pii/S0098300405001548
7. Mapping of Weather Radar Ground Clutter Using the Digital Elevation Model (SRTM). https://www.researchgate.net/publication/268006251
8. MFMINet: Multimodal fusion and cross-layer interaction network for semantic segmentation of high-resolution remote sensing images. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12887190/
9. DAM-CGNet: Semantic segmentation-based valley-bottom extraction from DEMs. *Computers & Geosciences* (2025). https://www.sciencedirect.com/science/article/abs/pii/S0098300425001621
10. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. arXiv:1505.04597. https://arxiv.org/abs/1505.04597
11. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. arXiv:1804.03999. https://arxiv.org/abs/1804.03999
12. Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA. arXiv:1807.10165. https://arxiv.org/abs/1807.10165
13. Chen, L.-C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+). ECCV. arXiv:1802.02611. https://arxiv.org/abs/1802.02611
14. Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS. arXiv:2105.15203. https://arxiv.org/abs/2105.15203
15. Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18:203–211. https://www.nature.com/articles/s41592-020-01008-z
16. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV. arXiv:1708.02002. https://arxiv.org/abs/1708.02002
17. Salehi, S.S.M., et al. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. MLMI. arXiv:1706.05721. https://arxiv.org/abs/1706.05721
18. Abraham, N., Khan, N.M. (2019). A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation. ISBI. arXiv:1810.07842. https://arxiv.org/abs/1810.07842
