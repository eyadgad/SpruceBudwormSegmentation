# Spruce-budworm radar segmentation — experiment framework

Config-driven PyTorch framework for comparing segmentation architectures on the
XAM (Val d'Irène) weather-radar dataset, detecting biological-scatterer
("dispersal") signal. Phase 1 = data audit, Phase 2 = literature review
(`docs/phase2_literature_review.md`), Phase 3 = this framework. See
`progress.md` for the full record and key decisions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124  # match your CUDA
pip install -r requirements.txt
```

## Data layout (already present under `Data/`)

- `Data/<year>/positives/XAM_<ts>_filtered_ppi.nc` — radar scenes with a dispersal event
- `Data/<year>/negatives/XAM_<ts>_filtered_ppi.nc` — all-background radar scenes
- `Data/Cleaned_Date_2013_2019/Cleaned_dispersal_DBZH_<year>.nc` — yearly masks (one variable per night)
- `Data/Cleaned_Date_2013_2019/xam_dem.nc`, `xam_beam_height_asl.nc` — static terrain grids
- `Data/matched_samples.csv` — positive-sample manifest (its absolute paths are stale; the
  framework re-resolves every path locally from the 12-digit timestamp)

## Run

IMPORTANT: use the project venv interpreter (it has torch/netCDF4/smp), NOT the conda
base `python`. On Windows use `run.bat`, which always invokes `.venv\Scripts\python.exe`.

```bat
:: ONE COMMAND — clean rerun of all experiments end to end (+ ensemble):
run.bat --fresh --ensemble

:: Resume/continue (skips experiments whose result JSON already exists):
run.bat

:: equivalently, without the launcher:
.venv\Scripts\python -m src.run --fresh --ensemble
```

Other entry points (also via the venv python):

```bat
.venv\Scripts\python -m src.data_prep      :: one-time data prep (auto-runs if artifacts missing)
.venv\Scripts\python -m src.evaluate --split test            :: re-evaluate checkpoints, no retraining
.venv\Scripts\python -m src.evaluate --split test --ensemble :: ensemble of finished members
```

Outputs land in `outputs/`: `checkpoints/`, per-experiment `experiments/*_result.json`
and `*_history.csv`, and `comparison_table.{csv,md}`.

## Export the evaluation dashboard

The static dashboard is expected in the sibling
`../sprucebudworm_progress.github.io` checkout and the raw radar archive in
`../Data`. Generate its compressed Sample Explorer assets without running a
model by migrating the existing 480×480 probability and ground-truth PNGs:

```bat
.venv\Scripts\python.exe scripts\export_dashboard_data.py --only packs --data-root ..\Data --site-dir ..\sprucebudworm_progress.github.io
.venv\Scripts\python.exe scripts\test_packed_samples.py --data-root ..\Data --site-dir ..\sprucebudworm_progress.github.io
```

Both path arguments have the sibling locations above as defaults and may be
omitted in the standard workspace. The pack stage refuses to publish partial
coverage: `samples.json` must contain exactly 615 unique timestamps and every
timestamp must resolve to a raw `XAM_<timestamp>_filtered_ppi.nc` volume.

Each scene becomes one deterministic gzip-compressed `SBW1` file plus one
120×120 lossless WebP thumbnail. A pack contains all four 8-bit probability
maps, one bit per ground-truth pixel, and categorical reflectivity. The latter
is computed in physical dBZ from the finite per-cell maximum of raw `TH[0:6]`,
then block-maximum downsampled from 960×960 to 480×480. `samples.json` records
the format, model order, URL templates, dimensions, version, and reflectivity
source under `sample_assets`.

The four viewer models are declared once in `VIEWER_MODELS`. This checkout has
the selected Attention UNet and comparison UNet++ checkpoints; the validated
7- and 8-elevation planes and metrics are preserved from the existing packs
when their checkpoints are unavailable. Supplying those two checkpoint files
lets the same `predict`/`images` stages regenerate all four models directly.

## How it works

- **Config** (`configs/base_config.yaml` + `configs/experiments.yaml`): every experiment
  is `base` deep-merged with its overrides. Vary architecture, loss, channels, target
  definition, patch/training/eval settings — all through config.
- **Channels** (config `channels`): any of `th_e<i>` (reflectivity sweep i), `height_e<i>`,
  `bh_e<i>` (static beam height), `dem`, `valid_mask`. `ETA_raw`/log-ETA channels from the
  reference notebook are **not available locally** (Phase 1) and are intentionally omitted.
- **Split by year** (`split`): whole years are pinned to train/val/test so no scene leaks
  across splits; the balanced default holds out 2014+2018 for test and 2019 for val.
- **Negatives**: included by default at `negatives.ratio` of positives per split
  (all-background target); controls false alarms.
- **Models** (`src/models/`, common interface -> single-channel logits): `unet`,
  `attention_unet`, `nnunet`, `smp_unetpp`, `smp_deeplabv3p`, `smp_segformer`.
- **Losses** (`src/losses.py`): `dice`, `dice_bce`, `focal`, `tversky`, `focal_tversky`, `bce`.
- **Training** (single GPU): AMP autocast + GradScaler, gradient clipping, gradient
  accumulation (`train.accum_steps`), warmup→cosine LR. Cheap patch validation each epoch.
- **Checkpoint/resume**: `<name>_best.pt` (saved the instant val Dice improves),
  `<name>_resume.pt` (full state, written every `train.snapshot_every` epochs; a flag marks
  training complete so a crash during evaluation does not retrain), `<name>_final.pt`,
  `<name>_result.json` (its presence => experiment is skipped on re-run).
- **Final evaluation**: full-scene sliding-window inference (+optional TTA) at a
  val-calibrated threshold on held-out scenes — reported as **macro** (mean per-scene, the
  headline) and **micro** (pixel-pooled). See the metric note below.

## About `nnunet`

`nnunet` here is a self-contained reproduction of nnU-Net's default **2D architecture**
(PlainConvUNet: InstanceNorm + LeakyReLU, strided-conv downsampling), trained inside this
framework's shared loop. It is **not** the full self-configuring nnU-Net autoML pipeline
(which runs its own preprocessing/CV/inference and does not fit a shared interface).

## Reading the results (important)

`dice` in the comparison table is **macro** (mean of per-scene Dice) — every scene counts
equally, so sparse/hard plumes pull it down. The per-epoch `val_dice` and `dice_micro` are
**micro** (pixel-pooled), dominated by dense plumes and therefore higher. The reference
notebook's 0.64 is a *macro* average over only ~10 test scenes (it reported ±0.22 variance);
this framework's macro is over 200+ test scenes and is a far more reliable estimate. Compare
like-for-like: macro-to-macro, micro-to-micro.
