# Runbook — publication retrain (two machines)

The procedure that produces every number in the paper. Supersedes
`RUNBOOK_night_split.md`, which describes the development-screening chain
(S0–S3b) on an older dataset revision.

**15 runs, ~106 h wall clock on two machines** (~184 h on one).

| stage | model | runs | machine |
|---|---|---|---|
| Final spatial model | Attention U-Net | 5 seeds (~106 h) | **A** |
| Spatial baseline | U-Net | 5 seeds (~53 h) | **B** |
| Presence model | Swin-Tiny | 4 new seeds (~25 h); seed 42 already exists | **B** |
| Cascade / oracle | — | evaluation only | A, after merge |

---

## Why the split is frozen, not rebuilt

`data_prep.build_manifest` assembles positives **and sampled negatives** into one row
set *before* `nights.assign_night_split`, which allocates whole nights by targeting
scan-count deficits. Changing the negative population therefore changes **which nights
land in train/val/test**. A manifest rebuilt on a second machine is not the same
experiment.

So: the manifest is mirrored, not regenerated, and both machines verify its SHA-256
before training. Never pass `--fresh`, and never run `src.data_prep` on machine B.

---

## 0. Both machines

```bat
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
set HF_TOKEN=hf_yourWriteToken
```

Both need the full radar archive at `Data\` — the Swin classifier reads the same
full-scene netCDF scans as the segmenters.

**On machine B, delete any pre-existing `artifacts_night\` first.** It must pull the
frozen split rather than reuse a locally built one.

Before the first launch, clear the stale inference cache — it holds the *previous*
segmenter's probability maps and would silently score the old model:

```bat
rmdir /s /q outputs\night_cascade\cache
```

---

## 1. Machine A — start first

```bat
run.bat --base-config configs\base_config_night.yaml ^
        --experiments configs\experiments_night_seeds_a.yaml
```

A publishes the split before training. Wait for:

```
[manifest] sha256=170c442d1dd7b878...
```

Record that hash — it goes in `FROZEN.md`. Only then start machine B.

## 2. Machine B — after the manifest line appears

```bat
:: step 1 — 5x U-Net (~53 h)
run.bat --base-config configs\base_config_night.yaml ^
        --experiments configs\experiments_night_seeds_b.yaml

:: step 2 — 4x Swin-Tiny (~25 h)
.venv\Scripts\python.exe -m src.classify --all ^
        --base-config configs\base_config_cascade_cls.yaml ^
        --experiments configs\experiments_cls_seeds.yaml
```

B must print `sync: manifest sha256=… matches the mirror` before training. If it prints
`MANIFEST MISMATCH`, **stop** — its `Data/` differs from A's and the two halves are not
comparable.

## 3. Interruptions

Re-run the identical command on whichever machine stopped. Finished runs skip, the
interrupted one resumes from its last snapshot (written every epoch), and anything
missing locally is pulled from the mirror. At most one epoch is lost.

## 4. Merge onto machine A

```bat
.venv\Scripts\python.exe scripts\pull_runs.py ^
        --base-config configs\base_config_night.yaml ^
        --experiments configs\experiments_night_seeds_b.yaml

.venv\Scripts\python.exe scripts\pull_runs.py ^
        --base-config configs\base_config_cascade_cls.yaml ^
        --experiments configs\experiments_cls_seeds.yaml
```

Use `pull_runs.py`, not `run.bat`: the latter also pulls, but falls through to
**training** if a pull comes back incomplete, which would silently redo days of work.
`pull_runs.py` never launches a run and exits non-zero naming anything missing.

## 5. Freeze — before test is touched

```bat
.venv\Scripts\python.exe scripts\era_guard.py
```

Expect **15** runs checked. The guard fails if any main-paper run was trained on the
superseded split, or inherits from one via `model.init_checkpoint`.

Then write and **commit** `outputs\night_split\FROZEN.md` containing: manifest SHA-256 ·
final segmentation model · final classifier · all seeds · segmentation threshold ·
classifier threshold · `cls_r_min` · max/mean night rule · FAR area `A` · fuzzy width σ ·
NSD tolerances · bootstrap settings.

## 6. Open the test set — exactly once

```bat
.venv\Scripts\python.exe -m src.finalize ^
        --base-config configs\base_config_night.yaml ^
        --experiments configs\experiments_night_seeds.yaml
```

Note the **combined** YAML (all 10 segmenters), not the per-machine ones.

## 7. Publication outputs

```bat
.venv\Scripts\python.exe scripts\export_variant_comparison.py ^
        --base-config configs\base_config_night.yaml ^
        --experiments configs\experiments_night_seeds.yaml ^
        --baseline unet_night_s42

.venv\Scripts\python.exe scripts\aggregate_seeds.py
```

Writes `outputs/night_split/comparison/`: night-level presence (max vs mean on Swin
p(t), plus the segmented-area baseline), AUPRC, paired U-Net vs Attention U-Net tests,
and `seeds.md` with mean ± SD over 5 seeds per family.

---

## Checks

| check | expected |
|---|---|
| `scripts\era_guard.py` | 15 runs, no superseded config or inherited checkpoint |
| `scripts\aggregate_seeds.py` | `n=5` for `night_base_attunet9` and `unet_night` |
| manifest SHA-256 | identical on both machines and in `FROZEN.md` |
| `eval.gate` in `base_config_night.yaml` | `"off"` — otherwise `finalize` and `export_variant_comparison` disagree |
| oracle-presence row | Dice⁺ gain exactly `+0.0000`; Global Dice gain ≈ `+0.069` |

## Known hazards

- **`--fresh` destroys the frozen split.** Never use it after freezing.
- **`outputs/night_cascade/cache/s0_val.npz`** caches sliding-window maps keyed by
  nothing but the filename. Delete it whenever the segmenter is retrained.
- **`GIT_TERMINAL_PROMPT=0`** is set in some shells here; a git push with an expired
  credential fails immediately rather than prompting.
