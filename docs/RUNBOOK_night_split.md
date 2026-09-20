# Runbook — night-split ablation (SUPERSEDED)

> ## ⚠ Do not follow this for the paper
>
> This describes the S0–S3b ablation chain, which ran on an **earlier dataset
> revision** (`negatives: ratio 0.3`, unbalanced — ~1440 training scans). The
> current frozen split uses `balanced: true, ratio: 1.0` (2158 training scans),
> and because negatives are sampled *before* the night→split assignment, the two
> eras do not even share the same validation scenes.
>
> Those runs are **development-stage architecture screening**: they informed which
> architecture was carried forward, and their numbers must never appear beside
> publication results.
>
> **For the publication retrain, use [`RUNBOOK_publication.md`](RUNBOOK_publication.md).**
>
> Kept here because the resume/interrupt mechanics, the timing figures, and the
> troubleshooting table below still describe how the trainer behaves.

How to run, interrupt, and resume the night-split experiment chain. Every command
is safe to re-run: finished experiments are skipped and an interrupted one
continues from its last snapshot.

Run all commands from the repo root (the directory containing `src/` and `configs/`).

---

## Safe to interrupt

Yes. `src/experiment.py` writes `<name>_resume.pt` **every epoch**
(`train.snapshot_every: 1`) containing model, optimizer, scheduler, AMP scaler,
epoch counter, best-so-far, early-stop counter, full history and RNG state. The
write goes to a `.tmp` file and is then atomically renamed, so a power loss
mid-write cannot corrupt it.

* **Ctrl-C, reboot, or crash** -> at most one epoch is lost.
* **Re-running the same command** -> `src/run.py` skips any experiment that
  already has `<name>_result.json`, and resumes the one that does not.
* A crash *after* training but *during* evaluation does not retrain: the snapshot
  carries a `training_complete` flag.

To stop cleanly, Ctrl-C the console, or:

```bat
taskkill /F /T /IM python.exe
```

Note: the venv `python.exe` is a redirector shim, so each logical run shows as
**two** PIDs (shim + child). That is normal, not a duplicate run.

**The machine must stay awake.** Sleep kills detached runs. Set the power plan to
never sleep before leaving it unattended.

---

## The sequence

### 0. One-time — already done

```bat
.venv\Scripts\python.exe -m src.data_prep --base-config configs\base_config_night.yaml
.venv\Scripts\python.exe scripts\test_night_split.py
```

Rebuilds `artifacts_night/` (manifest, split summary, norm stats). Only needed
again if the split config changes. It does **not** touch `artifacts/`.

### 1. Lambda probes — validation only

```bat
.venv\Scripts\python.exe -m src.run ^
    --base-config configs\base_config_night.yaml ^
    --experiments configs\experiments_night_lambda.yaml
```

Three 25-epoch runs (lambda = 0.1 / 0.3 / 1.0) ranked on validation patch Dice.
The test set is never read (`eval.defer_test: true`).

### 2. Pick lambda, then edit the config

```bat
.venv\Scripts\python.exe scripts\show_lambda_probes.py
```

Put the winning value into `lambda_cls:` for `night_mtl_attunet9`,
`night_tstack_attunet9` and `night_ltae_attunet9` in
`configs/experiments_night.yaml`. **Do this before step 3** — otherwise those
stages run at the placeholder 0.3.

### 3. The five-stage chain

```bat
.venv\Scripts\python.exe -m src.run ^
    --base-config configs\base_config_night.yaml ^
    --experiments configs\experiments_night.yaml
```

Runs S0 -> S1 -> S2 -> S3a -> S3b in order, each in its own subprocess so a CUDA
OOM cannot take down the rest. Re-run this exact command after any interruption.

### 4. Freeze, then score the test set once

```bat
.venv\Scripts\python.exe -m src.finalize ^
    --base-config configs\base_config_night.yaml ^
    --experiments configs\experiments_night.yaml
```

Writes `<name>_final_result.json` using the threshold already calibrated on
validation — it does not recalibrate, which would be fitting to test. Leaves
`<name>_result.json` alone so `src.run` still treats the run as finished.

### 5. Compare all variants

```bat
.venv\Scripts\python.exe scripts\export_variant_comparison.py ^
    --base-config configs\base_config_night.yaml ^
    --experiments configs\experiments_night.yaml
```

Writes `outputs/night_split/comparison/`: per-variant `samples.json` and
`presence_<name>.json` (from `src.presence.analyze_presence`, unchanged), plus
`comparison.{csv,md}` and `classification.{csv,md}`.

### 6. Re-run the tests

```bat
.venv\Scripts\python.exe scripts\test_night_split.py
.venv\Scripts\python.exe scripts\test_presence.py
```

---

## Checking progress

```bat
.venv\Scripts\python.exe scripts\show_night_progress.py
dir outputs\night_split\experiments\*_result.json
nvidia-smi
```

Expect roughly **900-1100 s/epoch** for the single-frame stages (S0-S2) and
**~2-3x that** for the temporal stages (S3a, S3b), which load three scenes per
sample. Training is disk-bound, not GPU-bound, so `nvidia-smi` showing ~20%
utilisation is normal and not a problem to fix. Avoid running other disk-heavy
work concurrently — it measurably slows the run.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| A run is skipped unexpectedly | `<name>_result.json` exists | Delete that one file to force a retrain |
| `no checkpoint for <name>` | training never beat `es_tolerance`, so no `_best.pt` | `finalize` / the export fall back to `_final.pt` and say so |
| Want to redo one stage | | Delete its `_result.json`, `_resume.pt`, `_best.pt`, `_final.pt` |
| Want to redo everything | | Delete `outputs/night_split/` (keeps `artifacts_night/`) |
| Out of disk | each run keeps ~95 MB resume + 2x 31 MB checkpoints | Resume files are deleted automatically on clean completion |

Never delete `artifacts/` — it holds the old scan-level split that the published
dashboard model depends on. The night split lives in `artifacts_night/`.
