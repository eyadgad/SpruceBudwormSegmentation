# Evaluation dashboard — implementation notes

Documents the analysis platform under `sprucebudworm_progress.github.io/`, the
export pipeline that feeds it, and the decisions behind both.

---

## 1. What replaced what

The previous site was a single 44 KB `index.html` with five tabs of hard-coded
experiment numbers embedded in a JavaScript array. Every value had to be
transcribed by hand from `outputs/experiments/*_result.json`, which is how an
incorrect claim ("best boundary metrics in the project") survived several
updates.

The new site is a 13-section dashboard whose every number is generated from the
experiment outputs by a script. Nothing is transcribed. The old file is kept at
`outputs/_old_site_index.html` for reference.

## 2. Architecture

Constraint: preserve the stack. The site is hosted on GitHub Pages with no build
step, so it stays plain HTML + CSS + ES modules, with pre-computed JSON standing
in for a backend.

```
Python (offline)                          Browser (online)
────────────────                          ────────────────
outputs/experiments/*  ─┐
artifacts/manifest.csv ─┼─► export_dashboard_data.py ─► data/*.json ─► section module
outputs/checkpoints/*  ─┘         (4 stages)            data/samples/*.png    │
                                                                              ▼
                                                            lib/{metrics,charts,table,data}.js
```

Analysis and presentation are kept apart:

- **`lib/metrics.js`** owns the metric registry (label, definition, formula,
  decimal places, direction) plus the statistics helpers (`mean`, `std`,
  `quantile`, `bootCI`, `wilcoxon`, `pearson`, `spearman`). Sections import from
  here, so a metric cannot be labelled or computed two different ways in two
  places.
- **`lib/charts.js`** returns SVG strings and knows nothing about radar.
- **`lib/table.js`** is a generic sortable/paginated table.
- **`lib/data.js`** owns fetching, caching and the loading/error/empty/N-A states.
- **`sections/*.js`** compose the above. Each exports `render(mount, query)`.

`main.js` is a hash router that `import()`s one section module per route, which
is what gives route-level code splitting on a site with no bundler.

## 3. The export pipeline

`scripts/export_dashboard_data.py`, four independent stages (`--only`):

| Stage | Reads | Writes | Notes |
|---|---|---|---|
| `experiments` | `outputs/experiments/*_{result,config,history,train.log}` | `experiments.json`, `histories.json` | Parses wall-clock duration and per-epoch seconds out of the timestamped logs |
| `dataset` | `artifacts/manifest.csv`, `artifacts/targets/*.npz` | `dataset.json`, `summary.json` | Computes target area under all three label definitions; audits night leakage |
| `predict` | checkpoints + every val/test scene | `samples.json`, `threshold.json` | Full 960×960 sliding-window inference for two models |
| `images` | checkpoints + test/val scenes | `data/samples/*.png` | 4 layers per scene, 615 scenes |

`predict` is the expensive stage (~40 min for 615 scenes × 2 models). Per scene
it records the confusion counts, region metrics, boundary metrics, connected
components and mean radial distance of each error type; it also accumulates
global threshold sweeps, probability histograms, reliability bins and 12-ring
radial error profiles.

### Why probability maps are PNGs

Storing the probability map as an 8-bit greyscale PNG lets the browser
re-threshold a scene in a canvas with zero further network traffic, which is
what makes the threshold slider in the sample explorer instant. The trade-off is
resolution: layers are downsampled 960 → 480 by block **maximum** (so thin
plumes survive), which thickens both masks and makes the in-browser readout
optimistic by a few hundredths of Dice. This is stated in the UI next to the
readout, and every reported metric is computed at full resolution in Python.

Coverage is both the test and validation splits — all 615 evaluated scenes,
2,460 PNGs, 37 MB. `samples.json` carries an `image_splits` field listing the
splits that actually have layers, so the UI reports coverage from the data
rather than assuming it, and a test asserts the two agree.

### Reading the overlay without relying on colour

The viewer's legend is generated per layer, not fixed. For the error view it
lists one row per class with a swatch, the class name, a plain-language meaning
("model says plume, label does not") and a **live pixel count and percentage**
that updates as the threshold moves — so each colour is named and quantified
rather than left to a key the reader has to memorise. The probability layer gets
a labelled 0 → 1 colour bar instead, and the ground-truth/prediction layers get
two-row legends.

An **"Hatch the error types"** toggle adds diagonal hatching in the canvas:
`\` stripes for false positives, `/` for false negatives, solid for correct
overlap. The swatches in the legend carry the same hatching, so the three
classes remain distinguishable in greyscale or with colour-vision deficiency.
This satisfies the "never colour alone" rule inside the raster view, where a
text label cannot be placed on each region.

## 4. Verification

Two test suites, both runnable and both currently passing:

```bash
.venv\Scripts\python.exe scripts\test_dashboard.py   # 60+ checks
node sprucebudworm_progress.github.io/assets/js/lib/metrics.test.js   # 49 checks
```

`test_dashboard.py` covers:
- export helpers (JSON-safety of NaN/Inf, log parsing, radial binning, connected
  components, block-max downsampling);
- that no `NaN`/`Infinity` token leaked into any JSON file;
- **that the recomputed metrics equal the training pipeline's own recorded
  results** for every reported metric;
- per-scene arithmetic (`TP+FP+FN+TN = 921,600`, `TP+FN = truth area`,
  `TP+FP = predicted area`, stored Dice matches its own counts, all metrics in
  `[0,1]`);
- split counts against `split_summary.json`, label-threshold nesting
  (`dbz5 ≤ dbz0 ≤ isfinite`), threshold-sweep monotonicity, image completeness;
- that the site's selection wording still matches the data — it asserts the
  selected run does **not** lead every metric, and tells you to update
  `experiments.js` if that ever changes.

`metrics.test.js` covers the statistics helpers against hand-computed values,
including bootstrap determinism (same input → identical interval) and Wilcoxon
behaviour on shifted and identical inputs.

Browser verification performed: all 13 routes render with no console errors and
no failed requests; filters, sorting, pagination, threshold slider, layer
switching, scene navigation, night-sibling links, deep links, Escape-to-close and
theme persistence all exercised; mobile (375 px) shows no horizontal overflow and
tables scroll within their own container; images all carry `alt`, inputs are
label-associated, sortable headers are keyboard-operable, tooltips are focusable.

## 5. Performance

Measured on a cold load of the overview:

| Metric | Value |
|---|---|
| Files fetched | 8 |
| Transfer | ~97 KB decoded |
| DOMContentLoaded | ~330 ms |
| Heavy files fetched | none (`samples.json`, `histories.json`, section modules deferred) |

Techniques used: route-level dynamic `import()`; a 1 KB `summary.json` for
sections that need only headline counts (instead of the 415 KB `dataset.json`);
`loading="lazy"` thumbnails capped at 120 per grid; paginated tables; canvas
re-thresholding instead of re-fetching; request de-duplication and caching in
`data.js`; hand-written SVG instead of a charting library.

## 6. Findings the dashboard surfaced

These came out of building it and are recorded in `progress.md`:

1. **Night overlap — investigated, and it does NOT inflate the score.**
   170 of 170 positive test scenes share a night with a training scene, which
   looks like textbook leakage. Tested three ways (see section 9): the model
   scores *no better* on scenes it was trained on (0.6222) than on held-out test
   scenes (0.6347), Mann-Whitney p = 0.89. It does not memorise, so there is
   nothing for the overlap to leak.
2. **The selection claim was wrong.** The selected run leads on test Dice only.
   Boundary IoU and NSD belong to the 7-elevation run, best validation Dice to
   the 6-elevation run. Corrected in `progress.md` and on the site, and now
   asserted by a test.
3. **Threshold 0.5 marginally beats the calibrated 0.15** on test Dice
   (0.636 vs 0.635). Dice is nearly flat from 0.15 to 0.6.
4. **The model is over-confident** — reliability bins sit below the diagonal.
   Probabilities are a ranking, not likelihoods.
5. **Target area dominates**: Spearman ρ = 0.732 with per-scene Dice.
6. **The two best models agree at ρ = 0.966** and fail on the same scenes, so
   the difficulty is in the data, not the architecture.
7. **Night-clustered bootstrap intervals are ~40% wider** than scene-level ones.

## 7. Known gaps

Each is shown in the UI as an explicit "not available" state with what it would
take, rather than silently omitted:

| Gap | Needed |
|---|---|
| Per-channel input distributions | A pass over the source netCDF recording histograms |
| Per-scene optimal threshold | One extra sweep pass in `stage_predict` (~200 KB) |
| Weather-conditioned performance | A weather table keyed by night |
| Visual model-vs-model comparison | Run `stage_images` for the comparison checkpoint too |
| Label-noise estimate | A second independent annotation |
| True generalisation estimate | A night-disjoint split and a retrain |

---

## 8. Code review findings (2026-08-05)

A review of the implementation after it shipped. Each item was verified in a
browser before being called a bug, and fixed items now have regression tests.

### Bugs found and fixed

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | A **closed** scene viewer reopened when the user pressed an arrow key | The `keydown` handler was bound on every `open()` but removed only on Escape, so closing via the ✕ or the backdrop left it live | `Modal` owns the handler and always unbinds it in `close()` |
| 2 | Handlers accumulated while paging through scenes | Each re-render bound another handler | The handler is bound once per open sequence, not per render |
| 3 | **The whole page became unscrollable** if the user navigated away with the viewer open | `document.body.style.overflow = 'hidden'` was only reset by `close()`, which a route change never called | Sections can return a `destroy()` hook; the router calls it before replacing a section, and the sample explorer forwards it to `Modal.destroy()` |
| 4 | Focus never entered the dialog, was not trapped, and was not restored on close | No focus management despite `role="dialog" aria-modal="true"` | `Modal` moves focus in (skipping disabled controls), traps Tab/Shift-Tab, and restores focus to the trigger |

Bug 3 was the worst: it left the site unusable until a reload, from an ordinary
sequence of actions (open a scene, click a sidebar link).

### Code quality

- **30 unused imports** across 10 section modules, left behind as sections were
  refactored. All removed; the count is now zero.
- **`card()` was copy-pasted into five sections** and the labelled-`<select>`
  builder into three. Both now live in `lib/ui.js`, so a styling or
  accessibility change is made once.

### Checked and deliberately left alone

- **Canvas repaint cost.** The threshold slider repaints 230,400 pixels per
  input event; measured at ~7 ms, comfortably inside a 60 fps frame. No
  throttling added — it would be complexity for no gain.
- **Escaping.** Every interpolation of scene metadata already goes through
  `esc()`. No injection risk found.
- **Grid cap of 120 tiles.** Deliberate, and the table view covers the rest.

### A correction to an earlier claim

While testing I said arrow navigation "skips scenes". It does not: stepping was
always exact, because each stacked handler recomputed from the same list and the
last one won. The listener leak was real, but its only user-visible effect was
bug 1. The earlier statement overstated what the evidence showed.

### Testing note

`python -m http.server` sends no cache headers, so a browser will happily run a
stale ES module and make a correct fix look broken. Bug 3 appeared unfixed for
this reason. Serve with `Cache-Control: no-store` when editing JavaScript.

---

## 9. Correction: the night-overlap claim was wrong

The dashboard originally led with "100% of test scenes share a night with
training, so every number is optimistic — the single biggest caveat." A reviewer
pushed back with the right question: *if the nights really are that similar,
the test score should be higher than it is.* Leakage only inflates a score if
the model can exploit it. That is measurable, so it was measured.

### The decisive test (`scripts/test_memorization.py`)

Score the selected checkpoint on scenes it was **fitted on**. If shared nights
let it recall training data, train Dice should be clearly higher.

| Split | Model trained on it? | n | Mean Dice |
|---|---|---|---|
| train | yes | 300 | **0.6222** |
| validation | no | 317 | 0.6316 |
| test | no, fully held out | 170 | **0.6347** |

Area-matched train-minus-test gap **+0.0094**; Mann-Whitney **p = 0.89**. Train
is marginally *worse*. Two supporting checks agree: Dice vs. minutes to the
nearest same-night training scan gives ρ = −0.040 (p = 0.60), and same-night
training-scan count, once plume area is controlled for, gives ρ = 0.133
(p = 0.08) while area itself gives ρ = 0.709.

### Why memorisation is absent

Training uses 16 random 256 px crops per scene with hflip/vflip/rot90
augmentation, so the network never sees a fixed 960×960 scene. And scans 30
minutes apart are not the near-duplicates the original claim assumed — plumes
advect and evolve materially between scans.

### What changed

The claim was **removed** from the site rather than left as a visible
retraction: a dashboard should present findings, not a record of a wrong turn.
The overview and data-exploration alarm notes and the three-check defence are
gone. What remains is a compact "Generalisation check" table framed as evidence
about the performance ceiling, plus the night overlap stated as a plain fact of
how the split works. This document and `progress.md` keep the full history,
since they are the engineering log. Deployment readiness
moved "unbiased held-out estimate" from **blocked** to **ready**, and the
night-disjoint re-split dropped from the #1 next experiment to a mid-priority
confirmation.

### What still stands

Night-level generalisation has not been measured *on its own*, and scope remains
one radar and 117 nights. Scenes within a night are still statistically
dependent, so scene-level confidence intervals remain too narrow — that is a
separate issue from leakage and is unchanged.

### The more useful finding underneath

Train Dice equalling test Dice means the model is **under-fitted, not
over-fitted** — it has not even fully fitted its training set. With six
architectures all plateauing near 0.63, the ceiling is **label noise and
genuinely ambiguous plume edges**, not capacity and not the split. That
redirects the top recommendation from "fix the split" to "attack the labels".
