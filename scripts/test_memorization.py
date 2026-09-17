"""Does the model memorise? Evaluate the selected checkpoint on TRAINING scenes.

Motivation: the manifest splits by year, so every test night also appears in
training. That is usually called leakage. But leakage only inflates a score if
the model can exploit it, and the counter-argument is simple: if test scenes
were effectively near-duplicates of training scenes, the test score should be
far higher than it is.

The clean way to settle it is to score the model on data it was literally
trained on. If train Dice is close to test Dice, there is no memorisation to
leak, and the ~0.63 plateau is a property of the task and the labels rather
than of the split. If train Dice is much higher, memorisation is real and the
test number is optimistic.

Because plume area dominates per-scene Dice, the comparison is also reported
area-matched.

Run:  .venv\\Scripts\\python.exe scripts\\test_memorization.py [--n 300]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import export_dashboard_data as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="random training scenes to score")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from src import config as cfgmod, data_prep, dataset as dsmod, engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man, norm = data_prep.load_artifacts(base)
    model, cfg, THR = X._load_model(X.SELECTED, device)
    ps = int(cfg["patch"]["size"]); ov = float(cfg["eval"].get("overlap", 0.5))
    print(f"model={X.SELECTED} threshold={THR} device={device}")

    tr = man[(man.split == "train") & (man.label == 1)]
    if args.n and args.n < len(tr):
        tr = tr.sample(args.n, random_state=args.seed)
    rows = tr.to_dict("records")
    print(f"scoring {len(rows)} TRAINING scenes (model saw these during fitting)")

    out = []
    eps = 1e-8
    for i, r in enumerate(rows, 1):
        x, y = dsmod.load_full_scene(cfg, r, norm)
        prob = engine.sliding_window_predict(model, x, device, ps, ov, False, gaussian=True)
        pred = prob > THR
        yb = y.astype(bool)
        tp = float((pred & yb).sum()); fp = float((pred & ~yb).sum()); fn = float((~pred & yb).sum())
        out.append({
            "ts": int(r["timestamp"]), "split": "train",
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps), "recall": tp / (tp + fn + eps),
            "gt_area": int(yb.sum()),
        })
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    tr_df = pd.DataFrame(out)
    S = json.loads((ROOT / "sprucebudworm_progress.github.io" / "data" / "samples.json").read_text())["samples"]
    ev = pd.DataFrame([s for s in S if s["label"] == 1])[["split", "dice", "precision", "recall", "gt_area"]]
    allo = pd.concat([tr_df[["split", "dice", "precision", "recall", "gt_area"]], ev])

    print("\n" + "=" * 66)
    print("PER-SPLIT DICE  (train = scenes the model was fitted on)")
    print("=" * 66)
    print(allo.groupby("split").agg(
        n=("dice", "size"), mean_dice=("dice", "mean"), median_dice=("dice", "median"),
        mean_prec=("precision", "mean"), mean_rec=("recall", "mean"),
        median_area=("gt_area", "median")).round(4).to_string())

    # area-matched: plume size dominates Dice, so compare like with like
    bands = [(0, 1e3, "<1k"), (1e3, 5e3, "1k-5k"), (5e3, 2e4, "5k-20k"),
             (2e4, 5e4, "20k-50k"), (5e4, 1.5e5, "50k-150k"), (1.5e5, 1e9, ">150k")]
    print("\n" + "=" * 66)
    print("AREA-MATCHED DICE  (train vs test within the same plume-size band)")
    print("=" * 66)
    print(f"{'band':>10} | {'n_train':>7} {'train':>7} | {'n_test':>6} {'test':>7} | {'gap':>7}")
    print("-" * 66)
    gaps = []
    for lo, hi, lab in bands:
        a = allo[(allo.split == "train") & (allo.gt_area >= lo) & (allo.gt_area < hi)]
        b = allo[(allo.split == "test") & (allo.gt_area >= lo) & (allo.gt_area < hi)]
        if len(a) < 3 or len(b) < 3:
            print(f"{lab:>10} | {len(a):>7} {'-':>7} | {len(b):>6} {'-':>7} | {'too few':>7}")
            continue
        g = a.dice.mean() - b.dice.mean()
        gaps.append((g, len(b)))
        print(f"{lab:>10} | {len(a):>7} {a.dice.mean():>7.4f} | {len(b):>6} {b.dice.mean():>7.4f} | {g:>+7.4f}")
    if gaps:
        w = sum(g * n for g, n in gaps) / sum(n for _, n in gaps)
        print("-" * 66)
        print(f"{'weighted':>10} | area-matched train-minus-test Dice gap: {w:+.4f}")

    from scipy import stats
    t = allo[allo.split == "train"].dice.values
    e = allo[allo.split == "test"].dice.values
    u = stats.mannwhitneyu(t, e, alternative="two-sided")
    print(f"\nMann-Whitney train vs test: U p = {u.pvalue:.4f}")
    print(f"raw train-minus-test mean Dice: {t.mean() - e.mean():+.4f}")

    print("\n" + "=" * 66)
    print("INTERPRETATION")
    print("=" * 66)
    if abs(w if gaps else t.mean() - e.mean()) < 0.03:
        print("The model scores essentially the SAME on scenes it trained on as on held-out")
        print("scenes. There is no memorisation for the shared nights to leak, so the night")
        print("overlap is NOT inflating the reported score. The ~0.63 plateau is a property")
        print("of the task/labels, not of the split.")
    else:
        print("The model scores materially higher on scenes it trained on, so memorisation")
        print("is real and the shared nights plausibly inflate the held-out score.")

    (ROOT / "outputs" / "memorization_check.json").write_text(json.dumps({
        "n_train_scored": len(tr_df),
        "train_mean_dice": float(t.mean()), "test_mean_dice": float(e.mean()),
        "area_matched_gap": float(w) if gaps else None,
        "mannwhitney_p": float(u.pvalue),
    }, indent=2))
    print("\nwrote outputs/memorization_check.json")


if __name__ == "__main__":
    main()
