"""Publication figures for the frozen night-split test set.

Writes PNG and PDF next to the evaluation site. The figures are the ones the
publication export was built to support: the five-seed aggregate, the
pre-specified paired test, the NSD-versus-tolerance curve with a
night-clustered band, scan/night ROC and precision-recall (AUPRC) for the
segmented-area score under max and mean night aggregation, and skill by plume
size. Also prints the pooled detection scores and the oracle gate, and writes
the test table of the extra runs listed in FROZEN.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import cluster_bootstrap_curve, paired_cluster_bootstrap  # noqa: E402

COMP = ROOT / "outputs" / "night_split" / "comparison"
OUT = ROOT / "sprucebudworm_progress.github.io" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44, 45, 46)
ATTN = "#0072B2"
UNET = "#D55E00"
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def _samples(name: str):
    path = COMP / f"{name}_samples.json"
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def _presence(name: str):
    return json.loads((COMP / f"presence_{name}.json").read_text(encoding="utf-8"))


def _save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def _pos(rows, split):
    return [r for r in rows if r.get("split") == split and int(r.get("label", 0)) == 1
            and r.get("dice") is not None]


def fig_seeds(cache) -> None:
    metrics = [
        ("dice", "Dice"),
        ("dice_pool", "Pooled Dice"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("nsd", "NSD"),
        ("bf1_fuzzy", "Fuzzy BF1"),
    ]
    # Pooled Dice is not a per-scene field; compute from tp/fp/fn.
    def pooled(rows):
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        return 2 * tp / (2 * tp + fp + fn)

    means, sds = {}, {}
    for fam, color in (("attn", ATTN), ("unet", UNET)):
        cols = {k: [] for k, _ in metrics}
        for seed in SEEDS:
            rows = _pos(cache[fam][seed], "test")
            for key, _ in metrics:
                cols[key].append(pooled(rows) if key == "dice_pool"
                                 else float(np.mean([r[key] for r in rows])))
        means[fam] = {k: float(np.mean(v)) for k, v in cols.items()}
        sds[fam] = {k: float(np.std(v, ddof=1)) for k, v in cols.items()}

    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    x = np.arange(len(metrics))
    w = 0.34
    for shift, fam, color, label in (
        (-w / 2, "attn", ATTN, "Attention U-Net"),
        (w / 2, "unet", UNET, "U-Net"),
    ):
        y = [means[fam][k] for k, _ in metrics]
        e = [sds[fam][k] for k, _ in metrics]
        ax.bar(x + shift, y, w, yerr=e, color=color, ecolor=INK, capsize=2.5,
               error_kw={"linewidth": 0.7, "capthick": 0.7}, label=label, zorder=2)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set_ylabel("Test score")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    _save(fig, "fig1_seeds")


def _pair(cache, seed, split, keys):
    a = {int(r["ts"]): r for r in cache["attn"][seed]
         if r.get("split") == split and int(r.get("label", 0)) == 1}
    b = {int(r["ts"]): r for r in cache["unet"][seed]
         if r.get("split") == split and int(r.get("label", 0)) == 1}
    common = sorted(set(a) & set(b))
    nights = [a[t]["night"] for t in common]
    out = {}
    for key in keys:
        boot = paired_cluster_bootstrap(
            [a[t].get(key) for t in common], [b[t].get(key) for t in common], nights)
        out[key] = boot
    return out


def fig_paired(cache) -> None:
    keys = ["dice", "iou", "precision", "recall", "nsd", "bf1_fuzzy"]
    labels = ["Dice", "IoU", "Precision", "Recall", "NSD", "Fuzzy BF1"]
    est = _pair(cache, 42, "test", keys)
    y = np.arange(len(keys))[::-1]
    delta = [est[k]["point"] for k in keys]
    lo = [est[k]["point"] - est[k]["lo"] for k in keys]
    hi = [est[k]["hi"] - est[k]["point"] for k in keys]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    ax.errorbar(delta, y, xerr=[lo, hi], fmt="o", color=ATTN, ecolor=INK,
                elinewidth=0.8, capsize=2.5, markersize=4.5, zorder=2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Attention U-Net $-$ U-Net")
    ax.set_xlim(-0.06, 0.06)
    _save(fig, "fig2_paired")
    return est


def fig_robust(cache) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharey=True)
    for ax, key, title in (
        (axes[0], "dice", "Dice"),
        (axes[1], "nsd", "NSD"),
    ):
        ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
        ys, deltas, los, his = [], [], [], []
        for i, seed in enumerate(SEEDS):
            est = _pair(cache, seed, "test", [key])[key]
            ys.append(i)
            deltas.append(est["point"])
            los.append(est["point"] - est["lo"])
            his.append(est["hi"] - est["point"])
        ax.errorbar(deltas, ys, xerr=[los, his], fmt="o", color=ATTN, ecolor=INK,
                    elinewidth=0.8, capsize=2.5, markersize=4.5)
        ax.set_yticks(range(len(SEEDS)), [f"Seed {s}" for s in SEEDS])
        ax.set_xlabel("Attention U-Net $-$ U-Net")
        ax.set_title(title, loc="left", fontweight="regular")
        ax.invert_yaxis()
    _save(fig, "fig3_robustness")


def fig_nsd(cache) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for fam, color, label in (
        ("attn", ATTN, "Attention U-Net"),
        ("unet", UNET, "U-Net"),
    ):
        rows = _pos(cache[fam][42], "test")
        taus = rows[0]["nsd_curve"]["taus_px"]
        curves = [r["nsd_curve"]["nsd"] for r in rows]
        nights = [r["night"] for r in rows]
        band = cluster_bootstrap_curve(curves, nights)
        x = np.asarray(taus, float)
        y = np.asarray(band["point"], float)
        lo = np.asarray(band["lo"], float)
        hi = np.asarray(band["hi"], float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.4, label=label)
    ax.set_xlabel("Boundary tolerance (px)")
    ax.set_ylabel("Normalised surface distance")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig4_nsd")


def _curve(block, kind):
    pts = block[kind]["points"]
    if kind == "roc":
        x = [p["false_positive_rate"] for p in pts]
        y = [p["true_positive_rate"] for p in pts]
        score = block["roc"]["auc"]
    else:
        x = [p["recall"] for p in pts]
        y = [p["precision"] for p in pts]
        score = block["pr"]["ap"]
    order = np.argsort(x)
    return np.asarray(x, float)[order], np.asarray(y, float)[order], score


def fig_presence() -> None:
    models = {
        "attn": (_presence("night_base_attunet9_s42")["models"][0], ATTN, "Attention U-Net"),
        "unet": (_presence("unet_night_s42")["models"][0], UNET, "U-Net"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    specs = [
        (axes[0], "roc", "False-positive rate", "True-positive rate", "ROC"),
        (axes[1], "pr", "Recall", "Precision", "Precision–recall"),
    ]
    styles = {
        ("scan",): ("-", 1.5),
    }
    for ax, kind, xlab, ylab, title in specs:
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=0)
        for fam, (model, color, label) in models.items():
            for agg, ls, name in (
                (None, "-", "scan"),
                ("max", "--", "night max"),
                ("mean", ":", "night mean"),
            ):
                block = model["scan"]["splits"]["test"] if agg is None else model["night"][agg]["splits"]["test"]
                x, y, score = _curve(block, kind)
                metric = "AUC" if kind == "roc" else "AP"
                ax.plot(x, y, color=color, linestyle=ls, linewidth=1.35,
                        label=f"{label}, {name} ({metric} {score:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="regular")
        if kind == "pr":
            base = models["attn"][0]["scan"]["splits"]["test"]["pr"]["baseline_precision"]
            ax.axhline(base, color="#bbbbbb", linewidth=0.6, linestyle="--", zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save(fig, "fig5_presence")


def oracle(cache) -> None:
    """Perfect presence gate: zero every quiet scan. Macro Dice cannot move."""
    for fam in ("attn", "unet"):
        for seed in (42,):
            rows = [r for r in cache[fam][seed] if r.get("split") == "test"]
            pos = [r for r in rows if int(r["label"]) == 1]
            tp = sum(r["tp"] for r in pos)
            fp_pos = sum(r["fp"] for r in pos)
            fn = sum(r["fn"] for r in pos)
            fp_all = sum(r["fp"] for r in rows)
            fp_oracle = fp_pos  # quiet scans contribute no false pixels
            def g(fp):
                return 2 * tp / (2 * tp + fp + fn)
            macro = float(np.mean([r["dice"] for r in pos]))
            print(f"oracle {fam} s{seed} macro {macro:.4f} "
                  f"global {g(fp_all):.4f} oracle_global {g(fp_oracle):.4f} "
                  f"gain {g(fp_oracle) - g(fp_all):+.4f}")


def fig_year(cache) -> None:
    """Skill against calendar year, the stability axis analogous to lead time."""
    years = list(range(2013, 2020))
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    for fam, color, label in (("attn", ATTN, "Attention U-Net"), ("unet", UNET, "U-Net")):
        means, sds, ns = [], [], []
        for year in years:
            vals = []
            n = None
            for seed in SEEDS:
                rows = [r for r in _pos(cache[fam][seed], "test") if int(r["year"]) == year]
                n = len(rows)
                vals.append(float(np.mean([r["dice"] for r in rows])) if rows else np.nan)
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)))
            ns.append(n)
        x = np.asarray(years, float)
        y = np.asarray(means)
        e = np.asarray(sds)
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, marker="o", markersize=4.5, linewidth=1.4, label=label)
    ax.set_xticks(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Test macro Dice")
    ax.set_ylim(0.2, 0.8)
    # Sample size is identical for both families.
    ymin = 0.22
    for year, n in zip(years, ns):
        ax.text(year, ymin, f"n={n}", ha="center", va="bottom", fontsize=7, color="#555555")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig6_year")


CELL_KM2 = 0.25  # one 500 m grid cell


def fig_size(cache) -> None:
    """Test macro Dice by quartile of labelled area (equal-count bins on the test positives)."""
    import pandas as pd
    ref = {int(r["ts"]): r for r in _pos(cache["attn"][42], "test")}
    area = pd.Series({ts: r["gt_area"] for ts, r in ref.items()})
    quart = pd.qcut(area, 4, labels=False)
    bins = [sorted(area.index[quart == q]) for q in range(4)]
    labels = [f"{area[b].min() * CELL_KM2:,.0f}–{area[b].max() * CELL_KM2:,.0f}" for b in bins]
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    x = np.arange(4)
    w = 0.34
    for shift, fam, color, label in ((-w / 2, "attn", ATTN, "Attention U-Net"),
                                     (w / 2, "unet", UNET, "U-Net")):
        per_seed = []
        for seed in SEEDS:
            rows = {int(r["ts"]): r for r in _pos(cache[fam][seed], "test")}
            per_seed.append([float(np.mean([rows[ts]["dice"] for ts in b])) for b in bins])
        per_seed = np.asarray(per_seed)
        m, sd = per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)
        ax.bar(x + shift, m, w, yerr=sd, color=color, ecolor=INK, capsize=2.5,
               error_kw={"linewidth": 0.7, "capthick": 0.7}, label=label, zorder=2)
        for q in range(4):
            print(f"size Q{q + 1} n={len(bins[q])} {labels[q]} km2 {fam} {m[q]:.3f} ± {sd[q]:.3f}")
    ax.set_xticks(x, [f"Q{q + 1}\n{labels[q]} km²\nn={len(bins[q])}" for q in range(4)])
    ax.set_ylabel("Test macro Dice")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    _save(fig, "fig8_size")


def detection_scores(cache) -> None:
    """Pooled POD, FAR and CSI on the positive test scans, mean ± SD over seeds."""
    for fam in ("attn", "unet"):
        pod, far, csi = [], [], []
        for seed in SEEDS:
            rows = _pos(cache[fam][seed], "test")
            tp = sum(r["tp"] for r in rows)
            fp = sum(r["fp"] for r in rows)
            fn = sum(r["fn"] for r in rows)
            pod.append(tp / (tp + fn))
            far.append(fp / (tp + fp))
            csi.append(tp / (tp + fp + fn))
        print(f"detection {fam} POD {np.mean(pod):.3f} ± {np.std(pod, ddof=1):.3f} "
              f"FAR {np.mean(far):.3f} ± {np.std(far, ddof=1):.3f} "
              f"CSI {np.mean(csi):.3f} ± {np.std(csi, ddof=1):.3f}")


EXTRAS = (
    ("S1 night-balanced", "night_split", "night_bal_attunet9_cur"),
    ("S2 + presence head", "night_split", "night_mtl_attunet9_cur"),
    ("S3a + temporal stack", "night_split", "night_tstack_attunet9_cur"),
    ("S3b + L-TAE", "night_split", "night_ltae_attunet9_cur"),
    ("Gated Attention U-Net", "night_cascade", "gated_attn_unet_cur"),
    ("TransUNet", "night_gated", "transunet_night"),
    ("Gated Swin-Attention", "night_gated", "gated_swin_attn_night"),
    ("Swin-UNet", "night_gated", "swin_unet_night"),
)


def extras() -> None:
    """Frozen single-seed runs on test, paired against Attention U-Net seed 42."""
    def run(group, name):
        exp = ROOT / "outputs" / group / "experiments"
        final = json.loads((exp / f"{name}_final_result.json").read_text(encoding="utf-8"))
        per = json.loads((exp / f"{name}_test_per_scene.json").read_text(encoding="utf-8"))
        return final, {int(r["ts"]): r for r in per if int(r["label"]) == 1}

    s0, s0_pos = run("night_split", "night_base_attunet9_s42")
    lines = ["run,name,threshold,val_dice,test_dice,d_dice,d_dice_lo,d_dice_hi,"
             "test_nsd,d_nsd,d_nsd_lo,d_nsd_hi,test_far_scan"]
    t0 = s0["test_full_scene"]
    lines.append(f"S0 Attention U-Net s42,night_base_attunet9_s42,{s0['calibrated_threshold']},"
                 f"{s0['val_full_scene']['dice']:.6f},{t0['dice']:.6f},,,,{t0['nsd']:.6f},,,,"
                 f"{t0['far_scan']:.6f}")
    for label, group, name in EXTRAS:
        final, pos = run(group, name)
        common = sorted(set(pos) & set(s0_pos))
        nights = [pos[ts]["night"] for ts in common]
        d = {k: paired_cluster_bootstrap([pos[ts][k] for ts in common],
                                         [s0_pos[ts][k] for ts in common], nights)
             for k in ("dice", "nsd")}
        t = final["test_full_scene"]
        lines.append(f"{label},{name},{final['calibrated_threshold']},{final['val_full_scene']['dice']:.6f},"
                     f"{t['dice']:.6f},{d['dice']['point']:.6f},{d['dice']['lo']:.6f},{d['dice']['hi']:.6f},"
                     f"{t['nsd']:.6f},{d['nsd']['point']:.6f},{d['nsd']['lo']:.6f},{d['nsd']['hi']:.6f},"
                     f"{t['far_scan']:.6f}")
        print("extras", lines[-1])
    (COMP / "extras_test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cache = {
        "attn": {s: _samples(f"night_base_attunet9_s{s}") for s in SEEDS},
        "unet": {s: _samples(f"unet_night_s{s}") for s in SEEDS},
    }
    fig_seeds(cache)
    fig_paired(cache)
    fig_robust(cache)
    fig_nsd(cache)
    fig_year(cache)
    fig_presence()
    fig_size(cache)
    detection_scores(cache)
    oracle(cache)
    extras()
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
