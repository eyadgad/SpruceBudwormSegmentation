"""Generate a configuration-sweep experiments file for the top-3 models.

Factorial over: architecture x target x channel-set (auxiliary ablation) x loss.
Writes configs/experiments_sweep.yaml, which run.py / evaluate.py consume like any
experiments file. No training here — this only emits config.

Channel-set axis is the DEM / beam-height ablation requested by the user:
  noaux   = radar + valid_mask
  dem     = radar + valid_mask + DEM
  beam    = radar + valid_mask + beam-height (bh_e0)
  dembeam = radar + valid_mask + DEM + beam-height
"radar" = th_e0..th_e{radar_elevs-1} (default 3 low elevations; use 6 for the rich base).

Usage (from project root):
  .venv\\Scripts\\python scripts\\gen_sweep.py --pass1          # 12-run auxiliary ablation
  .venv\\Scripts\\python scripts\\gen_sweep.py --pass2          # target x loss on --chanset
  .venv\\Scripts\\python scripts\\gen_sweep.py --full           # full 144-run grid
  .venv\\Scripts\\python scripts\\gen_sweep.py --elev           # 12-run elevation sweep (1..6 elevations)
Then:  run.bat --experiments configs\\experiments_sweep.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ---- axis definitions -------------------------------------------------------

ARCHS = {
    "unetpp":  {"model": {"name": "smp_unetpp", "encoder": "efficientnet-b2", "pretrained": True},
                "extra_train": {"accum_steps": 2}, "best_loss": "focaltv"},
    "deeplab": {"model": {"name": "smp_deeplabv3p", "encoder": "resnet34", "pretrained": True},
                "extra_train": {}, "best_loss": "focal"},
    "attunet": {"model": {"name": "attention_unet", "base_filters": 32},
                "extra_train": {}, "best_loss": "focal"},
}

TARGETS = {
    "isf":  {"mode": "isfinite"},
    "dbz0": {"mode": "threshold", "dbz_threshold": 0.0},
    "dbz5": {"mode": "threshold", "dbz_threshold": 5.0},
}

# auxiliary-channel ablation (radar filled in per --radar-elevs)
CHANSETS = {
    "noaux":   lambda radar: radar + ["valid_mask"],
    "dem":     lambda radar: radar + ["dem", "valid_mask"],
    "beam":    lambda radar: radar + ["bh_e0", "valid_mask"],
    "dembeam": lambda radar: radar + ["dem", "bh_e0", "valid_mask"],
}

LOSSES = {
    "focal":   {"name": "focal", "alpha": 0.25, "gamma": 2.0},
    "focaltv": {"name": "focal_tversky", "alpha": 0.3, "beta": 0.7, "gamma": 1.333},
    "tversky": {"name": "tversky", "tversky_alpha": 0.7, "tversky_beta": 0.3},  # precision-biased
    "dicebce": {"name": "dice_bce", "pos_weight": 10.0},
    # extras enabled only with --all-losses
    "bce":   {"name": "bce", "pos_weight": 10.0},
    "dice":  {"name": "dice"},
    "diceb": {"name": "dice_boundary", "pos_weight": 10.0, "boundary_weight": 0.5},
}
CORE_LOSSES = ["focal", "focaltv", "tversky", "dicebce"]
ALL_LOSSES = CORE_LOSSES + ["bce", "dice", "diceb"]
PASS2_LOSSES = ["focal", "focaltv"]


def radar_channels(elevs: int) -> list:
    return [f"th_e{i}" for i in range(elevs)]


# elevation-count sweep: 1..N low TH elevations, on the two selected architectures
ELEV_ARCHS = ["unetpp", "attunet"]


def _exp(arch: str, name: str, channels: list, target: str, loss: str, args) -> dict:
    a = ARCHS[arch]
    train = {"epochs": args.epochs, **a["extra_train"]}
    if args.workers is not None:
        train["num_workers"] = args.workers
    return {
        "name": name,
        "channels": channels,
        "model": dict(a["model"]),
        "loss": dict(LOSSES[loss]),
        "target": dict(TARGETS[target]),
        "train": train,
        "patch": {"patches_per_image": args.patches},
        "eval": {"tta": bool(args.tta)},
    }


def make_experiment(arch: str, target: str, chanset: str, loss: str, args) -> dict:
    return _exp(arch, f"sweep_{arch}_{target}_{chanset}_{loss}",
                CHANSETS[chanset](radar_channels(args.radar_elevs)), target, loss, args)


def build_elev(args) -> list:
    # radar (th_e0..th_e{k-1}) + valid_mask, no terrain (Experiment 2 showed it
    # does not help UNet++); fixed dBZ>=0 target and Focal Tversky loss.
    # k = 1..elev_max elevations. Already-run cells are skipped by run.py.
    exps = []
    for arch in ELEV_ARCHS:
        for k in range(1, args.elev_max + 1):
            es = list(range(k))
            channels = [f"th_e{i}" for i in es] + ["valid_mask"]
            tag = "".join(str(i) for i in es)   # single-digit indices -> unambiguous up to 10
            exps.append(_exp(arch, f"sweep_{arch}_dbz0_e{tag}_focaltv", channels, "dbz0", "focaltv", args))
    return exps


def build(args) -> list:
    exps = []
    if args.mode == "elev":
        return build_elev(args)
    if args.mode == "pass1":
        # auxiliary ablation: 3 arch x 4 channel sets, fixed dbz0 target + arch best loss
        for arch, a in ARCHS.items():
            for cs in CHANSETS:
                exps.append(make_experiment(arch, "dbz0", cs, a["best_loss"], args))
    elif args.mode == "pass2":
        # target x loss on a chosen channel set
        for arch in ARCHS:
            for target in TARGETS:
                for loss in PASS2_LOSSES:
                    exps.append(make_experiment(arch, target, args.chanset, loss, args))
    else:  # full
        losses = ALL_LOSSES if args.all_losses else CORE_LOSSES
        for arch in ARCHS:
            for target in TARGETS:
                for cs in CHANSETS:
                    for loss in losses:
                        exps.append(make_experiment(arch, target, cs, loss, args))
    return exps


def main():
    ap = argparse.ArgumentParser(description="Generate the configuration-sweep experiments file")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pass1", dest="mode", action="store_const", const="pass1", help="12-run auxiliary ablation (default)")
    g.add_argument("--pass2", dest="mode", action="store_const", const="pass2", help="target x loss on --chanset")
    g.add_argument("--full", dest="mode", action="store_const", const="full", help="full factorial grid")
    g.add_argument("--elev", dest="mode", action="store_const", const="elev",
                   help="elevation sweep: UNet++ and Attention UNet, dBZ>=0, Focal Tversky, 1..6 TH elevations")
    ap.set_defaults(mode="pass1")
    ap.add_argument("--radar-elevs", type=int, default=3, choices=[3, 6], help="# TH elevations in 'radar' base")
    ap.add_argument("--chanset", default="dembeam", choices=list(CHANSETS), help="channel set for pass2")
    ap.add_argument("--elev-max", type=int, default=6, choices=list(range(1, 11)),
                    help="max elevations for --elev sweep, 1..N (max 10; higher tilts are sparse)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patches", type=int, default=8, help="patches_per_image (8 halves epoch time vs 16)")
    ap.add_argument("--tta", action="store_true", help="enable TTA during the sweep (default off for speed)")
    ap.add_argument("--workers", type=int, default=None, help="override train.num_workers (default: keep base)")
    ap.add_argument("--all-losses", action="store_true", help="full grid uses all 7 losses instead of 4")
    ap.add_argument("--out", default="configs/experiments_sweep.yaml")
    args = ap.parse_args()

    exps = build(args)
    names = [e["name"] for e in exps]
    assert len(names) == len(set(names)), "duplicate experiment names generated"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Auto-generated by scripts/gen_sweep.py (mode={args.mode}, radar_elevs={args.radar_elevs}, "
              f"epochs={args.epochs}, patches={args.patches}, tta={bool(args.tta)}).\n"
              f"# {len(exps)} experiments. Each is deep-merged onto configs/base_config.yaml.\n"
              f"# Run: run.bat --experiments {args.out.replace('/', chr(92))}\n\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump({"experiments": exps}, f, sort_keys=False, default_flow_style=False, width=200)

    print(f"wrote {out} : {len(exps)} experiments (mode={args.mode}, radar_elevs={args.radar_elevs})")
    # brief breakdown
    from collections import Counter
    print("  archs:", dict(Counter(n.split('_')[1] for n in names)))
    print("  first few:", names[:4])


if __name__ == "__main__":
    main()
