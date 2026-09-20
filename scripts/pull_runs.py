"""Download finished runs from the HF mirror WITHOUT training anything.

Use this to merge work done on another machine. ``src.run`` / ``src.classify``
also pull, but they fall through to training if the pull comes back incomplete;
this script never launches a run, so it is the safe way to collect results.

    .venv\\Scripts\\python.exe scripts\\pull_runs.py ^
        --base-config configs\\base_config_night.yaml ^
        --experiments configs\\experiments_night_seeds_b.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import checkpoint as ckpt, config as cfgmod, paths, sync  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--artifacts", action="store_true",
                    help="also pull artifacts_night/ (manifest, norm stats)")
    args = ap.parse_args()

    base = cfgmod.load_base_config(args.base_config)
    if args.artifacts:
        sync.pull_artifacts(base)
        sync.check_manifest(base, strict=False)

    probe = sync.RunSync(base, "_probe", ROOT, ROOT)
    if not probe.enabled:
        raise SystemExit("[pull] no HF token — set HF_TOKEN (or .env). Nothing to do.")

    done, missing = [], []
    for exp in cfgmod.load_experiments(args.experiments):
        cfg = cfgmod.resolve_experiment(base, exp)
        name = exp["name"]
        exp_dir = paths.experiments_dir(cfg)
        ckpt_dir = paths.checkpoint_dir(cfg)
        before = ckpt.is_done(exp_dir, name)
        got = 0 if before else sync.pull_run(cfg, name, ckpt_dir, exp_dir)
        after = ckpt.is_done(exp_dir, name)
        state = "already local" if before else (
            f"pulled {got} file(s)" if after else "NOT ON MIRROR")
        print(f"  {name:28s} {state}")
        (done if after else missing).append(name)

    print(f"\n{len(done)} run(s) available locally; {len(missing)} missing.")
    if missing:
        print("missing: " + ", ".join(missing))
        print("Those runs have not finished on the other machine yet "
              "(a run is only mirrored as complete once its result JSON exists).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
