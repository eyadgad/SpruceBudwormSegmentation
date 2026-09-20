"""Mirror run artifacts to a Hugging Face dataset so a run survives its machine.

Code lives in git; weights, logs, results and the frozen manifest live here.
Together they make a run portable: clone the repo, provide ``Data/``, export
``HF_TOKEN``, and re-running the same command downloads whatever already exists
and continues from the last snapshot.

Nothing in this module may abort training. Every network call is wrapped: a
failed sync logs a warning and the run carries on against local disk.

Token resolution, in order:
  1. ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` in the environment
  2. a gitignored ``.env`` file in the repo root (``HF_TOKEN=hf_...``)
  3. a cached ``huggingface-cli login``
If none resolve, syncing is disabled and training runs purely locally.

NEVER hardcode a token here. This file is public on GitHub; a committed write
token lets anyone overwrite the checkpoint dataset, and the platform's secret
scanner revokes it -- usually mid-run, on an unattended multi-day job.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
DEFAULT_REPO = "eyadgad/budworm-checkpoints"
DEFAULT_EVERY = 50
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

log = logging.getLogger("sync")


def _read_env_file() -> dict:
    if not ENV_FILE.exists():
        return {}
    out = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        return {}
    return out


def load_token() -> Optional[str]:
    for var in TOKEN_VARS:
        val = os.environ.get(var)
        if val:
            return val.strip()
    env = _read_env_file()
    for var in TOKEN_VARS:
        if env.get(var):
            return env[var].strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def sync_cfg(cfg: dict) -> dict:
    s = dict((cfg or {}).get("sync") or {})
    s.setdefault("enabled", True)
    s.setdefault("repo_id", os.environ.get("BUDWORM_HF_REPO", DEFAULT_REPO))
    s.setdefault("every_epochs", DEFAULT_EVERY)
    s.setdefault("push_resume", True)
    return s


class RunSync:
    """Per-experiment view of the remote mirror. Safe to construct always."""

    def __init__(self, cfg: dict, name: str, ckpt_dir: Path, exp_dir: Path,
                 logger: Optional[logging.Logger] = None):
        self.cfg = sync_cfg(cfg)
        self.name = name
        self.ckpt_dir = Path(ckpt_dir)
        self.exp_dir = Path(exp_dir)
        self.log = logger or log
        self.repo_id = str(self.cfg["repo_id"])
        self.every = max(1, int(self.cfg.get("every_epochs", DEFAULT_EVERY)))
        self._api = None
        self._token = None
        self._ready = False
        self._warned = False
        if self.cfg.get("enabled", True):
            self._token = load_token()
            if not self._token:
                self.log.info(
                    "sync: no HF token found (set HF_TOKEN or add it to .env); "
                    "running local-only")

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True)) and bool(self._token)

    def _connect(self) -> bool:
        if self._ready:
            return True
        if not self.enabled:
            return False
        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self._token)
            self._api.create_repo(self.repo_id, repo_type="dataset",
                                  exist_ok=True, private=True)
            self._ready = True
        except Exception as exc:  # noqa: BLE001 - never abort training
            if not self._warned:
                self.log.warning(f"sync: cannot reach {self.repo_id} ({exc}); local-only")
                self._warned = True
            return False
        return True

    # -- remote paths ------------------------------------------------------
    def _remote(self, filename: str) -> str:
        return f"runs/{self.name}/{filename}"

    def _local_files(self, include_resume: bool = True) -> List[Path]:
        names = [
            self.ckpt_dir / f"{self.name}_best.pt",
            self.ckpt_dir / f"{self.name}_final.pt",
            self.exp_dir / f"{self.name}_result.json",
            self.exp_dir / f"{self.name}_config.json",
            self.exp_dir / f"{self.name}_history.csv",
            self.exp_dir / f"{self.name}_train.log",
        ]
        if include_resume and bool(self.cfg.get("push_resume", True)):
            names.insert(0, self.ckpt_dir / f"{self.name}_resume.pt")
        return [p for p in names if p.exists()]

    # -- push --------------------------------------------------------------
    def push(self, tag: str = "", include_resume: bool = True) -> int:
        """Upload whatever this run has produced. Returns files sent."""
        if not self._connect():
            return 0
        sent = 0
        for path in self._local_files(include_resume=include_resume):
            try:
                self._api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=self._remote(path.name),
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"{self.name}: {tag or 'sync'} ({path.name})",
                )
                sent += 1
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"sync: upload failed for {path.name} ({exc})")
        if sent:
            self.log.info(f"sync: pushed {sent} file(s) to {self.repo_id} [{tag or 'sync'}]")
        return sent

    def maybe_push(self, epoch_1based: int, total_epochs: int) -> bool:
        """Push on the configured cadence and on the final epoch."""
        if not self.enabled:
            return False
        if epoch_1based % self.every and epoch_1based != total_epochs:
            return False
        self.push(tag=f"epoch {epoch_1based}/{total_epochs}")
        return True

    # -- pull --------------------------------------------------------------
    def pull(self, overwrite: bool = False) -> int:
        """Download this run's artifacts that are missing locally."""
        if not self._connect():
            return 0
        try:
            remote = [f for f in self._api.list_repo_files(
                self.repo_id, repo_type="dataset")
                if f.startswith(f"runs/{self.name}/")]
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"sync: cannot list {self.repo_id} ({exc})")
            return 0
        if not remote:
            return 0
        got = 0
        for rpath in remote:
            fname = rpath.rsplit("/", 1)[-1]
            dest_dir = self.ckpt_dir if fname.endswith(".pt") else self.exp_dir
            dest = dest_dir / fname
            if dest.exists() and not overwrite:
                continue
            try:
                from huggingface_hub import hf_hub_download
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = hf_hub_download(repo_id=self.repo_id, filename=rpath,
                                      repo_type="dataset", token=self._token)
                import shutil
                shutil.copyfile(tmp, dest)
                got += 1
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"sync: download failed for {rpath} ({exc})")
        if got:
            self.log.info(f"sync: pulled {got} file(s) for {self.name} from {self.repo_id}")
        return got


# -- dataset-level artifacts (manifest / norm stats / split summary) --------

def _artifact_files(cfg: dict) -> List[Path]:
    adir = ROOT / str((cfg.get("data") or {}).get("artifacts_dir", "artifacts_night"))
    return [adir / n for n in ("manifest.csv", "norm_stats.json", "split_summary.json",
                               "manifest.sha256")]


def push_artifacts(cfg: dict, logger: Optional[logging.Logger] = None) -> int:
    """Mirror the frozen split so another machine cannot silently rebuild it.

    Regenerating the manifest is NOT equivalent: negatives are sampled before
    the night->split assignment, so a single missing .nc file reshuffles which
    nights land in val/test. That is exactly how the superseded-era split was
    created. Always pull the manifest rather than re-running data_prep.
    """
    lg = logger or log
    s = RunSync(cfg, "_artifacts", ROOT, ROOT, logger=lg)
    if not s._connect():
        return 0
    adir = str((cfg.get("data") or {}).get("artifacts_dir", "artifacts_night"))
    write_fingerprint(cfg)
    sent = 0
    for path in _artifact_files(cfg):
        if not path.exists():
            continue
        try:
            s._api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"artifacts/{adir}/{path.name}",
                repo_id=s.repo_id, repo_type="dataset",
                commit_message=f"artifacts: {adir}/{path.name}")
            sent += 1
        except Exception as exc:  # noqa: BLE001
            lg.warning(f"sync: artifact upload failed for {path.name} ({exc})")
    if sent:
        lg.info(f"sync: pushed {sent} artifact file(s)")
    return sent


def pull_artifacts(cfg: dict, logger: Optional[logging.Logger] = None,
                   overwrite: bool = False) -> int:
    lg = logger or log
    s = RunSync(cfg, "_artifacts", ROOT, ROOT, logger=lg)
    if not s._connect():
        return 0
    adir = str((cfg.get("data") or {}).get("artifacts_dir", "artifacts_night"))
    got = 0
    for path in _artifact_files(cfg):
        if path.exists() and not overwrite:
            continue
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            tmp = hf_hub_download(repo_id=s.repo_id,
                                  filename=f"artifacts/{adir}/{path.name}",
                                  repo_type="dataset", token=s._token)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, path)
            got += 1
        except Exception:  # noqa: BLE001 - absent remotely is normal
            continue
    if got:
        lg.info(f"sync: pulled {got} artifact file(s) from {s.repo_id}")
    return got


def pull_run(cfg: dict, name: str, ckpt_dir: Path, exp_dir: Path,
             logger: Optional[logging.Logger] = None) -> int:
    return RunSync(cfg, name, ckpt_dir, exp_dir, logger=logger).pull()


def manifest_fingerprint(cfg: dict) -> Optional[str]:
    """SHA-256 of the manifest, for FROZEN.md and cross-machine verification."""
    import hashlib
    adir = ROOT / str((cfg.get("data") or {}).get("artifacts_dir", "artifacts_night"))
    path = adir / "manifest.csv"
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_path(cfg: dict) -> Path:
    adir = ROOT / str((cfg.get("data") or {}).get("artifacts_dir", "artifacts_night"))
    return adir / "manifest.sha256"


def write_fingerprint(cfg: dict) -> Optional[str]:
    fp = manifest_fingerprint(cfg)
    if fp:
        path = _fingerprint_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fp + "\n", encoding="utf-8")
    return fp


def check_manifest(cfg: dict, logger: Optional[logging.Logger] = None,
                   strict: bool = True) -> bool:
    """Verify this machine's manifest matches the mirrored one.

    Two machines training halves of one experiment MUST share a split. The
    split is not reproducible from the config alone: negatives are sampled
    before the night->split assignment, so any difference in ``Data/`` moves
    nights between train/val/test. Comparing the recorded SHA-256 turns a silent
    divergence into an immediate stop.
    """
    lg = logger or log
    local = manifest_fingerprint(cfg)
    if local is None:
        msg = ("manifest missing: this machine has no frozen split. Pull it "
               "(src.sync.pull_artifacts) before training; do NOT run data_prep, "
               "which would build a different split.")
        if strict:
            raise SystemExit(f"[sync] {msg}")
        lg.warning(f"sync: {msg}")
        return False
    recorded = _fingerprint_path(cfg)
    if not recorded.exists():
        lg.info(f"sync: manifest sha256={local[:16]}… (no mirrored fingerprint to "
                f"compare against yet)")
        return True
    want = recorded.read_text(encoding="utf-8").strip()
    if want and want != local:
        msg = (f"MANIFEST MISMATCH — this machine's split differs from the mirror.\n"
               f"         local  {local}\n"
               f"         mirror {want}\n"
               f"  Results from the two machines are NOT comparable. Delete the local "
               f"artifacts dir and re-pull rather than training against this split.")
        if strict:
            raise SystemExit(f"[sync] {msg}")
        lg.warning(f"sync: {msg}")
        return False
    lg.info(f"sync: manifest sha256={local[:16]}… matches the mirror")
    return True
