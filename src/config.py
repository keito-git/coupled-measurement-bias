"""
Paths and process settings shared by all scripts.

DATA_ROOT     directory with the raw corpora (default: <repo>/data)
RESULTS_ROOT  directory with one sub-folder per experiment (default: <repo>/experiments)

Expected data layout:
    DATA_ROOT/raw/emotionlines/Friends/friends.json
    DATA_ROOT/raw/emotionlines/EmotionPush/emotionpush.json
    DATA_ROOT/raw/m3ed/annotation.json
    DATA_ROOT/processed/m3ed_softlabels.parquet   (written by src/prepare_m3ed.py)
"""
import os
from pathlib import Path

# Single-threaded numerical libraries; parallelism is over replications (one process each).
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "experiments")).resolve()

EL_RAW = DATA_ROOT / "raw" / "emotionlines"
FRIENDS_JSON = EL_RAW / "Friends" / "friends.json"
EMOTIONPUSH_JSON = EL_RAW / "EmotionPush" / "emotionpush.json"
M3ED_ANNOTATION = DATA_ROOT / "raw" / "m3ed" / "annotation.json"
M3ED_PARQUET = DATA_ROOT / "processed" / "m3ed_softlabels.parquet"


def results_dir(exp_id: str) -> Path:
    """Output folder of one experiment (created if missing)."""
    d = RESULTS_ROOT / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def n_workers(default: int) -> int:
    """Number of worker processes; override with the N_WORKERS environment variable."""
    return int(os.environ.get("N_WORKERS", default))
