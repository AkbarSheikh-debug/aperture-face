"""Project paths and small I/O helpers shared by every script."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]                      # submission/
DATA = Path(os.environ.get("FACE_DATA", ROOT.parent / "data"))  # data/ (git-ignored)
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LFW_ALIGNED = PROCESSED / "lfw_aligned"        # dataset/person_001/img_01.jpg layout
LFW_META = PROCESSED / "lfw_meta.csv"          # one row per aligned image
DIGIFACE_META = PROCESSED / "digiface_meta.csv"
SPLITS_JSON = PROCESSED / "splits.json"
YUNET = RAW / "face_detection_yunet_2023mar.onnx"
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"
PAIRS_DIR = RESULTS / "pairs"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_np_default)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_json(path, key, value):
    """Read-modify-write one top-level key, so several scripts can share one results file."""
    obj = load_json(path, default={}) or {}
    obj[key] = value
    save_json(obj, path)


def _np_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")
