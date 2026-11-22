"""Shared inference helpers for the web page and both servers (stateful Docker API, stateless Vercel API).

`analyze()` is the one call the web page needs per camera frame: face detection, head-pose cues and the
512-D embedding in a single round trip. Nothing is stored here.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from common.align import head_pose

ABOUT_KEYS = ("pairs", "verification", "identification", "edge", "edge_threshold")


def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def analyze(aligner, embedder, img, want_embedding: bool = True) -> dict:
    """Detect -> pose -> (optionally) embed the largest face. Coordinates are fractions of the frame."""
    h, w = img.shape[:2]
    faces = aligner.detect(img)
    if len(faces) == 0:
        return {"face": False, "n_faces": 0, "brightness": float(img.mean())}
    f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
    out = {"face": True, "n_faces": int(len(faces)), "score": round(float(f[14]), 3),
           "box": [float(f[0] / w), float(f[1] / h), float(f[2] / w), float(f[3] / h)],
           "brightness": float(img.mean())}
    out.update({k: round(v, 4) for k, v in head_pose(f[4:14].reshape(5, 2)).items()})
    if want_embedding and len(faces) == 1:
        crop, _ = aligner.align(img, pick="largest")
        out["embedding"] = [round(float(x), 5) for x in embedder.embed_crops([crop], tta=True)[0]]
    return out


def default_threshold(results_json: Path, fallback: float = 0.35) -> float:
    if not results_json.exists():
        return fallback
    r = json.loads(results_json.read_text())
    return float((r.get("edge_threshold") or r["verification"]["final_threshold"])["threshold"])


def about(results_json: Path) -> dict:
    """The handful of real evaluation numbers the About tab shows (read from the results file, never typed)."""
    if not results_json.exists():
        return {}
    r = json.loads(results_json.read_text())
    v, ident = r["verification"]["threshold_free_test"], r["identification"]
    edge = {(row["model"], row["runtime"], row["precision"], row.get("threads")): row for row in r.get("edge", {}).get("rows", [])}
    r18 = edge.get(("R18", "ONNX Runtime", "INT8", 1))
    et = r.get("edge_threshold", {})
    return {
        "test_identities": r["pairs"]["test"]["identities"],
        "test_pairs": r["pairs"]["test"]["positives"] + r["pairs"]["test"]["negatives"],
        "teacher": {"AUC": v["AUC"], "EER": v["EER"], "TAR@FAR=1%": v["TAR@FAR=1%"], "TAR@FAR=0.1%": v["TAR@FAR=0.1%"],
                    "rank1_single": ident["single_shot"]["rank1"], "rank1_multi": ident["multi_shot_3"]["rank1"],
                    "rank5_single": ident["single_shot"]["rank5"]},
        "served_model": ({"name": "ResNet18 student, INT8", "size_MB": r18["size_MB"], "p50_ms_1thread": r18["p50_ms"],
                          "AUC": r18["test_AUC"], "TAR@FAR=1%": r18["TAR@FAR=1%"]} if r18 else None),
        "served_at_threshold": ({"threshold": et["threshold"], "TAR": et["test_at_threshold"]["TPR(TAR)"],
                                 "FAR": et["test_at_threshold"]["FPR(FAR)"]} if et.get("test_at_threshold") else None),
    }
