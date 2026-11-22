"""Web-service helpers: one-call analysis (detect + pose + embedding) and the About-tab numbers."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints" / "face_r18_int8.onnx"
YUNET = ROOT / "checkpoints" / "face_detection_yunet_2023mar.onnx"
RESULTS = ROOT / "results" / "evaluation_results.json"
PHOTO = ROOT.parent / "data" / "raw" / "lfw" / "Colin_Powell" / "Colin_Powell_0020.jpg"


def test_about_reads_real_numbers_from_results_file():
    if not RESULTS.exists():
        pytest.skip("results not generated")
    from api.engine import about
    a = about(RESULTS)
    assert 0.99 < a["teacher"]["AUC"] <= 1.0 and a["test_pairs"] == 10000
    assert a["served_model"]["size_MB"] < 30 and 0 < a["served_at_threshold"]["FAR"] < 0.01


@pytest.mark.skipif(not (MODEL.exists() and YUNET.exists() and PHOTO.exists()), reason="model or photo not available")
def test_analyze_returns_unit_embedding_pose_and_handles_no_face():
    import cv2
    from api.engine import analyze
    from common.align import FaceAligner
    from common.onnx_runtime import OnnxEmbedder
    al, emb = FaceAligner(YUNET), OnnxEmbedder(MODEL, "cpu", threads=2)
    r = analyze(al, emb, cv2.imread(str(PHOTO)))
    e = np.array(r["embedding"])
    assert r["face"] and r["n_faces"] == 1 and e.shape == (512,) and abs(np.linalg.norm(e) - 1) < 1e-3
    assert abs(r["yaw"]) < 0.2 and 0.45 < r["pitch"] < 0.7
    blank = analyze(al, emb, np.full((240, 320, 3), 120, np.uint8))
    assert blank["face"] is False and "embedding" not in blank
