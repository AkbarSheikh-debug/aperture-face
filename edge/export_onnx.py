"""Export PyTorch checkpoints to ONNX (opset 17, dynamic batch) and verify numerical parity on real test faces.

Usage: python -m edge.export_onnx
"""
import argparse
import shutil

import numpy as np
import torch

from common.data import lfw_split, load_rgb112, eval_transform
from common.io import CKPT, RESULTS, YUNET, update_json
from common.onnx_runtime import OnnxEmbedder
from model import load_checkpoint

MODELS = {"best_model.pth": "face_r50.onnx", "r18_student.pth": "face_r18.onnx"}


def export(ckpt_path, onnx_path):
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
                      do_constant_folding=True, dynamo=False)
    return model


def parity(model, onnx_path, n=64):
    paths = lfw_split("test").path.tolist()[:n]
    tf = eval_transform()
    x = torch.stack([tf(load_rgb112(p)) for p in paths])
    with torch.no_grad():
        ref = model(x).numpy()
    got = OnnxEmbedder(onnx_path).run(x.numpy())
    cos = (ref * got).sum(1)
    return {"min_cosine_vs_pytorch": float(cos.min()), "max_abs_diff": float(np.abs(ref - got).max()), "n_images": n}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    report = {}
    for src, dst in MODELS.items():
        if not (CKPT / src).exists():
            print(f"[onnx] skip {src} (missing)")
            continue
        model = export(CKPT / src, CKPT / dst)
        p = parity(model, CKPT / dst)
        assert p["min_cosine_vs_pytorch"] > 0.9999, p
        report[dst] = p
        print(f"[onnx] {src} -> {dst}  parity {p}  size {(CKPT / dst).stat().st_size / 1e6:.1f} MB")
    update_json(RESULTS / "evaluation_results.json", "onnx_export", report)
    if YUNET.exists():                       # ship the detector next to the embedder for the API / demo / Docker
        shutil.copy(YUNET, CKPT / YUNET.name)


if __name__ == "__main__":
    main()
