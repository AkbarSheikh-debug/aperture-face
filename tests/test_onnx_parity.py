"""Exported ONNX models must reproduce the PyTorch embeddings; INT8 must stay close to FP32."""
import numpy as np
import pytest
import torch

from common.data import eval_transform, lfw_split, load_rgb112
from common.io import CKPT, LFW_META
from common.onnx_runtime import OnnxEmbedder
from model import load_checkpoint

CASES = [("best_model.pth", "face_r50.onnx"), ("r18_student.pth", "face_r18.onnx")]


@pytest.fixture(scope="module")
def batch():
    if not LFW_META.exists():
        pytest.skip("dataset not prepared")
    tf = eval_transform()
    return torch.stack([tf(load_rgb112(p)) for p in lfw_split("test").path[:32]])


@pytest.mark.parametrize("ckpt,onnx", CASES)
def test_fp32_parity(batch, ckpt, onnx):
    if not (CKPT / onnx).exists():
        pytest.skip(f"{onnx} not exported")
    model, _ = load_checkpoint(CKPT / ckpt)
    with torch.no_grad():
        ref = model(batch).numpy()
    got = OnnxEmbedder(CKPT / onnx).run(batch.numpy())
    assert (ref * got).sum(1).min() > 0.9999


@pytest.mark.parametrize("onnx", ["face_r18", "face_r50"])
def test_int8_close_to_fp32(batch, onnx):
    if not (CKPT / f"{onnx}_int8.onnx").exists():
        pytest.skip("int8 model not built")
    a = OnnxEmbedder(CKPT / f"{onnx}.onnx").run(batch.numpy())
    b = OnnxEmbedder(CKPT / f"{onnx}_int8.onnx").run(batch.numpy())
    assert np.mean((a * b).sum(1)) > 0.97
