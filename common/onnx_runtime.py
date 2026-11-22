"""Torch-free inference (ONNX Runtime). Used by the edge benchmark, the FastAPI service and the webcam demo, so
the deployed footprint is numpy + opencv + onnxruntime only."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_bgr(crops_bgr):
    """List of 112x112 BGR uint8 crops -> float32 NCHW in [-1, 1] (identical to the training normalisation)."""
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]).astype(np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class OnnxEmbedder:
    def __init__(self, model_path, device="cpu", threads=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                        "trt_engine_cache_path": str(Path(model_path).parent / "trt_cache")}),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.input = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()

    def run(self, x: np.ndarray) -> np.ndarray:
        e = self.sess.run(None, {self.input: x})[0].astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed_crops(self, crops_bgr, tta=False):
        x = preprocess_bgr(crops_bgr)
        e = self.run(x)
        if tta:
            e = e + self.run(np.ascontiguousarray(x[..., ::-1]))
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return e
