"""Post-training static INT8 quantisation (ONNX Runtime, QDQ format, per-channel INT8 weights, UINT8 activations).

Calibration uses 512 TRAINING images only (LFW train identities + DigiFace) - never val/test.

Usage: python -m edge.quantize_int8 [--model face_r18.onnx] [--method percentile]
"""
import argparse

import numpy as np
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common.data import lfw_split
from common.io import CKPT, DATA, DIGIFACE_META
from common.onnx_runtime import preprocess_bgr

import cv2


class FaceCalibReader(CalibrationDataReader):
    def __init__(self, paths, input_name="input", batch=16):
        self.paths, self.input_name, self.batch, self.i = paths, input_name, batch, 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        crops = [cv2.resize(cv2.imread(str(DATA / p)), (112, 112)) for p in self.paths[self.i:self.i + self.batch]]
        self.i += self.batch
        return {self.input_name: preprocess_bgr(crops)}

    def rewind(self):
        self.i = 0


def calibration_paths(n=512, seed=0):
    rng = np.random.default_rng(seed)
    train = lfw_split("train").path.to_numpy()
    paths = list(rng.choice(train, n // 2, replace=False))
    if DIGIFACE_META.exists():
        dg = pd.read_csv(DIGIFACE_META).path.to_numpy()
        paths += list(rng.choice(dg, n - len(paths), replace=False))
    else:
        paths += list(rng.choice(train, n - len(paths), replace=False))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=["face_r18.onnx", "face_r50.onnx"])
    ap.add_argument("--method", default="percentile", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--activations", default="u8", choices=["u8", "s8"],
                    help="u8 = U8S8 (fast on x86 AVX2/AVX-512/VNNI: 3.2x faster than FP32 on a Ryzen 9950X, vs 1.3x "
                         "for S8S8); s8 = S8S8 (preferred by some ARM / NPU back-ends)")
    args = ap.parse_args()
    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[args.method]
    for name in args.model:
        src = CKPT / name
        if not src.exists():
            print(f"[int8] skip {name}")
            continue
        prep = CKPT / name.replace(".onnx", "_prep.onnx")
        dst = CKPT / name.replace(".onnx", "_int8.onnx")
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
        quantize_static(str(prep), str(dst), FaceCalibReader(calibration_paths()), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8 if args.activations == "u8" else QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=method,
                        extra_options={"CalibPercentile": 99.99} if args.method == "percentile" else {})
        prep.unlink(missing_ok=True)
        print(f"[int8] {name} -> {dst.name}: {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
