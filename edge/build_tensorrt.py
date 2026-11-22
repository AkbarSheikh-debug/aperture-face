"""Build a TensorRT FP16 engine from the ONNX model and measure latency + accuracy with it.
(The RTX GPU stands in for an NVIDIA Jetson: the same ONNX -> TensorRT path is used on Jetson Orin/Xavier.)

Skipped gracefully when the `tensorrt` Python package is not installed (see requirements-edge.txt).

Usage: python -m edge.build_tensorrt [--onnx face_r18.onnx]
"""
import argparse
import time

import numpy as np

from common.io import CKPT, RESULTS, save_json


def to_fp16_onnx(onnx_path):
    """TensorRT >= 11 only builds strongly-typed networks: precision comes from the ONNX graph itself, so the FP32
    graph is converted to FP16 first (input/output kept FP32, normalisation ops kept FP32 for stability)."""
    import onnx
    from onnxconverter_common import float16

    out = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    m = float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True,
                                         op_block_list=["ReduceL2", "Div", "Sqrt", "ReduceSum"])
    onnx.save(m, str(out))
    return out


def build_engine(trt, onnx_path, engine_path, max_batch=64):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    legacy_fp16 = hasattr(trt.BuilderFlag, "FP16")          # TensorRT <= 10 (e.g. JetPack 6)
    if legacy_fp16:
        network = builder.create_network(0)
        src = onnx_path
    else:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        src = to_fp16_onnx(onnx_path)
    parser = trt.OnnxParser(network, logger)
    with open(src, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if legacy_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 112, 112), (1, 3, 112, 112), (max_batch, 3, 112, 112))
    config.add_optimization_profile(profile)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    engine_path.write_bytes(bytes(serialized))
    print(f"[trt] built {engine_path.name} in {time.time() - t0:.0f}s ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return trt.Runtime(logger).deserialize_cuda_engine(serialized)


class TrtEmbedder:
    def __init__(self, trt, engine):
        import torch

        self.torch, self.engine = torch, engine
        self.ctx = engine.create_execution_context()
        to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        # I/O precision is whatever the (strongly-typed) engine declares - read it instead of assuming FP32
        self.in_dtype = to_torch[engine.get_tensor_dtype("input")]
        self.out_dtype = to_torch[engine.get_tensor_dtype("embedding")]
        self.stream = torch.cuda.Stream()                     # non-default stream (TensorRT recommendation)

    def run(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.cuda.stream(self.stream):                  # copy + inference ordered on the same stream
            inp = torch.from_numpy(x).to("cuda", non_blocking=False).to(self.in_dtype).contiguous()
            out = torch.empty((x.shape[0], 512), dtype=self.out_dtype, device="cuda")
            self.ctx.set_input_shape("input", tuple(inp.shape))
            self.ctx.set_tensor_address("input", inp.data_ptr())
            self.ctx.set_tensor_address("embedding", out.data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        e = out.float().cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="*", default=["face_r18.onnx", "face_r50.onnx"])
    args = ap.parse_args()
    try:
        import tensorrt as trt
    except ImportError:
        print("[trt] tensorrt not installed - skipping (pip install -r requirements-edge.txt)")
        save_json({"skipped": "tensorrt not installed"}, RESULTS / "tensorrt.json")
        return
    from edge.benchmark_edge import accuracy_on_test, test_batches

    rows = []
    for name in args.onnx:
        onnx_path = CKPT / name
        if not onnx_path.exists():
            continue
        engine = build_engine(trt, onnx_path, CKPT / name.replace(".onnx", "_fp16.engine"))
        emb = TrtEmbedder(trt, engine)
        x1 = np.random.randn(1, 3, 112, 112).astype(np.float32)
        for _ in range(50):
            emb.run(x1)
        ts = []
        for _ in range(500):
            t = time.perf_counter(); emb.run(x1); ts.append((time.perf_counter() - t) * 1e3)
        acc, _ = accuracy_on_test(lambda x: emb.run(x), test_batches())
        rows.append({"model": name.replace(".onnx", ""), "runtime": f"TensorRT {trt.__version__}", "precision": "FP16",
                     "device": "GPU", "threads": None,
                     "size_MB": round((CKPT / name.replace(".onnx", "_fp16.engine")).stat().st_size / 1e6, 1),
                     "p50_ms": round(float(np.percentile(ts, 50)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
                     **acc})
        print(f"[trt] {rows[-1]}")
    save_json({"rows": rows}, RESULTS / "tensorrt.json")


if __name__ == "__main__":
    main()
