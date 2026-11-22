"""Edge benchmark: model size, compute, latency and accuracy for every deployable variant.

* accuracy  : the same 5k/5k TEST pairs, single forward pass (no TTA - what an edge device would run)
* latency   : batch 1, 50 warm-up + 300 timed runs; CPU limited to 1 and 4 threads as a stand-in for
              Raspberry-Pi-5 / Jetson-CPU class devices (absolute numbers on a desktop CPU are faster)
* pipeline  : YuNet detection + alignment + embedding on raw LFW photos, CPU 4 threads
* parity    : mean cosine between a variant's embeddings and its FP32 PyTorch counterpart

Usage: python -m edge.benchmark_edge
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.align import FaceAligner
from common.data import lfw_split, load_rgb112, eval_transform
from common.io import CKPT, DATA, PAIRS_DIR, RAW, RESULTS, YUNET, load_json, update_json
from common.metrics import auc, tar_at_far
from common.onnx_runtime import OnnxEmbedder
from common.plotting import INK2, SERIES, apply_style
from model import load_checkpoint

_CACHE = {}


def test_batches(bs=64):
    if "batches" not in _CACHE:
        df = lfw_split("test")
        tf = eval_transform()
        x = torch.stack([tf(load_rgb112(p)) for p in df.path]).numpy()
        _CACHE["batches"] = ([x[i:i + bs] for i in range(0, len(x), bs)], df.path.tolist())
    return _CACHE["batches"]


def accuracy_on_test(embed_fn, batches_paths):
    batches, paths = batches_paths
    emb = np.concatenate([embed_fn(b) for b in batches])
    idx = {p: i for i, p in enumerate(paths)}
    pairs = pd.read_csv(PAIRS_DIR / "pairs_test.csv")
    s = (emb[pairs.img1.map(idx)] * emb[pairs.img2.map(idx)]).sum(1)
    y = pairs.label.to_numpy()
    _CACHE.setdefault("embs", {})
    return {"test_AUC": round(auc(y, s), 5), "TAR@FAR=1%": round(tar_at_far(y, s, 1e-2)[0], 4),
            "TAR@FAR=0.1%": round(tar_at_far(y, s, 1e-3)[0], 4)}, emb


def latency(fn, x, warm=50, n=300):
    for _ in range(warm):
        fn(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1e3)
    return round(float(np.percentile(ts, 50)), 2), round(float(np.percentile(ts, 95)), 2)


def count_gmacs(model):
    macs = []

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            macs.append(o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, torch.nn.Linear):
            macs.append(m.in_features * m.out_features)

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model.cpu().eval()(torch.zeros(1, 3, 112, 112))
    for h in hs:
        h.remove()
    return sum(macs) / 1e9


def main():
    apply_style()
    bp = test_batches()
    x1 = bp[0][0][:1]
    rows, ref = [], {}

    for arch, ckpt in (("R50", "best_model.pth"), ("R18", "r18_student.pth")):
        if not (CKPT / ckpt).exists():
            continue
        model, _ = load_checkpoint(CKPT / ckpt, "cuda")
        info = {"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                "GMACs": round(count_gmacs(model), 2)}
        model.cuda()
        f = lambda x, m=model: m(torch.from_numpy(x).cuda()).float().cpu().numpy()  # noqa: E731
        with torch.no_grad():
            acc, emb = accuracy_on_test(f, bp)
            p50, p95 = latency(f, x1)
        ref[arch] = emb
        info["fp32_size_MB"] = round(info["params_M"] * 4, 1)
        rows.append({"model": arch, "runtime": "PyTorch", "precision": "FP32", "device": "GPU", "threads": None,
                     "size_MB": info["fp32_size_MB"], **info, "p50_ms": p50, "p95_ms": p95, **acc, "parity_cos": 1.0})
        _CACHE[arch] = info

        for prec, fname in (("FP32", f"face_{arch.lower()}.onnx"), ("INT8", f"face_{arch.lower()}_int8.onnx")):
            path = CKPT / fname
            if not path.exists():
                continue
            size = round(path.stat().st_size / 1e6, 1)
            sess_acc = OnnxEmbedder(path, "cpu", threads=8)
            acc, emb = accuracy_on_test(sess_acc.run, bp)
            parity = round(float((emb * ref[arch]).sum(1).mean()), 5)
            for th in (1, 4):
                s = OnnxEmbedder(path, "cpu", threads=th)
                p50, p95 = latency(s.run, x1, n=200 if arch == "R50" else 300)
                rows.append({"model": arch, "runtime": "ONNX Runtime", "precision": prec, "device": "CPU",
                             "threads": th, "size_MB": size, **info, "p50_ms": p50, "p95_ms": p95, **acc,
                             "parity_cos": parity})
                print(rows[-1], flush=True)
            if prec == "FP32":
                s = OnnxEmbedder(path, "cuda")
                if "CUDAExecutionProvider" in s.providers:
                    p50, p95 = latency(s.run, x1)
                    rows.append({"model": arch, "runtime": "ONNX Runtime CUDA", "precision": prec, "device": "GPU",
                                 "threads": None, "size_MB": size, **info, "p50_ms": p50, "p95_ms": p95, **acc,
                                 "parity_cos": parity})

    trt = load_json(RESULTS / "tensorrt.json", {}) or {}
    for r in trt.get("rows", []):
        arch = "R18" if "r18" in r["model"] else "R50"
        rows.append({**r, "model": arch, **_CACHE.get(arch, {})})

    # ---- full pipeline on raw photos: detect + align + embed (R18 INT8, CPU 4 threads)
    pipe = {}
    int8 = CKPT / "face_r18_int8.onnx"
    if int8.exists():
        aligner, emb = FaceAligner(YUNET), OnnxEmbedder(int8, "cpu", threads=4)
        meta = pd.read_csv(DATA / "processed" / "lfw_meta.csv").query("split == 'test'").head(100)
        imgs = [cv2.imread(str(RAW / "lfw" / n / f)) for n, f in zip(meta.orig_name, meta.orig_file)]
        det_t, emb_t = [], []
        for img in imgs[:10]:
            emb.embed_crops([aligner.align(img)[0]])
        for img in imgs:
            t = time.perf_counter(); crop, _ = aligner.align(img); t1 = time.perf_counter()
            emb.embed_crops([crop]); t2 = time.perf_counter()
            det_t.append((t1 - t) * 1e3); emb_t.append((t2 - t1) * 1e3)
        pipe = {"image_size": "250x250 (LFW)", "detector": "YuNet 2023mar (CPU)", "embedder": "R18 INT8 (CPU, 4 threads)",
                "detect_align_ms_p50": round(float(np.median(det_t)), 2), "embed_ms_p50": round(float(np.median(emb_t)), 2),
                "total_ms_p50": round(float(np.median(np.add(det_t, emb_t))), 2),
                "fps_estimate": round(1000 / float(np.median(np.add(det_t, emb_t))), 1)}
        print(f"[edge] pipeline: {pipe}")

    # ---- the edge model has its own score distribution -> calibrate ITS threshold on val (same FAR target as
    # the R50 submission model), then report it on test. Used by the API and the webcam demo.
    if int8.exists():
        res = load_json(RESULTS / "evaluation_results.json")
        op = res["verification"]["final_threshold"]["operating_point"]
        far = {"FAR=1%": 1e-2, "FAR=0.1%": 1e-3}.get(op, 1e-3)
        s8 = OnnxEmbedder(int8, "cpu", threads=8)

        def emb_split(split):
            df_ = lfw_split(split)
            crops = [cv2.imread(str(DATA / p)) for p in df_.path]
            return df_, np.concatenate([s8.embed_crops(crops[i:i + 64], tta=True) for i in range(0, len(crops), 64)])

        vdf, ve = emb_split("val")
        iu, ju = np.triu_indices(len(vdf), 1)
        dup = vdf.dup_cluster.to_numpy(); keep = dup[iu] != dup[ju]
        same = vdf.identity.to_numpy()[iu[keep]] == vdf.identity.to_numpy()[ju[keep]]
        vs = (ve[iu[keep]] * ve[ju[keep]]).sum(1)
        from common.metrics import binary_metrics, threshold_at_far
        t_edge = threshold_at_far(vs[~same], far)
        tdf, te = emb_split("test")
        idx = {p: i for i, p in enumerate(tdf.path)}
        pairs = pd.read_csv(PAIRS_DIR / "pairs_test.csv")
        ts = (te[pairs.img1.map(idx)] * te[pairs.img2.map(idx)]).sum(1)
        m = binary_metrics(pairs.label.to_numpy(), ts, t_edge)
        update_json(RESULTS / "evaluation_results.json", "edge_threshold",
                    {"model": int8.name, "operating_point": op, "threshold": t_edge, "calibrated_on": "val all-vs-all",
                     "test_at_threshold": m})
        print(f"[edge] R18 INT8 threshold ({op}, val) = {t_edge:.4f}; test: {m}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "edge_benchmark.csv", index=False)
    update_json(RESULTS / "evaluation_results.json", "edge", {"rows": rows, "pipeline": pipe})
    print(df.to_string())

    # ---- chart: CPU single-thread latency per deployable variant, annotated with accuracy
    cpu = df[(df.device == "CPU") & (df.threads == 1)].copy()
    if len(cpu):
        cpu["label"] = cpu.model + " " + cpu.precision + "  (" + cpu.size_MB.astype(str) + " MB)"
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(cpu) + 1.4))
        ax.barh(cpu.label, cpu.p50_ms, color=SERIES[0], height=0.55)
        for i, (v, a) in enumerate(zip(cpu.p50_ms, cpu["TAR@FAR=1%"])):
            ax.text(v, i, f"  {v:.1f} ms  |  TAR@FAR1% {a * 100:.1f}%", va="center", fontsize=8.5, color=INK2)
        ax.invert_yaxis(); ax.set_xlabel("Latency per face, batch 1, 1 CPU thread (ms, p50)")
        ax.set_xlim(0, cpu.p50_ms.max() * 1.9); ax.grid(axis="y", visible=False)
        ax.set_title("Edge deployment: latency vs accuracy (ONNX Runtime, CPU)")
        fig.savefig(RESULTS / "edge_benchmark.png"); plt.close(fig)


if __name__ == "__main__":
    main()
