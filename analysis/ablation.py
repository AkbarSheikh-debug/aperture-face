"""Ablation: why ArcFace + triplet, and why synthetic pre-training?

Every variant is trained by train.py with config overrides and scored the same way:
  val  (model-selection data) : all-vs-all AUC, TAR@FAR=1%
  test (reported only)        : 5k/5k pairs AUC, EER, TAR@FAR=1%, single-shot Rank-1
Decisions (which recipe to ship) are made on the val columns; test columns are shown for completeness.

Usage: python -m analysis.ablation [--only lfw_arcface ...] [--skip-train]
"""
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from common.data import lfw_split
from common.embed import embed_paths
from common.io import CKPT, PAIRS_DIR, RESULTS, ROOT, save_json
from common.metrics import auc, cmc, eer, tar_at_far
from model import ImageNetGAPBaseline, load_checkpoint

LFW_FROM_IMAGENET = ["init_checkpoint=null", "optim.lr=0.05", "optim.backbone_lr_mult=0.2", "optim.head_lr_mult=2.0",
                     "epochs=30", "freeze_bn_epochs=0"]

VARIANTS = {
    # name: (description, train overrides or None, checkpoint, tta)
    "imagenet_frozen": ("ImageNet ResNet50, GAP features, no face training", None, None, True),
    "lfw_softmax": ("LFW only: softmax cross-entropy", ["loss.kind=softmax"] + LFW_FROM_IMAGENET, None, True),
    "lfw_triplet": ("LFW only: batch-hard triplet", ["loss.kind=triplet"] + LFW_FROM_IMAGENET, None, True),
    "lfw_arcface": ("LFW only: ArcFace", ["loss.kind=arcface"] + LFW_FROM_IMAGENET, None, True),
    "lfw_arcface_triplet": ("LFW only: ArcFace + 0.1 triplet", ["loss.kind=arcface_triplet"] + LFW_FROM_IMAGENET,
                            None, True),
    "v1_synthetic_only": ("v1: 4k synthetic ids only (no real faces in stage 1) + LFW fine-tune", None,
                          "v1_best_model.pth", True),
    "stage1_only": ("Stage 1 only: 10k synthetic ids + LFW-train ids jointly, no stage-2 fine-tune", None,
                    "r50_digiface.pth", True),
    "final_no_tta": ("Stage 1 + stage-2 LFW fine-tune, no flip-TTA", None, "best_model.pth", False),
    "final": ("Stage 1 + stage-2 LFW fine-tune, flip-TTA (submission)", None, "best_model.pth", True),
}


def score(model, tta, device="cuda"):
    out = {}
    val = lfw_split("val")
    e = embed_paths(model, val.path.tolist(), device, tta=tta)
    iu, ju = np.triu_indices(len(val), 1)
    dup = val.dup_cluster.to_numpy(); keep = dup[iu] != dup[ju]; iu, ju = iu[keep], ju[keep]
    lab = (val.identity.to_numpy()[iu] == val.identity.to_numpy()[ju]).astype(int)
    s = (e[iu] * e[ju]).sum(1)
    out["val_AUC"] = auc(lab, s); out["val_TAR@FAR=1%"] = tar_at_far(lab, s, 1e-2)[0]

    test = lfw_split("test")
    e = embed_paths(model, test.path.tolist(), device, tta=tta)
    idx = {p: i for i, p in enumerate(test.path)}
    pairs = pd.read_csv(PAIRS_DIR / "pairs_test.csv")
    s = (e[pairs.img1.map(idx)] * e[pairs.img2.map(idx)]).sum(1)
    y = pairs.label.to_numpy()
    out["test_AUC"] = auc(y, s); out["test_EER"] = eer(y, s)[0]
    out["test_TAR@FAR=1%"] = tar_at_far(y, s, 1e-2)[0]
    ids = test.identity.to_numpy(); rng = np.random.default_rng(42)
    g_idx = [rng.choice(np.flatnonzero(ids == i)) for i in sorted(set(ids))]
    pmask = np.ones(len(ids), bool); pmask[g_idx] = False
    curve, _, _ = cmc(e[pmask] @ e[g_idx].T, ids[g_idx], ids[pmask], max_rank=5)
    out["test_Rank1"] = float(curve[0]); out["test_Rank5"] = float(curve[4])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing ablation checkpoints")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv = RESULTS / "ablation.csv"
    prev = pd.read_csv(csv).set_index("variant") if csv.exists() else pd.DataFrame()
    rows = {k: r.to_dict() for k, r in prev.iterrows()} if len(prev) else {}

    for name, (desc, overrides, ckpt, tta) in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        if overrides is not None:
            ckpt = f"ablation_{name}.pth"
            if not (args.skip_train and (CKPT / ckpt).exists()):
                cmd = [sys.executable, "train.py", "--config", "configs/finetune_lfw.yaml", "--set",
                       f"name=ablation_{name}", f"out=checkpoints/{ckpt}", *overrides]
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
        if name == "imagenet_frozen":
            model = ImageNetGAPBaseline("resnet50").to(device).eval()
        else:
            if not (CKPT / ckpt).exists():
                print(f"[ablation] skip {name}: {ckpt} missing")
                continue
            model, _ = load_checkpoint(CKPT / ckpt, device)
        m = score(model, tta, device)
        rows[name] = {"description": desc, **m}
        print(f"[ablation] {name}: { {k: round(v, 4) for k, v in m.items()} }", flush=True)
        order = [k for k in VARIANTS if k in rows]
        pd.DataFrame([{"variant": k, **rows[k]} for k in order]).to_csv(csv, index=False)
    save_json(rows, RESULTS / "ablation.json")


if __name__ == "__main__":
    main()
