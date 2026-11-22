"""Similarity Grad-CAM: which facial regions drive the cosine score between two faces?

For a pair (A, B): target = cos(f(A), f(B).detach()); gradients of that score w.r.t. the last ResNet stage of A
weight its activation maps (Grad-CAM, Selvaraju et al. 2017). A model that learned faces should focus on
eyes / nose / mouth, not on hair, background or clothing.

Usage: python -m analysis.gradcam [--n 4]
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.data import eval_transform, load_rgb112
from common.io import CKPT, DATA, RESULTS
from common.plotting import INK2, apply_style
from model import load_checkpoint


def cam_for(model, xa, xb):
    feats = {}
    layer = model.body[-1]
    h = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    xa = xa.requires_grad_(True)
    ea = model(xa)
    h.remove()
    with torch.no_grad():
        eb = model(xb)
    act = feats["a"]
    act.retain_grad()
    score = (ea * eb).sum()
    model.zero_grad()
    score.backward()
    w = act.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(112, 112), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    return cam / (cam.max() + 1e-8), float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    apply_style()
    model, _ = load_checkpoint(CKPT / "best_model.pth", "cuda")
    model.eval()
    s = pd.read_csv(RESULTS / "scores_test.csv")
    pairs = s[s.label == 1].sample(args.n, random_state=3)
    tf = eval_transform()
    fig, axes = plt.subplots(args.n, 4, figsize=(7.6, 2.0 * args.n))
    for row, (_, r) in zip(axes, pairs.iterrows()):
        xa = tf(load_rgb112(r.img1))[None].cuda(); xb = tf(load_rgb112(r.img2))[None].cuda()
        cam, sc = cam_for(model, xa, xb)
        ia = cv2.cvtColor(cv2.imread(str(DATA / r.img1)), cv2.COLOR_BGR2RGB)
        ib = cv2.cvtColor(cv2.imread(str(DATA / r.img2)), cv2.COLOR_BGR2RGB)
        heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        over = (0.55 * ia + 0.45 * heat).astype(np.uint8)
        for ax, im, t in zip(row, (ia, ib, heat, over), ("Image A", "Image B", "Grad-CAM (A)", f"Overlay  cos={sc:.2f}")):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(t, fontsize=8.5, color=INK2)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle("Which regions drive the similarity score? (genuine test pairs)", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "gradcam.png"); plt.close(fig)
    print("[gradcam] results/gradcam.png")


if __name__ == "__main__":
    main()
