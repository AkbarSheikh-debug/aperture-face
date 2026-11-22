"""Task 4 - embeddings + cosine similarity.

  Image A -> ResNet -> 512-D L2-normalised embedding A
  Image B -> ResNet -> 512-D L2-normalised embedding B      cosine(A, B) = A . B

Every unique image is embedded once (flip-TTA by default). Outputs
  results/scores_{val,test}.csv          pair-level scores (img1, img2, label, cosine)
  results/embeddings_{val,test}.npz      every val/test image embedding (used by gallery_probe.py)
  results/allpairs_{val,test}.npz        all-vs-all genuine / impostor scores (for stable low-FAR estimates)

Usage: python evaluate_pairs.py [--ckpt checkpoints/best_model.pth] [--no-tta]
"""
import argparse

import numpy as np
import pandas as pd
import torch

from common.data import lfw_split
from common.embed import cosine, embed_paths
from common.io import CKPT, PAIRS_DIR, RESULTS
from model import load_checkpoint


def all_vs_all(emb, df):
    n = len(df)
    iu, ju = np.triu_indices(n, k=1)
    dup = df.dup_cluster.to_numpy()
    keep = dup[iu] != dup[ju]
    iu, ju = iu[keep], ju[keep]
    ids = df.identity.to_numpy()
    s = (emb[iu] * emb[ju]).sum(1)
    lab = ids[iu] == ids[ju]
    return s[lab].astype(np.float32), s[~lab].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT / "best_model.pth"))
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output files (used by the ablation)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    print(f"[eval] model {ckpt['config']}  best val during training: {ckpt.get('val')}")

    for split in ("val", "test"):
        df = lfw_split(split)
        emb = embed_paths(model, df.path.tolist(), device, tta=not args.no_tta)
        assert emb.shape[1] == ckpt["config"]["emb_dim"] and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
        index = {p: i for i, p in enumerate(df.path)}
        pairs = pd.read_csv(PAIRS_DIR / f"pairs_{split}.csv")
        a = emb[pairs.img1.map(index).to_numpy()]
        b = emb[pairs.img2.map(index).to_numpy()]
        pairs["cosine"] = cosine(a, b)
        pairs.to_csv(RESULTS / f"scores_{split}{args.tag}.csv", index=False)
        np.savez_compressed(RESULTS / f"embeddings_{split}{args.tag}.npz", emb=emb, path=df.path.to_numpy(),
                            identity=df.identity.to_numpy(), dup_cluster=df.dup_cluster.to_numpy())
        gen, imp = all_vs_all(emb, df)
        np.savez_compressed(RESULTS / f"allpairs_{split}{args.tag}.npz", genuine=gen, impostor=imp)
        g, i = pairs[pairs.label == 1].cosine, pairs[pairs.label == 0].cosine
        print(f"[eval] {split}: {len(pairs)} pairs | genuine mean {g.mean():.3f} +- {g.std():.3f} | "
              f"impostor mean {i.mean():.3f} +- {i.std():.3f} | all-vs-all: {len(gen)} genuine, {len(imp)} impostor")


if __name__ == "__main__":
    main()
