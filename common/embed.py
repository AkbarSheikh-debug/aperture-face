"""Batch embedding of image paths (PyTorch). Each image is embedded once; flip-TTA averages the embedding of
the image and its mirror, then re-normalises."""
import numpy as np
import torch
import torch.nn.functional as F

from common.data import make_eval_loader


@torch.no_grad()
def embed_paths(model, paths, device="cuda", tta=True, batch_size=256, num_workers=0):
    model.eval().to(device)
    out = []
    for x, _ in make_eval_loader(paths, batch_size, num_workers):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            e = model(x).float()
            if tta:
                e = e + model(torch.flip(x, dims=[3])).float()
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Inputs are already L2-normalised, so this is a dot product;
    we re-normalise anyway so the function is correct for any input."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)
