"""Datasets, transforms and the P x K identity sampler."""
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from common.io import DATA, DIGIFACE_META, LFW_META

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)   # maps [0,1] -> [-1,1]; same at train / eval / ONNX / API


class RandomDownscale:
    """Simulate low-resolution / compressed cameras (common on edge devices)."""

    def __init__(self, p=0.2, min_size=32):
        self.p, self.min_size = p, min_size

    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.randint(self.min_size, img.size[0])
        return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)


class RandomBlur:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        return img


def train_transform(strength: str = "heavy", size: int = 112):
    ops = [T.RandomHorizontalFlip()]
    if strength == "heavy":
        ops += [
            T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.25, 0.05)], p=0.8),
            T.RandomGrayscale(p=0.1),
            RandomBlur(0.15),
            RandomDownscale(0.2),
        ]
    elif strength == "light":
        ops += [T.RandomApply([T.ColorJitter(0.2, 0.2, 0.1)], p=0.5)]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if strength == "heavy":
        ops += [T.RandomErasing(p=0.2, scale=(0.02, 0.12), value=0)]   # occlusion (glasses, masks, hands)
    return T.Compose(ops)


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def load_rgb112(path):
    img = Image.open(DATA / path).convert("RGB")
    if img.size != (112, 112):
        img = img.resize((112, 112), Image.BILINEAR)
    return img


class FaceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_rgb112(self.paths[i])), self.labels[i]


class PKSampler(Sampler):
    """Each batch = P identities x K images. Identity-balanced (stops LFW George_W_Bush from dominating)
    and guarantees in-batch positives for the batch-hard triplet term."""

    def __init__(self, labels, P, K, n_batches=None, seed=0):
        self.P, self.K = P, K
        self.by_class = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_class[int(y)].append(i)
        self.classes = list(self.by_class)
        self.n_batches = n_batches or max(1, len(labels) // (P * K))
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c in self.rng.sample(self.classes, min(self.P, len(self.classes))):
                idx = self.by_class[c]
                batch += self.rng.sample(idx, self.K) if len(idx) >= self.K else self.rng.choices(idx, k=self.K)
            yield batch


def load_lfw_meta():
    return pd.read_csv(LFW_META)


def lfw_split(split: str, eval_only: bool = True):
    """Rows of one identity-disjoint split. For val/test, eval_only keeps the per-identity capped subset."""
    df = load_lfw_meta()
    df = df[df.split == split]
    if eval_only and split in ("val", "test"):
        df = df[df.in_eval]
    return df.reset_index(drop=True)


def lfw_train_set():
    df = lfw_split("train")
    ids = sorted(df.identity.unique())
    remap = {k: i for i, k in enumerate(ids)}
    return df.path.tolist(), df.identity.map(remap).to_numpy(), len(ids)


def digiface_train_set(max_ids=None):
    df = pd.read_csv(DIGIFACE_META)
    if max_ids:
        keep = sorted(df.label.unique())[:max_ids]
        df = df[df.label.isin(keep)]
    ids = sorted(df.label.unique())
    remap = {k: i for i, k in enumerate(ids)}
    return df.path.tolist(), df.label.map(remap).to_numpy(), len(ids)


def make_eval_loader(paths, batch_size=256, num_workers=4, flip=False):
    tf = eval_transform() if not flip else T.Compose([T.RandomHorizontalFlip(p=1.0), eval_transform()])
    ds = FaceDataset(paths, np.zeros(len(paths)), tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       pin_memory=True)
