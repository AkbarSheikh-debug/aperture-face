"""Training objectives. The classifier weights live HERE, never in the embedding model, so anything that
calls model(x) gets an L2-normalised embedding and cosine similarity can never be computed on logits."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin softmax (Deng et al., CVPR 2019): logit_y = s * cos(theta_y + m)."""

    def __init__(self, emb_dim, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, labels, margin=None):
        m = self.m if margin is None else margin
        emb = F.normalize(emb.float(), dim=1)
        cos = F.linear(emb, F.normalize(self.W, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        logits = cos
        if m > 0:
            sin = torch.sqrt(1.0 - cos ** 2)
            phi = cos * math.cos(m) - sin * math.sin(m)                         # cos(theta + m)
            # keep the target logit monotonic when theta + m > pi (InsightFace trick)
            phi = torch.where(cos > math.cos(math.pi - m), phi, cos - math.sin(math.pi - m) * m)
            onehot = F.one_hot(labels, cos.shape[1]).bool()
            logits = torch.where(onehot, phi, cos)
        loss = F.cross_entropy(self.s * logits, labels)
        acc = (cos.argmax(1) == labels).float().mean()
        return loss, acc


class SoftmaxHead(nn.Module):
    """Plain identity classification with cross-entropy (baseline in the ablation)."""

    def __init__(self, emb_dim, n_classes, **_):
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, emb, labels, margin=None):
        logits = self.fc(emb.float())
        return F.cross_entropy(logits, labels), (logits.argmax(1) == labels).float().mean()


class BatchHardTripletLoss(nn.Module):
    """Hermans et al. 2017: for each anchor, hardest positive and hardest negative inside the P x K batch."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        emb = F.normalize(emb.float(), dim=1)
        d = (2.0 - 2.0 * emb @ emb.t()).clamp_min(1e-12).sqrt()                # euclidean on the unit sphere
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
        d_pos = d.masked_fill(~same | eye, -1.0).max(1).values
        d_neg = d.masked_fill(same, 1e4).min(1).values
        valid = d_pos >= 0
        if valid.sum() == 0:
            return emb.sum() * 0.0
        return F.relu(d_pos[valid] - d_neg[valid] + self.margin).mean()


class CombinedLoss(nn.Module):
    """kind: 'arcface' | 'softmax' | 'triplet' | 'arcface_triplet'."""

    def __init__(self, kind, emb_dim, n_classes, s=64.0, m=0.5, triplet_margin=0.3, triplet_weight=0.1):
        super().__init__()
        self.kind = kind
        self.head = None
        if kind in ("arcface", "arcface_triplet"):
            self.head = ArcFaceHead(emb_dim, n_classes, s, m)
        elif kind == "softmax":
            self.head = SoftmaxHead(emb_dim, n_classes)
        self.triplet = BatchHardTripletLoss(triplet_margin) if "triplet" in kind else None
        self.tw = 1.0 if kind == "triplet" else triplet_weight

    def forward(self, emb, labels, margin=None):
        logs = {}
        loss = torch.zeros((), dtype=torch.float32, device=emb.device)
        if self.head is not None:
            l_cls, acc = self.head(emb, labels, margin)
            loss = loss + l_cls
            logs.update(cls=l_cls.item(), acc=acc.item())
        if self.triplet is not None:
            l_tri = self.triplet(emb, labels)
            loss = loss + self.tw * l_tri
            logs["triplet"] = l_tri.item()
        return loss, logs
