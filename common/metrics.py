"""Verification and identification metrics, all computed from real cosine similarity scores."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------------------------------------- verification
def roc(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    return fpr, tpr, thr


def auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def eer(labels, scores):
    """Equal error rate: the point where FAR (=FPR) equals FRR (=1-TPR). Linear interpolation between ROC points."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    # interpolate between i and its neighbour where the sign of (fnr - fpr) changes
    j = i + 1 if i + 1 < len(fpr) and (fnr[i] - fpr[i]) * (fnr[i + 1] - fpr[i + 1]) <= 0 else max(i - 1, 0)
    d_i, d_j = fnr[i] - fpr[i], fnr[j] - fpr[j]
    w = 0.0 if d_i == d_j else d_i / (d_i - d_j)
    e = fpr[i] + w * (fpr[j] - fpr[i])
    t = thr[i] + w * (thr[j] - thr[i]) if np.isfinite(thr[i]) and np.isfinite(thr[j]) else thr[i]
    return float(e), float(t)


def threshold_at_far(neg_scores, far):
    """Smallest threshold whose empirical FAR (fraction of impostors with score >= thr) is <= far."""
    neg = np.sort(np.asarray(neg_scores))[::-1]
    k = int(np.floor(far * len(neg)))            # number of impostors we are allowed to accept
    if k >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))   # strictly above the (k+1)-th highest impostor


def tar_at_far(labels, scores, far):
    labels, scores = np.asarray(labels), np.asarray(scores)
    thr = threshold_at_far(scores[labels == 0], far)
    return float((scores[labels == 1] >= thr).mean()), thr


def binary_metrics(labels, scores, thr):
    labels, scores = np.asarray(labels).astype(bool), np.asarray(scores)
    pred = scores >= thr
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
    tn = int((~pred & ~labels).sum()); fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(thr), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "TPR(TAR)": recall, "FPR(FAR)": fp / (fp + tn) if fp + tn else 0.0,
        "FRR": fn / (fn + tp) if fn + tp else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
    }


def best_f1_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    cand = np.unique(np.quantile(scores, np.linspace(0, 1, 2001)))
    f1s = [binary_metrics(labels, scores, t)["f1"] for t in cand]
    return float(cand[int(np.argmax(f1s))])


def bootstrap_ci(labels, scores, fn, n=1000, seed=0, alpha=0.05):
    """Stratified bootstrap (resample genuine and impostor scores separately). Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    vals = []
    for _ in range(n):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        vals.append(fn(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------- identification
def cmc(sim, gallery_labels, probe_labels, max_rank=20):
    """sim: [n_probe, n_gallery]. Gallery holds one entry per identity. Returns cmc[k-1] = P(true id in top-k),
    and the 1-based rank of the true identity for every probe."""
    gallery_labels = np.asarray(gallery_labels)
    order = np.argsort(-sim, axis=1)
    ranked = gallery_labels[order]
    hits = ranked == np.asarray(probe_labels)[:, None]
    ranks = hits.argmax(1) + 1
    ranks[~hits.any(1)] = sim.shape[1] + 1
    curve = np.array([(ranks <= k).mean() for k in range(1, max_rank + 1)])
    return curve, ranks, order


def retrieval_map(emb, labels, exclude=None):
    """Re-ID style mean average precision: every image queries all other images.
    exclude: optional bool [n, n] mask of pairs to ignore (e.g. near-duplicates)."""
    labels = np.asarray(labels)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    if exclude is not None:
        sim[exclude] = -np.inf
    aps = []
    for i in range(len(labels)):
        valid = np.isfinite(sim[i])
        order = np.argsort(-sim[i][valid])
        rel = (labels[valid][order] == labels[i]).astype(float)
        if rel.sum() == 0:
            continue
        prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
        aps.append((prec * rel).sum() / rel.sum())
    return float(np.mean(aps))


def open_set_dir(sim_known, true_known, sim_unknown, gallery_labels, far):
    """Open-set identification (ISO/IEC 19795-1 style).
    Threshold chosen so that a fraction `far` of unknown probes are (wrongly) accepted; DIR = fraction of known
    probes that are rank-1 correct AND above that threshold."""
    gallery_labels = np.asarray(gallery_labels)
    unk_max = sim_unknown.max(1)
    thr = threshold_at_far(unk_max, far)
    top = sim_known.argmax(1)
    correct = gallery_labels[top] == np.asarray(true_known)
    detected = sim_known.max(1) >= thr
    return float((correct & detected).mean()), float(thr)
