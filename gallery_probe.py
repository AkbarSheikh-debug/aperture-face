"""Task 5 - Gallery / Probe identification on the held-out TEST identities.

Protocols
  single-shot (primary): gallery = 1 seeded image per identity, probes = all remaining images
  multi-shot           : gallery template = mean of 3 seeded images per identity (re-normalised)
  open-set             : 20% of test identities are NOT enrolled; their images are "unknown" probes.
                         DIR@FAR = known probes correctly identified at rank 1 AND above a threshold that
                         wrongly accepts only FAR of the unknown probes.
  retrieval mAP        : every test image queries every other test image (Re-ID style).

Outputs results/gallery_probe_results.csv (one row per probe), results/cmc_curve.png,
and the 'identification' block of results/evaluation_results.json.

Usage: python gallery_probe.py [--seed 42]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.io import RESULTS, update_json
from common.metrics import cmc, open_set_dir, retrieval_map
from common.plotting import INK2, SERIES, apply_style


def build_gallery(emb, ids, rng, shots, enrolled):
    g_emb, g_lab, probe_mask = [], [], np.ones(len(ids), bool)
    for ident in enrolled:
        idx = np.flatnonzero(ids == ident)
        pick = rng.choice(idx, min(shots, len(idx) - 1), replace=False)   # always leave >= 1 probe
        t = emb[pick].mean(0)
        g_emb.append(t / np.linalg.norm(t)); g_lab.append(ident)
        probe_mask[pick] = False
    return np.stack(g_emb), np.array(g_lab), probe_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--unknown-frac", type=float, default=0.2)
    args = ap.parse_args()
    apply_style()
    z = np.load(RESULTS / f"embeddings_test{args.tag}.npz", allow_pickle=True)
    emb, ids, paths, dup = z["emb"], z["identity"], z["path"], z["dup_cluster"]
    all_ids = np.array(sorted(set(ids)))
    out, curves, rows = {}, {}, []

    for name, shots in (("single_shot", 1), ("multi_shot_3", 3)):
        rng = np.random.default_rng(args.seed)
        g_emb, g_lab, pmask = build_gallery(emb, ids, rng, shots, all_ids)
        sim = emb[pmask] @ g_emb.T                                       # cosine (unit vectors)
        curve, ranks, order = cmc(sim, g_lab, ids[pmask], max_rank=20)
        curves[name] = curve
        out[name] = {"gallery_identities": int(len(g_lab)), "gallery_images_per_identity": shots,
                     "probes": int(pmask.sum()), "rank1": float(curve[0]), "rank5": float(curve[4]),
                     "rank10": float(curve[9])}
        print(f"[gallery] {name}: {out[name]}")
        if name == "single_shot":
            for p, t, r, o, srow in zip(paths[pmask], ids[pmask], ranks, order, sim):
                top = o[:5]
                rows.append({"probe": p, "true_identity": t, "predicted_identity": g_lab[top[0]],
                             "top1_score": round(float(srow[top[0]]), 4),
                             "correct_rank1": bool(g_lab[top[0]] == t), "rank_of_true_identity": int(r),
                             **{f"top{k + 1}_id": g_lab[top[k]] for k in range(5)},
                             **{f"top{k + 1}_score": round(float(srow[top[k]]), 4) for k in range(5)}})

    # ---- open-set
    rng = np.random.default_rng(args.seed + 7)
    unknown = rng.choice(all_ids, int(round(args.unknown_frac * len(all_ids))), replace=False)
    known = np.array([i for i in all_ids if i not in set(unknown)])
    g_emb, g_lab, pmask = build_gallery(emb, ids, np.random.default_rng(args.seed), 1, known)
    is_unknown = np.isin(ids, unknown)
    kmask = pmask & ~is_unknown
    sim_k, sim_u = emb[kmask] @ g_emb.T, emb[is_unknown] @ g_emb.T
    out["open_set"] = {"enrolled_identities": int(len(known)), "unknown_identities": int(len(unknown)),
                       "known_probes": int(kmask.sum()), "unknown_probes": int(is_unknown.sum())}
    for far in (0.01, 0.05, 0.1):
        d, t = open_set_dir(sim_k, ids[kmask], sim_u, g_lab, far)
        out["open_set"][f"DIR@FAR={far:g}"] = d
        out["open_set"][f"threshold@FAR={far:g}"] = t
    print(f"[gallery] open-set: {out['open_set']}")

    # ---- retrieval mAP (ignore near-duplicate pairs)
    out["retrieval_mAP"] = retrieval_map(emb, ids, exclude=(dup[:, None] == dup[None, :]) & ~np.eye(len(dup), dtype=bool))
    print(f"[gallery] retrieval mAP: {out['retrieval_mAP']:.4f}")

    update_json(RESULTS / "evaluation_results.json", "identification" + args.tag, out)
    if args.tag:
        return
    pd.DataFrame(rows).to_csv(RESULTS / "gallery_probe_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ks = np.arange(1, 21)
    for (name, curve), col, lab in zip(curves.items(), SERIES, ("Single-shot gallery (1 img/id)",
                                                               "Multi-shot gallery (mean of 3 imgs/id)")):
        ax.plot(ks, curve * 100, color=col, marker="o", ms=4, label=lab)
    for name, col in zip(curves, SERIES):
        ax.annotate(f"R1 {curves[name][0] * 100:.1f}%", (1, curves[name][0] * 100), xytext=(6, -4 if name == "single_shot" else 6),
                    textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("Rank k"); ax.set_ylabel("Identification rate (%)")
    ax.set_title(f"CMC curve - {len(all_ids)} unseen test identities")
    ax.legend(loc="lower right")
    fig.savefig(RESULTS / "cmc_curve.png"); plt.close(fig)


if __name__ == "__main__":
    main()
