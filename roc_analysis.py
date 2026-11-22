"""Tasks 6 + 7 - ROC / AUC and threshold selection, from the real cosine scores written by evaluate_pairs.py.

Protocol (no test-set tuning):
  1. candidate thresholds are computed on the VALIDATION identities (all-vs-all val scores: ~10^5-10^6
     impostors, so FAR=0.1% is estimated from hundreds of false accepts, not 5)
        - EER threshold, FAR=1% threshold, FAR=0.1% threshold, max-F1 threshold
  2. those fixed thresholds are applied unchanged to the TEST pairs and reported (with bootstrap 95% CIs)
  3. the final operating threshold is the one named by --operating-point

Threshold trade-off: raising the threshold accepts fewer impostors (FAR down) but rejects more genuine users
(FRR up); lowering it does the opposite. far_frr_vs_threshold.png shows both curves.

Usage: python roc_analysis.py [--operating-point far_1e-3] [--tag ""]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.io import RESULTS, update_json
from common.metrics import auc, best_f1_threshold, binary_metrics, bootstrap_ci, eer, roc, tar_at_far, threshold_at_far
from common.plotting import INK2, MUTED, SERIES, apply_style, threshold_line

OP_LABELS = {"eer": "EER", "far_1e-2": "FAR=1%", "far_1e-3": "FAR=0.1%", "max_f1": "max-F1"}


def val_thresholds(val_pairs, val_all):
    lab = np.r_[np.ones(len(val_all["genuine"])), np.zeros(len(val_all["impostor"]))]
    s = np.r_[val_all["genuine"], val_all["impostor"]]
    return {
        "eer": eer(lab, s)[1],
        "far_1e-2": threshold_at_far(val_all["impostor"], 1e-2),
        "far_1e-3": threshold_at_far(val_all["impostor"], 1e-3),
        # max-F1 depends on the class ratio, so it is fitted on the balanced val PAIRS
        "max_f1": best_f1_threshold(val_pairs.label.to_numpy(), val_pairs.cosine.to_numpy()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operating-point", default="far_1e-3", choices=list(OP_LABELS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    apply_style()

    val = pd.read_csv(RESULTS / f"scores_val{args.tag}.csv")
    test = pd.read_csv(RESULTS / f"scores_test{args.tag}.csv")
    val_all = np.load(RESULTS / f"allpairs_val{args.tag}.npz")
    test_all = np.load(RESULTS / f"allpairs_test{args.tag}.npz")
    y, s = test.label.to_numpy(), test.cosine.to_numpy()

    # ---------------------------------------------------------------- threshold-free metrics on test pairs
    fpr, tpr, thr = roc(y, s)
    test_auc = auc(y, s)
    test_eer, test_eer_thr = eer(y, s)
    tar2, _ = tar_at_far(y, s, 1e-2)
    tar3, _ = tar_at_far(y, s, 1e-3)
    ya = np.r_[np.ones(len(test_all["genuine"])), np.zeros(len(test_all["impostor"]))]
    sa = np.r_[test_all["genuine"], test_all["impostor"]]
    rng_ci = lambda fn: bootstrap_ci(y, s, fn, n=args.bootstrap)  # noqa: E731
    threshold_free = {
        "pairs": {"genuine": int(y.sum()), "impostor": int((y == 0).sum())},
        "AUC": test_auc, "AUC_95CI": rng_ci(auc),
        "EER": test_eer, "EER_95CI": rng_ci(lambda a, b: eer(a, b)[0]), "EER_threshold_on_test": test_eer_thr,
        "TAR@FAR=1%": tar2, "TAR@FAR=1%_95CI": rng_ci(lambda a, b: tar_at_far(a, b, 1e-2)[0]),
        "TAR@FAR=0.1%": tar3,
        "all_vs_all": {"genuine": int(len(test_all["genuine"])), "impostor": int(len(test_all["impostor"])),
                       "AUC": auc(ya, sa), "EER": eer(ya, sa)[0],
                       "TAR@FAR=1%": tar_at_far(ya, sa, 1e-2)[0], "TAR@FAR=0.1%": tar_at_far(ya, sa, 1e-3)[0],
                       "TAR@FAR=0.01%": tar_at_far(ya, sa, 1e-4)[0]},
        "score_stats": {"genuine_mean": float(s[y == 1].mean()), "genuine_std": float(s[y == 1].std()),
                        "impostor_mean": float(s[y == 0].mean()), "impostor_std": float(s[y == 0].std())},
    }

    # ------------------------------------------------------ thresholds fixed on val, applied to test
    vt = val_thresholds(val, val_all)
    operating = {}
    for name, t in vt.items():
        m = binary_metrics(y, s, t)
        m["val_threshold_source"] = "val all-vs-all impostors" if name.startswith("far") or name == "eer" else "val pairs"
        operating[name] = m
    final_name = args.operating_point
    final_thr = vt[final_name]
    final = dict(operating[final_name], operating_point=OP_LABELS[final_name],
                 rule=f"cosine >= {final_thr:.4f} -> Match ; cosine < {final_thr:.4f} -> Non-Match",
                 f1_95CI=rng_ci(lambda a, b: binary_metrics(a, b, final_thr)["f1"]))
    print(f"[roc] AUC={test_auc:.4f}  EER={test_eer:.4f}  TAR@FAR1%={tar2:.4f}  TAR@FAR0.1%={tar3:.4f}")
    for k, m in operating.items():
        print(f"[roc] {OP_LABELS[k]:>9} thr={m['threshold']:.4f}  TAR={m['TPR(TAR)']:.4f}  FAR={m['FPR(FAR)']:.4f}  "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} acc={m['accuracy']:.4f}")
    print(f"[roc] FINAL ({OP_LABELS[final_name]}): {final['rule']}")

    key = "verification" + args.tag
    update_json(RESULTS / "evaluation_results.json", key, {
        "threshold_free_test": threshold_free, "val_thresholds": vt,
        "test_at_val_thresholds": operating, "final_threshold": final})
    if args.tag:
        return

    # ------------------------------------------------------------------------------------------ figures
    ops = [(OP_LABELS[k], vt[k]) for k in ("eer", "far_1e-2", "far_1e-3")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, logx in zip(axes, (False, True)):
        ax.plot(fpr, tpr, color=SERIES[0], label=f"Test pairs  (AUC = {test_auc:.4f})")
        fa, ta, _ = roc(ya, sa)
        ax.plot(fa, ta, color=SERIES[1], lw=1.6, label="Test all-vs-all")
        if not logx:
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
        ax.set_xlabel("False Positive Rate (FAR)"); ax.set_ylabel("True Positive Rate (TAR)")
        for lab, t in ops:
            m = binary_metrics(y, s, t)
            ax.plot(max(m["FPR(FAR)"], 1e-5), m["TPR(TAR)"], "o", ms=8, mfc=SERIES[2], mec="#fcfcfb", mew=2)
            if logx:   # labels only on the log panel: the three points sit on top of each other on the linear one
                ax.annotate(f"{lab} (thr {t:.2f})", (max(m["FPR(FAR)"], 1e-5), m["TPR(TAR)"]), xytext=(8, -14),
                            textcoords="offset points", fontsize=8, color=INK2)
        if logx:
            ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_title("ROC, log-scale FAR")
        else:
            ax.set_title("ROC curve (test identities)")
        ax.set_ylim(0, 1.01)
        ax.legend(loc="lower right")
    fig.savefig(RESULTS / "roc_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(min(s.min(), -0.2), 1.0, 90)
    ax.hist(s[y == 0], bins=bins, color=SERIES[1], alpha=0.75, label=f"Impostor (different person), n={int((y == 0).sum())}",
            edgecolor="#fcfcfb", linewidth=0.5)
    ax.hist(s[y == 1], bins=bins, color=SERIES[0], alpha=0.75, label=f"Genuine (same person), n={int(y.sum())}",
            edgecolor="#fcfcfb", linewidth=0.5)
    threshold_line(ax, final_thr, f"final threshold {final_thr:.3f} ({OP_LABELS[final_name]})")
    threshold_line(ax, vt["eer"], f"EER thr {vt['eer']:.3f}", color=MUTED, y_text=0.86)
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Number of pairs")
    ax.set_title("Similarity distribution - test pairs")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.8))
    fig.savefig(RESULTS / "similarity_distribution.png"); plt.close(fig)

    ts = np.linspace(-0.2, 1.0, 600)
    far_c = [(test_all["impostor"] >= t).mean() for t in ts]
    frr_c = [(test_all["genuine"] < t).mean() for t in ts]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(ts, far_c, color=SERIES[1], label="FAR - impostors accepted (false accepts)")
    ax.plot(ts, frr_c, color=SERIES[0], label="FRR - genuine users rejected (false rejects)")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1.05)
    threshold_line(ax, final_thr, f"final {final_thr:.3f}")
    ax.set_xlabel("Cosine-similarity threshold  (higher = stricter)"); ax.set_ylabel("Error rate (log)")
    ax.set_title("Threshold trade-off: false accepts vs false rejects (test, all-vs-all)")
    ax.legend(loc="lower left")
    fig.savefig(RESULTS / "far_frr_vs_threshold.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(np.clip(fpr, 1e-5, 1), np.clip(1 - tpr, 1e-5, 1), color=SERIES[0], label="Test pairs")
    ax.plot(np.clip(fa, 1e-6, 1), np.clip(1 - ta, 1e-6, 1), color=SERIES[1], lw=1.6, label="Test all-vs-all")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FAR"); ax.set_ylabel("FRR"); ax.set_title("DET curve (lower-left is better)")
    ax.legend(loc="upper right")
    fig.savefig(RESULTS / "det_curve.png"); plt.close(fig)
    print("[roc] figures written to results/")


if __name__ == "__main__":
    main()
