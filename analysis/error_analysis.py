"""Failure analysis on the TEST pairs at the final (val-chosen) threshold:
the highest-scoring impostor pairs (false accepts / near misses) and lowest-scoring genuine pairs (false rejects).

Usage: python -m analysis.error_analysis [--k 8]
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from common.io import DATA, RESULTS, load_json, update_json
from common.plotting import INK, INK2, apply_style


def show_pairs(axes, rows, title_fn):
    for ax_pair, (_, r) in zip(axes, rows.iterrows()):
        for ax, p in zip(ax_pair, (r.img1, r.img2)):
            ax.imshow(cv2.cvtColor(cv2.imread(str(DATA / p)), cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        ax_pair[0].set_title(title_fn(r), fontsize=8.5, color=INK2, loc="left", x=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    apply_style()
    thr = load_json(RESULTS / "evaluation_results.json")["verification"]["final_threshold"]["threshold"]
    s = pd.read_csv(RESULTS / "scores_test.csv")
    fa = s[s.label == 0].nlargest(args.k, "cosine")
    fr = s[s.label == 1].nsmallest(args.k, "cosine")
    meta = pd.read_csv(DATA / "processed" / "lfw_meta.csv").set_index("path")
    name = lambda p: meta.loc[p, "orig_name"].replace("_", " ")  # noqa: E731

    fig = plt.figure(figsize=(1.45 * args.k, 7.9))
    gs = fig.add_gridspec(5, args.k, height_ratios=[1, 1, 0.32, 1, 1], hspace=0.3, wspace=0.05,
                          left=0.01, right=0.99, top=0.93, bottom=0.01)
    pairs_fa = [(fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])) for i in range(args.k)]
    pairs_fr = [(fig.add_subplot(gs[3, i]), fig.add_subplot(gs[4, i])) for i in range(args.k)]
    spacer = fig.add_subplot(gs[2, :])                 # dedicated row for the second heading (no overlap)
    spacer.axis("off")
    spacer.text(0.0, 0.2, "Hardest genuine pairs (same person, lowest similarity)", color=INK, fontsize=10,
                weight="bold", transform=spacer.transAxes)
    show_pairs(pairs_fa, fa, lambda r: f"{r.cosine:.2f}" + (" FA" if r.cosine >= thr else ""))
    show_pairs(pairs_fr, fr, lambda r: f"{r.cosine:.2f}" + (" FR" if r.cosine < thr else ""))
    fig.text(0.01, 0.97, f"Hardest impostor pairs (different people, highest similarity) - threshold {thr:.3f}",
             color=INK, fontsize=10, weight="bold")
    fig.savefig(RESULTS / "error_cases.png"); plt.close(fig)

    out = {"threshold": thr,
           "false_accepts": int((s[s.label == 0].cosine >= thr).sum()),
           "false_rejects": int((s[s.label == 1].cosine < thr).sum()),
           "hardest_impostors": [{"a": name(r.img1), "b": name(r.img2), "cosine": round(r.cosine, 4)} for _, r in fa.iterrows()],
           "hardest_genuine": [{"person": name(r.img1), "cosine": round(r.cosine, 4)} for _, r in fr.iterrows()]}
    update_json(RESULTS / "evaluation_results.json", "error_analysis", out)
    print(out)


if __name__ == "__main__":
    main()
