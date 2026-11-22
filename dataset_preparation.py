"""Task 1 - dataset preparation.

LFW (real faces, evaluation + fine-tuning)
  1. keep identities with >= 4 images
  2. YuNet face detection -> subject face -> 5-point similarity alignment -> 112x112 crop
  3. write  data/processed/lfw_aligned/person_001/img_01.jpg  (+ mapping back to the original LFW names)
  4. perceptual-hash (pHash) every aligned crop; near-duplicate images are linked, and identities that share a
     near-duplicate are put in the SAME split, so no image can leak across train / val / test
  5. identity-disjoint split 70 / 10 / 20 (seeded); val/test capped at 30 images per identity for balance
DigiFace-1M (synthetic faces, pre-training only)
  6. index the extracted images (already tightly cropped & aligned by the dataset authors), resize to 112

Usage:  python dataset_preparation.py [--quick] [--skip-digiface]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common.align import FaceAligner
from common.io import DATA, DIGIFACE_META, LFW_ALIGNED, LFW_META, PROCESSED, RAW, SPLITS_JSON, YUNET, save_json


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def phash_int(img_path) -> int:
    return int(str(imagehash.phash(Image.open(img_path), hash_size=8)), 16)


def near_duplicate_pairs(hashes: np.ndarray, max_dist: int):
    """All index pairs (i<j) whose 64-bit pHash Hamming distance <= max_dist. Vectorised in chunks."""
    h = hashes.astype(np.uint64)
    pairs, dists_sample = [], []
    for s in range(0, len(h), 512):
        x = np.bitwise_xor(h[s:s + 512, None], h[None, :])
        d = np.unpackbits(x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8), axis=2).sum(2)
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
        dists_sample.append(d[:, ::7].ravel())
    return pairs, np.concatenate(dists_sample)


def prepare_lfw(args):
    lfw_dir = Path(args.lfw_dir)
    people = sorted(p for p in lfw_dir.iterdir() if p.is_dir())
    people = [p for p in people if len(list(p.glob("*.jpg"))) >= args.min_images]
    if args.quick:
        people = people[:80]
    print(f"[lfw] identities with >= {args.min_images} images: {len(people)}")

    aligner = FaceAligner(args.yunet)
    rows, status = [], Counter()
    for pid, person in enumerate(tqdm(people, desc="align"), start=1):
        out_dir = LFW_ALIGNED / f"person_{pid:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, src in enumerate(sorted(person.glob("*.jpg")), start=1):
            dst = out_dir / f"img_{k:02d}.jpg"
            img = cv2.imread(str(src))
            crop, info = aligner.align(img)
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            status[info["status"]] += 1
            rows.append(dict(path=dst.relative_to(DATA).as_posix(), identity=f"person_{pid:03d}",
                             orig_name=person.name, orig_file=src.name, det_status=info["status"],
                             det_score=round(info["score"], 4), n_faces=info["n_faces"]))
    df = pd.DataFrame(rows)
    print(f"[lfw] images: {len(df)}  detection: {dict(status)}")

    # ---- near-duplicate detection on the ORIGINAL full photos (background + framing included).
    # Hashing the aligned crops is a poor duplicate detector: every aligned face shares the same layout, so
    # different people collide at small Hamming distances (verified visually - see README).
    df["phash"] = [phash_int(lfw_dir / n / f) for n, f in tqdm(zip(df.orig_name, df.orig_file), desc="phash",
                                                                 total=len(df))]
    cand, dist_sample = near_duplicate_pairs(df.phash.to_numpy(np.uint64), args.dup_hamming)
    # stage 2: confirm each candidate with a pixel-level check (normalised correlation of 64x64 grey thumbnails
    # of the full photos). Same photo re-encoded / re-cropped -> corr >= 0.93; different photos -> < 0.92.
    thumbs = {}

    def thumb(i):
        if i not in thumbs:
            g = cv2.resize(cv2.imread(str(lfw_dir / df.orig_name[i] / df.orig_file[i]), cv2.IMREAD_GRAYSCALE),
                           (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            g -= g.mean()
            thumbs[i] = g / (np.linalg.norm(g) + 1e-8)
        return thumbs[i]

    pairs = [(i, j, d) for i, j, d in cand if float(thumb(i) @ thumb(j)) >= args.dup_corr]
    print(f"[lfw] pHash candidates (hamming <= {args.dup_hamming}): {len(cand)} -> confirmed duplicates: {len(pairs)}")
    for i, j, d in pairs:
        print(f"        duplicate: {df.orig_file[i]} ~ {df.orig_file[j]} (hamming {d})")
    uf_img = UnionFind(range(len(df)))
    for i, j, _ in pairs:
        uf_img.union(i, j)
    df["dup_cluster"] = [uf_img.find(i) for i in range(len(df))]
    cross_id = [(i, j, d) for i, j, d in pairs if df.identity[i] != df.identity[j]]
    print(f"[lfw] confirmed duplicate pairs: {len(pairs)}  "
          f"cross-identity: {len(cross_id)}  median pHash distance overall: {np.median(dist_sample):.0f}")

    # ---- identity groups: identities that share a near-duplicate must land in the same split
    uf_id = UnionFind(df.identity.unique())
    for i, j, _ in cross_id:
        uf_id.union(df.identity[i], df.identity[j])
    groups = {}
    for ident in df.identity.unique():
        groups.setdefault(uf_id.find(ident), []).append(ident)
    group_list = sorted(groups.values())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(group_list)
    n_ids = df.identity.nunique()
    target = {"train": args.split[0] * n_ids, "val": args.split[1] * n_ids}
    split_of, counts = {}, Counter()
    for g in group_list:
        s = "train" if counts["train"] < target["train"] else "val" if counts["val"] < target["val"] else "test"
        for ident in g:
            split_of[ident] = s
        counts[s] += len(g)
    df["split"] = df.identity.map(split_of)

    # ---- cap eval identities (val/test) at N images so no single person dominates the pairs
    df["in_eval"] = df.split != "train"
    for ident, sub in df[df.split != "train"].groupby("identity"):
        if len(sub) > args.eval_cap:
            drop = sub.sample(len(sub) - args.eval_cap, random_state=args.seed).index
            df.loc[drop, "in_eval"] = False
    df.loc[df.split == "train", "in_eval"] = False
    df["phash"] = df.phash.map(lambda v: f"{int(v):016x}")
    df.to_csv(LFW_META, index=False)
    pd.DataFrame(cross_id, columns=["i", "j", "hamming"]).assign(
        a=lambda d: df.path[d.i].values, b=lambda d: df.path[d.j].values).to_csv(
        PROCESSED / "lfw_cross_identity_duplicates.csv", index=False)

    summary = {"source": "LFW (Labeled Faces in the Wild), http://vis-www.cs.umass.edu/lfw/",
               "min_images_per_identity": args.min_images, "identities": int(n_ids), "images": int(len(df)),
               "detection": dict(status), "near_duplicate_pairs": len(pairs),
               "cross_identity_near_duplicates": len(cross_id), "seed": args.seed, "splits": {}}
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        ev = sub[sub.in_eval] if s != "train" else sub
        summary["splits"][s] = {"identities": int(sub.identity.nunique()), "images": int(len(sub)),
                                "images_used": int(len(ev)),
                                "images_per_identity_min_median_max": [int(x) for x in ev.groupby("identity").size()
                                                                        .agg(["min", "median", "max"])]}
    ids = {s: set(df[df.split == s].identity) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    return summary


def prepare_digiface(args):
    roots = [Path(d) for d in args.digiface_dir if Path(d).exists()]
    if not roots:
        print(f"[digiface] {args.digiface_dir} not found - skipping (pre-training stage will be unavailable)")
        return None
    rows = []
    subj_dirs = sorted((p for r in roots for p in r.iterdir() if p.is_dir() and p.name.isdigit()),
                       key=lambda p: int(p.name))
    if args.max_digiface_ids:
        subj_dirs = subj_dirs[:args.max_digiface_ids]
    if args.quick:
        subj_dirs = subj_dirs[:200]
    sizes = Counter()
    for d in tqdm(subj_dirs, desc="digiface index"):
        files = sorted(d.glob("*.png"))
        for f in files:
            rows.append(dict(path=f.relative_to(DATA).as_posix(), label=int(d.name)))
        if files and len(sizes) < 5:
            sizes[Image.open(files[0]).size] += 1
    df = pd.DataFrame(rows)
    df.to_csv(DIGIFACE_META, index=False)
    info = {"source": "DigiFace-1M (Bae et al., WACV 2023), https://github.com/microsoft/DigiFace1M",
            "license": "Microsoft Research Use of Data Agreement v1.0 (non-commercial research only)",
            "identities": int(df.label.nunique()), "images": int(len(df)),
            "native_image_sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}}
    print(f"[digiface] {info}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw-dir", default=str(RAW / "lfw"))
    ap.add_argument("--digiface-dir", nargs="+", default=[str(RAW / "digiface" / "extracted"),
                                                          str(RAW / "digiface" / "extracted_more")],
                    help="one or more folders of extracted DigiFace-1M parts (subject-id sub-folders)")
    ap.add_argument("--max-digiface-ids", type=int, default=None)
    ap.add_argument("--yunet", default=str(YUNET))
    ap.add_argument("--min-images", type=int, default=4)
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    ap.add_argument("--eval-cap", type=int, default=30)
    ap.add_argument("--dup-hamming", type=int, default=8)
    ap.add_argument("--dup-corr", type=float, default=0.93)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny subset for a smoke test")
    ap.add_argument("--skip-digiface", action="store_true")
    ap.add_argument("--skip-lfw", action="store_true")
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SPLITS_JSON.read_text()) if SPLITS_JSON.exists() else {}
    if not args.skip_lfw:
        summary["lfw"] = prepare_lfw(args)
    if not args.skip_digiface:
        summary["digiface"] = prepare_digiface(args)
    save_json(summary, SPLITS_JSON)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
