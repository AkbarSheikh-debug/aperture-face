"""Task 3 - positive / negative pair generation (run separately for the val and test splits).

Positives (label 1): two different images of the same identity.
  * the pool is every unordered intra-identity combination C(n_i, 2)
  * each identity gets a quota proportional to sqrt(pool size), capped at its pool size (leftover quota is
    redistributed) - large identities contribute more pairs, but cannot dominate the set
  * sampled WITHOUT replacement inside each identity
Negatives (label 0): images of two different identities.
  * identity pair drawn uniformly, then one image from each - identity-balanced impostors
Duplicate / leakage controls
  * every pair is stored in canonical order (min(path), max(path)) in a set -> no repeats, no (b, a) mirror
  * self-pairs are impossible; near-duplicate images (same pHash cluster) are never paired as a positive
    (that would be the same photo twice and inflate accuracy)
  * pairs are built only from images of ONE split; tests/test_pairs.py asserts none of them is a train image

Usage: python generate_pairs.py [--n 5000] [--seed 42]
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common.data import lfw_split
from common.io import PAIRS_DIR, RESULTS, update_json


def allocate_quota(pool_sizes: dict, total: int) -> dict:
    """sqrt-proportional quota with caps; leftover redistributed until filled or every pool is exhausted."""
    quota = {k: 0 for k in pool_sizes}
    remaining = total
    active = {k for k, v in pool_sizes.items() if v > 0}
    while remaining > 0 and active:
        w = {k: np.sqrt(pool_sizes[k]) for k in active}
        z = sum(w.values())
        added = 0
        for k in sorted(active):
            room = pool_sizes[k] - quota[k]
            give = min(room, max(1, int(round(remaining * w[k] / z))))
            quota[k] += give
            added += give
        remaining = total - sum(quota.values())
        if remaining < 0:                                   # rounding overshoot: trim from the largest quotas
            for k in sorted(quota, key=quota.get, reverse=True):
                if remaining == 0:
                    break
                cut = min(quota[k] - 1, -remaining) if quota[k] > 1 else 0
                quota[k] -= cut
                remaining += cut
        active = {k for k in active if quota[k] < pool_sizes[k]}
        if added == 0:
            break
    return quota


def make_pairs(df: pd.DataFrame, n_target: int, seed: int):
    rng = np.random.default_rng(seed)
    by_id = {k: g for k, g in df.groupby("identity")}

    # ---- positives
    pools = {}
    for ident, g in by_id.items():
        recs = list(zip(g.path, g.dup_cluster))
        pools[ident] = [(a[0], b[0]) for a, b in itertools.combinations(recs, 2) if a[1] != b[1]]
    quota = allocate_quota({k: len(v) for k, v in pools.items()}, n_target)
    pos = set()
    for ident in sorted(pools):
        pool = pools[ident]
        if quota[ident] == 0:
            continue
        for i in rng.choice(len(pool), quota[ident], replace=False):
            a, b = pool[i]
            pos.add((min(a, b), max(a, b), ident, ident))

    # ---- negatives (same count as positives -> balanced)
    n_neg = len(pos)
    ids = sorted(by_id)
    paths = {k: g.path.to_numpy() for k, g in by_id.items()}
    neg, tries = set(), 0
    while len(neg) < n_neg and tries < 50 * n_neg:
        tries += 1
        i1, i2 = rng.choice(len(ids), 2, replace=False)
        a = rng.choice(paths[ids[i1]]); b = rng.choice(paths[ids[i2]])
        ka, kb = (a, ids[i1]), (b, ids[i2])
        (a, ia), (b, ib) = sorted([ka, kb])
        neg.add((a, b, ia, ib))

    rows = [dict(img1=a, img2=b, id1=x, id2=y, label=1) for a, b, x, y in sorted(pos)]
    rows += [dict(img1=a, img2=b, id1=x, id2=y, label=0) for a, b, x, y in sorted(neg)]
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    stats = {"positives": len(pos), "negatives": len(neg), "identities": len(ids), "images": int(len(df)),
             "positive_pool_size": int(sum(len(v) for v in pools.values())),
             "identities_in_positives": int(sum(1 for v in quota.values() if v > 0)),
             "max_positive_pairs_one_identity": int(max(quota.values()))}
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="target positives (= negatives)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for split, seed_off in (("val", 1), ("test", 2)):
        df = lfw_split(split)
        pairs, stats = make_pairs(df, args.n, args.seed + seed_off)
        assert not pairs.duplicated(["img1", "img2"]).any()
        assert (pairs.img1 != pairs.img2).all()
        pairs.to_csv(PAIRS_DIR / f"pairs_{split}.csv", index=False)
        all_stats[split] = stats
        print(f"[pairs] {split}: {stats}")
    update_json(RESULTS / "evaluation_results.json", "pairs", all_stats)


if __name__ == "__main__":
    main()
