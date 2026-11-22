"""Pair generation: uniqueness, no self pairs, balance, labels, split-locality (no leakage)."""
import numpy as np
import pandas as pd
import pytest

from common.io import LFW_META, PAIRS_DIR
from generate_pairs import allocate_quota, make_pairs


def toy_df():
    rows = []
    for i in range(30):
        for k in range(4 + (i % 5) * 6):
            rows.append(dict(path=f"p{i}/img{k}.jpg", identity=f"id{i}", dup_cluster=i * 1000 + k))
    rows.append(dict(path="p0/dup.jpg", identity="id0", dup_cluster=0))   # near-duplicate of p0/img0
    return pd.DataFrame(rows)


def test_quota_respects_caps_and_total():
    q = allocate_quota({"a": 6, "b": 1000, "c": 0}, 500)
    assert q["a"] <= 6 and q["c"] == 0 and sum(q.values()) == 500


def test_toy_pairs_valid():
    df = toy_df()
    pairs, stats = make_pairs(df, 800, seed=0)
    assert stats["positives"] == stats["negatives"]
    assert not pairs.duplicated(["img1", "img2"]).any()
    assert (pairs.img1 != pairs.img2).all()
    assert (pairs.img1 < pairs.img2).all()                                   # canonical order
    assert ((pairs.id1 == pairs.id2) == (pairs.label == 1)).all()
    dup_pair = {("p0/dup.jpg", "p0/img0.jpg")}
    assert not dup_pair & set(zip(pairs.img1, pairs.img2))                   # near-duplicates never paired


def test_deterministic():
    a, _ = make_pairs(toy_df(), 300, seed=5)
    b, _ = make_pairs(toy_df(), 300, seed=5)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.skipif(not (PAIRS_DIR / "pairs_test.csv").exists(), reason="run generate_pairs.py first")
@pytest.mark.parametrize("split", ["val", "test"])
def test_real_pairs_have_no_train_images(split):
    meta = pd.read_csv(LFW_META)
    pairs = pd.read_csv(PAIRS_DIR / f"pairs_{split}.csv")
    used = set(pairs.img1) | set(pairs.img2)
    train_paths = set(meta[meta.split == "train"].path)
    assert not used & train_paths
    split_of = dict(zip(meta.path, meta.split))
    assert {split_of[p] for p in used} == {split}
    assert not pairs.duplicated(["img1", "img2"]).any()
    assert abs(pairs.label.mean() - 0.5) < 1e-9
