"""Identity-disjoint splits and duplicate isolation on the real prepared dataset."""
import pandas as pd
import pytest

from common.io import LFW_META

pytestmark = pytest.mark.skipif(not LFW_META.exists(), reason="run dataset_preparation.py first")


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(LFW_META)


def test_identity_disjoint(meta):
    ids = {s: set(meta[meta.split == s].identity) for s in ("train", "val", "test")}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_near_duplicates_never_cross_splits(meta):
    assert (meta.groupby("dup_cluster").split.nunique() == 1).all()


def test_min_images_and_unique_paths(meta):
    assert meta.groupby("identity").size().min() >= 4
    assert meta.path.is_unique


def test_requirements_met(meta):
    assert meta.identity.nunique() >= 50 and len(meta) >= 500
