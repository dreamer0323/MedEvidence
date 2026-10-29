"""混合检索：分词 + RRF 融合。"""
from __future__ import annotations

import pytest

from src.retrieve.bm25 import tokenize
from src.retrieve.hybrid import rrf_fuse


# --------------------------------------------------------------- 分词

def test_tokenize_splits_hyphens():
    assert tokenize("Whole-Slide Image") == ["whole", "slide", "image"]


def test_tokenize_lowercases_and_keeps_entity_ids():
    assert tokenize("LH510 GT450 S360") == ["lh510", "gt450", "s360"]


def test_tokenize_drops_stopwords():
    tokens = tokenize("the scanner of the slide")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "scanner" in tokens


def test_tokenize_keeps_numbers():
    assert tokenize("0.26 um/pixel") == ["0", "26", "um", "pixel"]


# --------------------------------------------------------------- RRF

def test_rrf_fuse_basic_scores():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    scores = dict(fused)

    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_fuse_tie_breaks_by_id():
    # a 和 b 分数相同 → 按 id 升序
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    assert [item[0] for item in fused] == ["a", "b"]


def test_rrf_fuse_rank_order():
    fused = rrf_fuse([["a", "b", "c"]], k=60)
    assert [item[0] for item in fused] == ["a", "b", "c"]
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_fuse_keeps_ids_from_single_list():
    fused = rrf_fuse([["a"], ["b"]], k=60)
    ids = [item[0] for item in fused]

    assert set(ids) == {"a", "b"}
    assert len(ids) == 2


def test_rrf_fuse_promotes_items_in_both_lists():
    # c 只在一路出现；a 在两路都靠前 → a 必须排在 c 前面
    fused = rrf_fuse([["a", "c"], ["a"]], k=60)
    assert fused[0][0] == "a"
