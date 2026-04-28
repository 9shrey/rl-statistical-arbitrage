from __future__ import annotations

import pytest

from arb.backtest.splitter import walk_forward_splits


def test_walk_forward_basic_shape() -> None:
    folds = walk_forward_splits(n=100, train_window=40, test_window=20, step=20)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_start < f.train_end <= f.test_start < f.test_end
        assert f.test_end - f.test_start == 20


def test_walk_forward_no_overlap_train_test() -> None:
    folds = walk_forward_splits(n=200, train_window=60, test_window=30, step=30)
    for f in folds:
        assert f.train_end == f.test_start  # adjacency, no overlap


def test_walk_forward_validates_inputs() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(n=100, train_window=0, test_window=10, step=10)


def test_walk_forward_returns_empty_when_too_short() -> None:
    folds = walk_forward_splits(n=10, train_window=100, test_window=20, step=20)
    assert folds == []
