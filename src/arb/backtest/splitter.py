"""Walk-forward index splitter."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: int  # inclusive
    train_end: int    # exclusive
    test_start: int   # inclusive
    test_end: int     # exclusive


def walk_forward_splits(
    n: int,
    train_window: int,
    test_window: int,
    step: int,
) -> list[Fold]:
    """Expanding-train walk-forward splits over an integer index of length n."""
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError("windows must be positive")
    if train_window + test_window > n:
        return []

    folds: list[Fold] = []
    fold_id = 0
    train_start = 0
    train_end = train_window
    while train_end + test_window <= n:
        test_start = train_end
        test_end = test_start + test_window
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        train_end += step  # expanding train, fixed test
    return folds
