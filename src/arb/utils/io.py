"""IO helpers: JSON / CSV / parquet hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_array(arr: np.ndarray, decimals: int = 8) -> str:
    """Stable, content-based hash of a numeric array (rounded for float noise)."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()
