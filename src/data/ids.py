"""Sinh id ổn định cho văn bản và cho cặp (job, resume)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd


def _digest(text: str, algorithm: str, length: int) -> str:
    h = hashlib.new(algorithm, text.encode("utf-8"))
    return h.hexdigest()[:length]


def hash_texts(texts: Iterable[str], algorithm: str = "sha1", length: int = 16) -> pd.Series:
    return pd.Series([_digest(t, algorithm, length) for t in texts], dtype="object")


def text_id(texts: pd.Series, hashing_cfg: dict | None = None) -> pd.Series:
    """Trả về Series id 16 ký tự hex, index khớp với `texts`."""
    cfg = hashing_cfg or {}
    algorithm = cfg.get("algorithm", "sha1")
    length = int(cfg.get("length", 16))
    return hash_texts(texts.astype(str), algorithm, length)
