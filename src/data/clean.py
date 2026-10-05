"""Làm sạch bảng cặp: rút gọn khoảng trắng, loại rỗng, gỡ trùng lặp."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_WS = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_text(value: str) -> str:
    """Chuẩn hoá khoảng trắng nhưng GIỮ nguyên cấu trúc dòng.

    Văn bản CV/JD có heading dạng `Summary...` nối liền; nếu gộp toàn bộ
    xuống một dòng thì heading_regex của B3h mất dấu hiệu. Vì vậy chỉ thu gọn
    khoảng trắng ngang và rút gọn khoảng 3 dòng trở lên.
    """
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NEWLINES.sub("\n\n", text).strip()


def clean_pairs(df: pd.DataFrame, min_chars: int = 40) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chuẩn hoá, lọc rỗng và gỡ trùng. Trả về (df, report)."""
    out = df.copy()
    out["resume_text"] = out["resume_text"].map(normalize_text)
    out["job_text"] = out["job_text"].map(normalize_text)

    before = len(out)
    short_mask = (out["resume_text"].str.len() < min_chars) | (
        out["job_text"].str.len() < min_chars
    )
    out = out.loc[~short_mask].copy()

    n_short = int(short_mask.sum())
    dup_mask = out.duplicated(subset=["job_id", "resume_id"], keep="first")
    n_dup = int(dup_mask.sum())
    out = out.loc[~dup_mask].reset_index(drop=True)

    conflict = out.groupby(["job_id", "resume_id"])["label"].nunique().gt(1).sum()
    report = {
        "rows_in": int(before),
        "rows_dropped_short_text": n_short,
        "rows_dropped_duplicate_pair": n_dup,
        "rows_out": int(len(out)),
        "pairs_with_conflicting_labels": int(conflict),
        "min_chars": min_chars,
    }
    return out, report
