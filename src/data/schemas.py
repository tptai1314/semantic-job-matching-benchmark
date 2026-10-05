"""Định nghĩa cột và kiểm tra schema của bảng cặp (job, resume)."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "pair_id",
    "job_id",
    "resume_id",
    "resume_text",
    "job_text",
    "label",
    "relevance",
    "source_split",
]


class SchemaError(ValueError):
    """Bảng không đúng schema bắt buộc."""


def check_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    missing = [c for c in (required or REQUIRED_COLUMNS) if c not in df.columns]
    if missing:
        raise SchemaError(f"Thieu cot: {missing}")


def build_pairs(
    resume_text: pd.Series,
    job_text: pd.Series,
    label_raw: pd.Series,
    label_map: dict[str, int],
    relevance: dict[int, int],
    source_split: pd.Series | None = None,
    hashing_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Dựng bảng cặp chuẩn, tự sinh id ổn định từ nội dung văn bản.

    Dataset HuggingFace không có sẵn `job_id`/`resume_id`, nên id được sinh bằng
    hash nội dung: cùng văn bản luôn cho cùng id, và không phụ thuộc thứ tự dòng.
    """
    hashing_cfg = hashing_cfg or {"algorithm": "sha1", "length": 16}
    if len(resume_text) != len(job_text) or len(resume_text) != len(label_raw):
        raise SchemaError("Do dai resume/job/label khong bang nhau")

    unknown = sorted(set(label_raw.unique()) - set(label_map))
    if unknown:
        raise SchemaError(f"Nhan khong nam trong label_map: {unknown}")

    df = pd.DataFrame(
        {
            "resume_text": resume_text.astype(str),
            "job_text": job_text.astype(str),
            "label": label_raw.astype(str),
        }
    )
    df["label"] = df["label"].map(label_map).astype("int8")
    df["relevance"] = df["label"].map(relevance).astype("int8")

    from src.data.ids import text_id

    df["job_id"] = text_id(df["job_text"], hashing_cfg)
    df["resume_id"] = text_id(df["resume_text"], hashing_cfg)
    df["pair_id"] = text_id(df["job_id"] + "\x1f" + df["resume_id"], hashing_cfg)
    df["source_split"] = (
        source_split.astype(str).to_numpy() if source_split is not None else "unknown"
    )

    ordered = [c for c in REQUIRED_COLUMNS if c in df.columns] + [
        c for c in df.columns if c not in REQUIRED_COLUMNS
    ]
    return df[ordered]
