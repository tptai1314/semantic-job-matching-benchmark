"""Tải dataset gốc từ HuggingFace và gộp train/test."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import get


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Tải dataset gốc, gộp mọi split thành một bảng cặp thô.

    Bước 2.2 của PLAN: gộp train (6241) + test (1759) = 8000 cặp, sau đó mới
    chia fold mới. Nếu không gộp, CV trong split test chính thức đã nằm trong
    train (476/477 = 99,8%), khiến mọi kết luận về generalization không đáng tin.
    """
    from datasets import load_dataset

    repo = get(cfg, "data.hf_dataset")
    revision = get(cfg, "data.dataset_revision")
    ds = load_dataset(repo, revision=revision)

    frames = []
    for split_name, hf_split in ds.items():
        frame = hf_split.to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    cols = get(cfg, "data.columns")
    rename = {
        cols["resume"]: "resume_text",
        cols["job"]: "job_text",
        cols["label"]: "label",
    }
    raw = raw.rename(columns=rename)
    missing = [c for c in ("resume_text", "job_text", "label") if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset thieu cot: {missing}. Co san: {list(raw.columns)}")
    return raw[["resume_text", "job_text", "label", "source_split"]]


def dataset_fingerprint(cfg: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    """Thông tin để ghi vào docs/research_log.md và tái lập sau này."""
    import datasets

    return {
        "hf_dataset": get(cfg, "data.hf_dataset"),
        "dataset_revision": get(cfg, "data.dataset_revision"),
        "datasets_version": datasets.__version__,
        "n_rows": int(len(raw)),
        "n_resumes": int(raw["resume_text"].nunique()),
        "n_jobs": int(raw["job_text"].nunique()),
        "label_counts": {str(k): int(v) for k, v in raw["label"].value_counts().items()},
        "rows_per_source_split": {
            str(k): int(v) for k, v in raw["source_split"].value_counts().items()
        },
    }
