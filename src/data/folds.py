"""Chia fold theo nhóm và đo rò rỉ giữa các fold.

Hai hệ split bắt buộc (PLAN 2.10):
  - `job_id`      : cặp cùng JD không nằm ở hai fold (split chính, đánh giá JD mới).
  - `resume_id`   : CV không nằm ở hai fold (split phụ, đánh giá CV mới).

Máy chỉ có 351 JD và 643 CV, nên cả hai hệ đều bị giới hạn bởi số nhóm; bảng audit
được ghi ra để báo cáo `n_jd` và `n_cv` kèm theo mọi bảng kết quả.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_group_folds(
    df: pd.DataFrame, group_col: str, n_splits: int, seed: int = 42
) -> pd.DataFrame:
    """Gán `fold` cho từng dòng theo GroupKFold trên `group_col`."""
    if group_col not in df.columns:
        raise ValueError(f"Thieu cot nhom: {group_col}")
    n_groups = df[group_col].nunique()
    if n_groups < n_splits:
        raise ValueError(f"Chi {n_groups} nhom '{group_col}' < n_splits={n_splits}")

    groups = df[group_col].to_numpy()
    fold_ids = np.full(len(df), -1, dtype=np.int8)
    for fold, (_, idx) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        fold_ids[idx] = fold

    out = df.copy()
    out["fold"] = fold_ids
    out["split_seed"] = seed
    out.attrs["group_col"] = group_col
    return out


def audit_leakage(df: pd.DataFrame, group_col: str, fold_col: str = "fold") -> pd.DataFrame:
    """Đếm JD / CV / cặp bị chia nhóm giữa các fold.

    Với GroupKFold trên `group_col`, số nhóm dùng chung giữa các fold phải bằng 0.
    Cột còn lại đo mức rò rỉ chéo (ví dụ split theo `job_id` nhưng CV vẫn lặp).
    """
    if group_col not in df.columns:
        raise ValueError(f"Thieu cot nhom: {group_col}")

    other = "resume_id" if group_col == "job_id" else "job_id"
    folds = sorted(df[fold_col].unique())

    rows = []
    for f in folds:
        val = df.loc[df[fold_col] == f]
        train = df.loc[df[fold_col] != f]
        shared_groups = set(val[group_col]) & set(train[group_col])
        shared_other = set(val[other]) & set(train[other])
        val_jd, val_cv = set(val["job_id"]), set(val["resume_id"])
        tr_jd, tr_cv = set(train["job_id"]), set(train["resume_id"])

        rows.append(
            {
                "group_by": group_col,
                "fold": int(f),
                "n_val_rows": int(len(val)),
                "n_val_pairs_unique": len(val_jd) * len(val_cv),
                "n_train_rows": int(len(train)),
                "n_jd_val": len(val_jd),
                "n_cv_val": len(val_cv),
                "n_jd_train": len(tr_jd),
                "n_cv_train": len(tr_cv),
                f"jd_shared_train_val": len(val_jd & tr_jd),
                f"cv_shared_train_val": len(val_cv & tr_cv),
                f"{group_col}_shared_train_val": len(shared_groups),
                f"{other}_shared_train_val": len(shared_other),
                "cv_shared_pct_of_val": round(100 * len(val_cv & tr_cv) / len(val_cv), 2),
            }
        )
    return pd.DataFrame(rows)


def split_summary(df: pd.DataFrame, group_col: str, fold_col: str = "fold") -> dict[str, Any]:
    """Thống kê tổng quan một hệ split: độ cân bằng, số nhóm, nhãn."""
    per_fold = []
    for f in sorted(df[fold_col].unique()):
        val = df.loc[df[fold_col] == f]
        per_fold.append(
            {
                "fold": int(f),
                "n_rows": int(len(val)),
                "n_jd": int(val["job_id"].nunique()),
                "n_cv": int(val["resume_id"].nunique()),
                "label_mean": round(float(val["label"].mean()), 4),
            }
        )
    counts = pd.Series([p["n_rows"] for p in per_fold])
    return {
        "group_by": group_col,
        "n_splits": len(per_fold),
        "n_groups_total": int(df[group_col].nunique()),
        "rows_min": int(counts.min()),
        "rows_max": int(counts.max()),
        "rows_ratio_max_min": round(float(counts.max() / counts.min()), 3),
        "jd_per_fold_min": min(p["n_jd"] for p in per_fold),
        "cv_per_fold_min": min(p["n_cv"] for p in per_fold),
        "per_fold": per_fold,
    }


def write_folds(df: pd.DataFrame, out_dir: str | Path, prefix: str = "fold_") -> list[Path]:
    """Ghi mỗi fold thành file json: tập cặp train/val và danh sách nhóm."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in sorted(df["fold"].unique()):
        val = df.loc[df["fold"] == f]
        payload = {
            "fold": int(f),
            "group_by": df.attrs.get("group_col"),
            "n_train_rows": int(len(df) - len(val)),
            "n_val_rows": int(len(val)),
            "val_job_ids": sorted(set(val["job_id"])),
            "val_resume_ids": sorted(set(val["resume_id"])),
            "val_pair_ids": val["pair_id"].tolist(),
        }
        path = out / f"{prefix}{int(f)}.json"
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        written.append(path)
    return written


def attach_fold(
    pairs: pd.DataFrame,
    folds: pd.DataFrame,
    key: str = "pair_id",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Ghép cột fold của một hệ split vào bảng cặp gốc theo `pair_id`.

    `fold_col` là tên cột đích, nên ghép nhiều hệ split sẽ cho `fold` (chính) và
    `fold_resume_id` (phụ) mà không đụng tên cột đã có.
    """
    keep = folds[[key, "fold"]].rename(columns={"fold": fold_col})
    if keep.duplicated(subset=[key]).any():
        raise ValueError(f"Mot cap '{key}' biet nhieu fold trong he split nay")
    return pairs.merge(keep, on=key, how="inner", validate="one_to_one")
